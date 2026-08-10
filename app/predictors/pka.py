"""
pKa Prediction — v9: Routed ensemble (v7 + v8)

Routes queries to the best model based on molecular substructure:
- Sulfonamides / aromatic N-H → v8 ensemble (per-atom charges, 5 seeds)
- Everything else → v7 ensemble (no charges, 6 seeds)

Both v7 and v8 use per-atom prediction heads trained on microscopic + macroscopic
data with mixed loss. v9 achieves scaffold RMSE 1.255 and SAMPL7 RMSE 0.535.

Fallback chain: v9 routed → v4 (v2/v3 ensembles) → v1 → empirical.
"""

import logging
import os
from typing import Optional
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger("novomcp-properties.pka")

# Model holders
_v7_models: list = []   # v7 ensemble (no charges, per-atom head)
_v8_models: list = []   # v8 ensemble (full charges, per-atom head)
_v7_featurizer = None
_v8_featurizer = None
_v2_models: list = []   # v4 fallback: v2 ensemble (global xTB)
_v3_models: list = []   # v4 fallback: v3 ensemble (per-atom charges)
_v1_model = None         # v1 fallback (no xTB)
_backend: Optional[str] = None

# A "trained" backend is one backed by real model weights. "rdkit-empirical" is
# NOT trained — it's a crude SMARTS estimator, and serving it silently when the
# weights failed to load would hand a self-hoster low-quality pKa without any
# signal. We treat that case as unavailable unless explicitly opted into.
_TRAINED_BACKENDS = ("chemprop-v9-routed", "chemprop-v4-routed", "chemprop-v1")


def _weights_loaded() -> bool:
    return _backend in _TRAINED_BACKENDS


def _empirical_allowed() -> bool:
    """Opt-in escape hatch: serve clearly-labeled empirical estimates on purpose."""
    return os.getenv("PKA_ALLOW_EMPIRICAL", "").strip().lower() in ("1", "true", "yes")


def is_available() -> bool:
    """True when trained weights loaded, or empirical serving is explicitly opted into."""
    return _weights_loaded() or _empirical_allowed()

# Routing SMARTS
_SULFONAMIDE_SMARTS = None
_AROMATIC_NH_SMARTS = None


class AtomPkaHead(nn.Module):
    """Per-atom FFN — must match training architecture."""

    def __init__(self, d_h, n_global=5, hidden=256, n_layers=3, dropout=0.15):
        super().__init__()
        layers = []
        in_dim = d_h + n_global
        for i in range(n_layers):
            layers.append(nn.Linear(in_dim if i == 0 else hidden, hidden))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(hidden, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, H_v, global_features, batch_idx):
        global_per_atom = global_features[batch_idx]
        combined = torch.cat([H_v, global_per_atom], dim=-1)
        return self.mlp(combined).squeeze(-1)


@dataclass
class PkaResult:
    smiles: str
    pka_values: list[float]
    ionizable_groups: list[str]
    method: str
    confidence: Optional[float] = None
    uncertainty: Optional[float] = None
    model_version: str = ""


def initialize():
    """Load v9 routed ensemble (v7 + v8). Falls back to v4/v1/empirical."""
    global _backend, _v7_models, _v8_models, _v7_featurizer, _v8_featurizer
    global _v2_models, _v3_models, _v1_model
    global _SULFONAMIDE_SMARTS, _AROMATIC_NH_SMARTS

    from rdkit import Chem
    _SULFONAMIDE_SMARTS = Chem.MolFromSmarts("[#16](=O)(=O)[NX3]")
    _AROMATIC_NH_SMARTS = Chem.MolFromSmarts("[nH]")

    # Load v9: v7 (no charges) + v8 (charges)
    v7_dir = os.getenv("PKA_V7_MODEL_DIR", "/app/models/pka_v9_nochg")
    v8_dir = os.getenv("PKA_V8_MODEL_DIR", "/app/models/pka_v9_chg")

    _v7_models, _v7_featurizer = _load_v9_ensemble(v7_dir, "v7-nochg")
    _v8_models, _v8_featurizer = _load_v9_ensemble(v8_dir, "v8-charges")

    if _v7_models or _v8_models:
        _backend = "chemprop-v9-routed"
        logger.info(f"pKa v9: {len(_v7_models)} v7 models, {len(_v8_models)} v8 models")
    else:
        # Fall back to v4
        _load_v4_fallback()

    return _backend is not None


def _load_v9_ensemble(ckpt_dir: str, label: str):
    """Load v7/v8 ensemble: BondMessagePassing + AtomPkaHead from .pt files."""
    from chemprop import nn as chemprop_nn, featurizers
    import glob

    if not os.path.isdir(ckpt_dir):
        logger.info(f"pKa {label}: directory not found: {ckpt_dir}")
        return [], None

    ckpt_files = sorted(glob.glob(os.path.join(ckpt_dir, "*.pt")))
    if not ckpt_files:
        logger.info(f"pKa {label}: no .pt files in {ckpt_dir}")
        return [], None

    # Peek at first checkpoint to detect charge mode
    peek = torch.load(ckpt_files[0], map_location="cpu")
    peek_args = peek.get("args", {})
    has_charges = peek_args.get("with_charges", False)

    extra_atom_fdim = 1 if has_charges else 0
    featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer(extra_atom_fdim=extra_atom_fdim)
    atom_fdim, bond_fdim = featurizer.shape

    models = []
    for path in ckpt_files:
        try:
            ckpt = torch.load(path, map_location="cpu")
            args = ckpt.get("args", {})

            mp = chemprop_nn.BondMessagePassing(
                d_v=atom_fdim, d_e=bond_fdim,
                d_h=args.get("hidden_size", 300),
                depth=args.get("depth", 4),
                dropout=args.get("dropout", 0.15),
            )
            mp.load_state_dict(ckpt["mp_state"])
            mp.eval()

            head = AtomPkaHead(
                d_h=args.get("hidden_size", 300), n_global=5,
                hidden=args.get("head_hidden", 256),
                n_layers=args.get("head_layers", 3),
                dropout=args.get("dropout", 0.15),
            )
            head.load_state_dict(ckpt["head_state"])
            head.eval()

            models.append((mp, head))
            logger.info(f"pKa {label}: loaded {os.path.basename(path)} "
                        f"(epoch={ckpt.get('epoch')}, val_mse={ckpt.get('val_mse', 0):.4f})")
        except Exception as e:
            logger.warning(f"pKa {label}: failed to load {path}: {e}")

    if models:
        logger.info(f"pKa {label}: {len(models)} models loaded (charges={has_charges})")

    return models, featurizer


def _load_v4_fallback():
    """Load v4 models as fallback if v9 not available."""
    global _backend, _v2_models, _v3_models, _v1_model

    from chemprop.models import MPNN
    import glob

    # v3 ensemble
    v3_dir = os.getenv("PKA_V3_MODEL_DIR", "/app/models/pka_v3_ensemble")
    if os.path.isdir(v3_dir):
        for path in sorted(glob.glob(os.path.join(v3_dir, "*.ckpt"))):
            try:
                m = MPNN.load_from_checkpoint(path, map_location="cpu")
                m.eval()
                _v3_models.append(m)
            except Exception:
                pass

    # v2 ensemble
    v2_dir = os.getenv("PKA_V2_MODEL_DIR", "/app/models/pka_v2_ensemble")
    if os.path.isdir(v2_dir):
        for path in sorted(glob.glob(os.path.join(v2_dir, "*.ckpt"))):
            try:
                m = MPNN.load_from_checkpoint(path, map_location="cpu")
                m.eval()
                _v2_models.append(m)
            except Exception:
                pass

    # v1 single
    v1_path = "/app/models/pka-v1/best_model.ckpt"
    if os.path.exists(v1_path):
        try:
            _v1_model = MPNN.load_from_checkpoint(v1_path, map_location="cpu")
            _v1_model.eval()
        except Exception:
            pass

    if _v2_models or _v3_models:
        _backend = "chemprop-v4-routed"
        logger.info(f"pKa v4 fallback: {len(_v2_models)} v2, {len(_v3_models)} v3")
    elif _v1_model:
        _backend = "chemprop-v1"
    else:
        _backend = "rdkit-empirical"
        if _empirical_allowed():
            logger.warning(
                "pKa: no trained weights found — serving RDKit empirical estimates "
                "because PKA_ALLOW_EMPIRICAL is set. Output is low-confidence and is "
                "labeled method='rdkit-empirical'."
            )
        else:
            logger.error(
                "pKa: trained model weights failed to load (no checkpoints found). "
                "The pKa endpoints will return 503 (model unavailable) rather than "
                "silently serving crude empirical estimates. Wire the weights, or set "
                "PKA_ALLOW_EMPIRICAL=1 to explicitly opt into labeled empirical output."
            )


def _should_route_to_charges(smiles: str) -> bool:
    """Check if molecule should be routed to v8 (charges) model."""
    from rdkit import Chem
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False
    if _SULFONAMIDE_SMARTS and mol.HasSubstructMatch(_SULFONAMIDE_SMARTS):
        return True
    if _AROMATIC_NH_SMARTS and mol.HasSubstructMatch(_AROMATIC_NH_SMARTS):
        return True
    return False


def predict(smiles: str) -> Optional[PkaResult]:
    """Predict pKa with smart routing and ensemble averaging."""
    if _backend is None:
        return None

    if _backend == "rdkit-empirical":
        return _predict_empirical(smiles)

    if _backend == "chemprop-v9-routed":
        return _predict_v9(smiles)

    if _backend == "chemprop-v4-routed":
        return _predict_v4(smiles)

    if _backend == "chemprop-v1":
        return _predict_v1(smiles)

    return _predict_empirical(smiles)


@torch.no_grad()
def _predict_v9(smiles: str) -> Optional[PkaResult]:
    """Predict using v9 routed ensemble (v7 + v8)."""
    try:
        from rdkit import Chem
        from chemprop import data as chemprop_data

        route_to_charges = _should_route_to_charges(smiles)

        if route_to_charges and _v8_models:
            models, featurizer = _v8_models, _v8_featurizer
            route_label = "v8-charges"
        elif _v7_models:
            models, featurizer = _v7_models, _v7_featurizer
            route_label = "v7-nochg"
        elif _v8_models:
            models, featurizer = _v8_models, _v8_featurizer
            route_label = "v8-charges"
        else:
            return _predict_v4(smiles)

        # Get xTB data
        xtb_data = _get_xtb_data(smiles)
        global_feats = np.array([
            xtb_data.get("energy_hartree", 0.0),
            xtb_data.get("homo_ev", 0.0),
            xtb_data.get("lumo_ev", 0.0),
            xtb_data.get("gap_ev", 0.0),
            xtb_data.get("dipole_debye", 0.0),
        ], dtype=np.float32)

        # Build datapoint
        has_charges = (route_label == "v8-charges")
        if has_charges:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return _predict_empirical(smiles)

            # Get V_f matching featurizer's atom count
            dp_probe = chemprop_data.MoleculeDatapoint.from_smi(smiles, x_d=global_feats)
            try:
                mg = featurizer(dp_probe.mol, dp_probe.V_f)
                n_feat = mg.V.shape[0]
            except Exception:
                n_feat = mol.GetNumHeavyAtoms()

            charges = xtb_data.get("partial_charges", [])
            mol_h = Chem.AddHs(mol)
            if charges and len(charges) == mol_h.GetNumAtoms():
                heavy_charges = [charges[i] for i in range(mol_h.GetNumAtoms())
                               if mol_h.GetAtomWithIdx(i).GetAtomicNum() > 1]
            else:
                heavy_charges = []

            if len(heavy_charges) == n_feat:
                V_f = np.array(heavy_charges, dtype=np.float32).reshape(-1, 1)
            elif len(charges) == n_feat:
                V_f = np.array(charges, dtype=np.float32).reshape(-1, 1)
            else:
                V_f = np.zeros((n_feat, 1), dtype=np.float32)

            dp = chemprop_data.MoleculeDatapoint.from_smi(smiles, x_d=global_feats)
            dp.V_f = V_f
        else:
            dp = chemprop_data.MoleculeDatapoint.from_smi(smiles, x_d=global_feats)

        ds = chemprop_data.MoleculeDataset([dp], featurizer)
        loader = chemprop_data.build_dataloader(ds, batch_size=1, shuffle=False, num_workers=0)

        batch = next(iter(loader))
        bmg = batch.bmg
        global_f = torch.tensor(global_feats).unsqueeze(0)

        # Per-atom predictions from all ensemble members
        all_preds = []
        for mp, head in models:
            H_v = mp(bmg)
            pka_all = head(H_v, global_f, bmg.batch)
            all_preds.append(pka_all.cpu().numpy())

        if not all_preds:
            return _predict_v4(smiles)

        stacked = np.stack(all_preds)
        means = stacked.mean(axis=0)
        stds = stacked.std(axis=0) if len(all_preds) > 1 else np.zeros_like(means)

        # Find ionizable atoms and their predictions
        ionizable_groups = _detect_ionizable_groups(smiles)
        ionizable_indices = _find_ionizable_atoms(smiles)

        pka_values = []
        uncertainties = []
        if ionizable_indices:
            for idx in ionizable_indices:
                if idx < len(means):
                    pka_values.append(round(float(means[idx]), 2))
                    uncertainties.append(float(stds[idx]))
        else:
            # No ionizable atoms detected — return best prediction
            best_idx = int(np.argmin(np.abs(means - 7.4)))  # Closest to physiological pH
            if best_idx < len(means):
                pka_values.append(round(float(means[best_idx]), 2))
                uncertainties.append(float(stds[best_idx]))

        if not pka_values:
            return _predict_empirical(smiles)

        mean_uncertainty = np.mean(uncertainties)
        confidence = _compute_confidence(mean_uncertainty, len(models))

        return PkaResult(
            smiles=smiles,
            pka_values=sorted(pka_values),
            ionizable_groups=ionizable_groups,
            method=f"chemprop-v9-{route_label}",
            confidence=confidence,
            uncertainty=round(float(mean_uncertainty), 3),
            model_version=f"v9-{route_label}-{len(models)}",
        )
    except Exception as e:
        logger.error(f"pKa v9 prediction failed for {smiles}: {e}")
        return _predict_v4(smiles)


def _predict_v4(smiles: str) -> Optional[PkaResult]:
    """Fallback to v4 routed ensemble."""
    from rdkit import Chem

    is_sulfonamide = False
    if _SULFONAMIDE_SMARTS:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            is_sulfonamide = mol.HasSubstructMatch(_SULFONAMIDE_SMARTS)

    if is_sulfonamide and _v3_models:
        return _predict_v4_v3(smiles)
    elif _v2_models:
        return _predict_v4_v2(smiles)
    elif _v1_model:
        return _predict_v1(smiles)
    return _predict_empirical(smiles)


def _predict_v4_v2(smiles: str) -> Optional[PkaResult]:
    """v4 fallback: v2 ensemble."""
    try:
        from chemprop import data as chemprop_data, featurizers
        import lightning as L

        featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()
        xtb_data = _get_xtb_data(smiles)
        x_d = np.array([
            xtb_data.get("energy_hartree", 0.0), xtb_data.get("homo_ev", 0.0),
            xtb_data.get("lumo_ev", 0.0), xtb_data.get("gap_ev", 0.0),
            xtb_data.get("dipole_debye", 0.0),
        ], dtype=np.float32)

        mol_data = [chemprop_data.MoleculeDatapoint.from_smi(smiles, x_d=x_d)]
        dataset = chemprop_data.MoleculeDataset(mol_data, featurizer)
        loader = chemprop_data.build_dataloader(dataset, batch_size=1, shuffle=False, num_workers=0)

        predictions = []
        trainer = L.Trainer(accelerator="cpu", devices=1, logger=False, enable_progress_bar=False)
        for model in _v2_models:
            with torch.inference_mode():
                preds = trainer.predict(model, loader)
            predictions.append(float(np.concatenate(preds, axis=0)[0][0]))

        mean_pka = np.mean(predictions)
        std_pka = np.std(predictions) if len(predictions) > 1 else None

        return PkaResult(
            smiles=smiles, pka_values=[round(float(mean_pka), 2)],
            ionizable_groups=_detect_ionizable_groups(smiles),
            method="chemprop-v4-v2route",
            confidence=_compute_confidence(std_pka, len(_v2_models)),
            uncertainty=round(float(std_pka), 3) if std_pka else None,
            model_version=f"v4-v2-{len(_v2_models)}",
        )
    except Exception as e:
        logger.error(f"pKa v4-v2 failed: {e}")
        return _predict_empirical(smiles)


def _predict_v4_v3(smiles: str) -> Optional[PkaResult]:
    """v4 fallback: v3 ensemble."""
    try:
        from rdkit import Chem
        from chemprop import data as chemprop_data, featurizers
        import lightning as L

        xtb_data = _get_xtb_data(smiles)
        charges = xtb_data.get("partial_charges", [])
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return _predict_empirical(smiles)
        mol_h = Chem.AddHs(mol)
        n_heavy = mol.GetNumHeavyAtoms()

        if charges and len(charges) == mol_h.GetNumAtoms():
            heavy_charges = [charges[i] for i in range(mol_h.GetNumAtoms())
                           if mol_h.GetAtomWithIdx(i).GetAtomicNum() > 1]
        else:
            heavy_charges = [0.0] * n_heavy

        V_f = np.array(heavy_charges, dtype=np.float32).reshape(-1, 1)
        x_d = np.array([
            xtb_data.get("energy_hartree", 0.0), xtb_data.get("homo_ev", 0.0),
            xtb_data.get("lumo_ev", 0.0), xtb_data.get("gap_ev", 0.0),
            xtb_data.get("dipole_debye", 0.0),
        ], dtype=np.float32)

        featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer(extra_atom_fdim=1)
        mol_data = [chemprop_data.MoleculeDatapoint.from_smi(smiles, V_f=V_f, x_d=x_d)]
        dataset = chemprop_data.MoleculeDataset(mol_data, featurizer)
        loader = chemprop_data.build_dataloader(dataset, batch_size=1, shuffle=False, num_workers=0)

        predictions = []
        trainer = L.Trainer(accelerator="cpu", devices=1, logger=False, enable_progress_bar=False)
        for model in _v3_models:
            with torch.inference_mode():
                preds = trainer.predict(model, loader)
            predictions.append(float(np.concatenate(preds, axis=0)[0][0]))

        mean_pka = np.mean(predictions)
        std_pka = np.std(predictions) if len(predictions) > 1 else None

        return PkaResult(
            smiles=smiles, pka_values=[round(float(mean_pka), 2)],
            ionizable_groups=_detect_ionizable_groups(smiles),
            method="chemprop-v4-v3route",
            confidence=_compute_confidence(std_pka, len(_v3_models)),
            uncertainty=round(float(std_pka), 3) if std_pka else None,
            model_version=f"v4-v3-{len(_v3_models)}",
        )
    except Exception as e:
        logger.error(f"pKa v4-v3 failed: {e}")
        return _predict_empirical(smiles)


def _predict_v1(smiles: str) -> Optional[PkaResult]:
    """v1 fallback (no xTB)."""
    try:
        from chemprop import data as chemprop_data, featurizers
        import lightning as L

        featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()
        mol_data = [chemprop_data.MoleculeDatapoint.from_smi(smiles)]
        dataset = chemprop_data.MoleculeDataset(mol_data, featurizer)
        loader = chemprop_data.build_dataloader(dataset, batch_size=1, shuffle=False, num_workers=0)

        trainer = L.Trainer(accelerator="cpu", devices=1, logger=False, enable_progress_bar=False)
        with torch.inference_mode():
            preds = trainer.predict(_v1_model, loader)
        pka_val = float(np.concatenate(preds, axis=0)[0][0])

        return PkaResult(
            smiles=smiles, pka_values=[round(pka_val, 2)],
            ionizable_groups=_detect_ionizable_groups(smiles),
            method="chemprop-v1", confidence=0.70, model_version="v1",
        )
    except Exception as e:
        logger.error(f"pKa v1 failed: {e}")
        return _predict_empirical(smiles)


def _find_ionizable_atoms(smiles: str) -> list[int]:
    """Find ionizable atom indices using SMARTS patterns."""
    from rdkit import Chem
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return []

    patterns = [
        "[NH]S(=O)(=O)", "[NH2]S(=O)(=O)", "[nH]",
        "C(=O)[OH]", "c[OH]",
        "[NH2;!$([NH2]S(=O))]", "[NH;!$([NH]S(=O));!$([nH])]",
        "[SH]", "[OH;!$([OH]C=O);!$([OH]c)]",
    ]
    ionizable = set()
    for smarts in patterns:
        pat = Chem.MolFromSmarts(smarts)
        if pat:
            for match in mol.GetSubstructMatches(pat):
                ionizable.add(match[0])
    return sorted(ionizable)


def _compute_confidence(std: Optional[float], n_models: int) -> float:
    if n_models <= 1 or std is None:
        return 0.75
    if std < 0.5:
        return 0.90
    if std < 1.5:
        return 0.75
    return 0.50


def _get_xtb_data(smiles: str) -> dict:
    """Get xTB data (global features + per-atom charges) from novomcp-qm.

    Delegates to the shared _xtb_client module. The return format is
    identical to the original inline implementation: dict with
    energy_hartree, homo_ev, lumo_ev, gap_ev, dipole_debye, partial_charges.
    """
    from app.predictors._xtb_client import get_xtb_data
    return get_xtb_data(smiles)


def _predict_empirical(smiles: str) -> Optional[PkaResult]:
    """Empirical pKa estimation using RDKit functional group detection."""
    try:
        from rdkit import Chem
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        groups = _detect_ionizable_groups(smiles)
        pka_map = {
            "carboxylic_acid": 4.2, "phenol": 10.0, "primary_amine": 10.6,
            "secondary_amine": 10.8, "tertiary_amine": 9.8, "sulfonamide": 10.0,
            "thiol": 8.3, "imidazole": 6.0, "pyridine": 5.2, "tetrazole": 4.9,
            "phosphate": 2.1, "guanidine": 13.5,
        }
        pka_values = [pka_map[g] for g in groups if g in pka_map]
        if not pka_values:
            pka_values = [7.0]
            groups = ["none_detected"]

        return PkaResult(
            smiles=smiles, pka_values=[round(v, 2) for v in sorted(pka_values)],
            ionizable_groups=groups, method="rdkit-empirical",
            confidence=0.30, model_version="empirical",
        )
    except Exception as e:
        logger.error(f"pKa empirical failed: {e}")
        return None


def _detect_ionizable_groups(smiles: str) -> list[str]:
    """Detect ionizable functional groups via SMARTS."""
    from rdkit import Chem
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return []

    patterns = {
        "carboxylic_acid": "[CX3](=O)[OX2H1]",
        "phenol": "[OX2H]c1ccccc1",
        "primary_amine": "[NX3H2;!$(NC=O)]",
        "secondary_amine": "[NX3H1;!$(NC=O)]([#6])[#6]",
        "tertiary_amine": "[NX3H0;!$(NC=O)]([#6])([#6])[#6]",
        "sulfonamide": "[#16](=O)(=O)[NX3H]",
        "thiol": "[SX2H]",
        "imidazole": "c1c[nH]cn1",
        "pyridine": "n1ccccc1",
        "tetrazole": "c1nnn[nH]1",
        "guanidine": "[NX3H2]C(=[NX2H])[NX3H2]",
    }

    found = []
    for name, smarts in patterns.items():
        pattern = Chem.MolFromSmarts(smarts)
        if pattern and mol.HasSubstructMatch(pattern):
            found.append(name)
    return found


def is_ready() -> bool:
    # Honest readiness: empirical-only mode (missing weights) is NOT ready unless
    # empirical serving is explicitly opted into. Without this, /health reports
    # "healthy" while every prediction is a crude empirical estimate.
    return is_available()


def get_info() -> dict:
    return {
        "backend": _backend or "not_loaded",
        "ready": is_ready(),
        "weights_loaded": _weights_loaded(),
        "empirical_only": _backend == "rdkit-empirical",
        "v7_models": len(_v7_models),
        "v8_models": len(_v8_models),
        "v2_models": len(_v2_models),
        "v3_models": len(_v3_models),
        "v1_fallback": _v1_model is not None,
    }

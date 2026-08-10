"""
Solubility Prediction

Phase 1: AqSolPred-style model using RDKit descriptors (room temperature LogS)
Phase 2: Custom Chemprop model trained on AqSolDB + BigSolDB with temperature feature

Returns aqueous solubility as LogS (log10 mol/L).
"""

import logging
import os
from typing import Optional
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger("novomcp-properties.solubility")

_backend: Optional[str] = None
_custom_model = None
_descriptor_model = None

# The trained model. "esol-descriptors" is a crude Delaney heuristic, not a
# trained model — serving it silently when the checkpoint is missing hands the
# caller low-quality solubility with no signal. Treat that as unavailable unless
# explicitly opted into (mirrors the pKa predictor).
_TRAINED_BACKEND = "chemprop-aqsoldb"


def _weights_loaded() -> bool:
    return _backend == _TRAINED_BACKEND


def _esol_allowed() -> bool:
    """Opt-in to serve the ESOL descriptor heuristic on purpose."""
    return os.getenv("SOLUBILITY_ALLOW_ESOL", "").strip().lower() in ("1", "true", "yes")


def is_available() -> bool:
    """Trained model loaded, or the ESOL heuristic explicitly opted into."""
    return _weights_loaded() or _esol_allowed()


@dataclass
class SolubilityResult:
    smiles: str
    logs: float  # log10(mol/L)
    solubility_mg_ml: Optional[float]
    temperature_k: float
    category: str  # highly_soluble, soluble, slightly_soluble, insoluble
    method: str
    confidence: Optional[float] = None


def initialize():
    """Load solubility model."""
    global _backend, _custom_model, _descriptor_model

    # Phase 2: Custom Chemprop model with temperature
    try:
        model_path = os.getenv("SOLUBILITY_MODEL_PATH", "/app/models/solubility_chemprop")
        if os.path.exists(model_path):
            from chemprop.models import MPNN
            _custom_model = MPNN.load_from_checkpoint(model_path, map_location="cpu")
            _custom_model.eval()
            _backend = "chemprop-aqsoldb"
            logger.info(f"Solubility: loaded custom Chemprop model from {model_path}")
            return True
    except Exception as e:
        logger.warning(f"Solubility: custom model not available: {e}")

    # Phase 1: RDKit descriptor-based estimation (ESOL-like)
    try:
        from rdkit import Chem
        from rdkit.Chem import Descriptors
        _backend = "esol-descriptors"
        if _esol_allowed():
            logger.warning(
                "Solubility: no trained model found — serving ESOL descriptor "
                "estimates because SOLUBILITY_ALLOW_ESOL is set. Low-confidence, "
                "labeled method='esol-delaney'."
            )
        else:
            logger.error(
                "Solubility: trained model failed to load (no checkpoint found). "
                "The solubility endpoints will return 503 (model unavailable) rather "
                "than silently serving ESOL descriptor estimates. Wire the weights, or "
                "set SOLUBILITY_ALLOW_ESOL=1 to opt into labeled ESOL output."
            )
        return True
    except Exception as e:
        logger.error(f"Solubility: failed to initialize: {e}")
        _backend = None
        return False


def predict(smiles: str, temperature_k: float = 298.15) -> Optional[SolubilityResult]:
    """Predict aqueous solubility."""
    if _backend is None:
        return None

    if _backend == "chemprop-aqsoldb":
        return _predict_chemprop(smiles, temperature_k)
    else:
        return _predict_esol(smiles, temperature_k)


def _predict_chemprop(smiles: str, temperature_k: float) -> Optional[SolubilityResult]:
    """Predict using custom-trained Chemprop with temperature feature."""
    try:
        import torch
        from chemprop import data as chemprop_data, featurizers

        featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()
        mol_data = [chemprop_data.MoleculeDatapoint.from_smi(
            smiles,
            x_d=np.array([(temperature_k - 298.15) / 50.0], dtype=np.float32),
        )]
        dataset = chemprop_data.MoleculeDataset(mol_data, featurizer)
        loader = chemprop_data.build_dataloader(dataset, batch_size=1, shuffle=False, num_workers=0)

        with torch.inference_mode():
            import lightning as L
            trainer = L.Trainer(accelerator="cpu", devices=1, logger=False, enable_progress_bar=False)
            preds = trainer.predict(_custom_model, loader)

        logs = float(np.concatenate(preds, axis=0)[0][0])

        return _build_result(smiles, logs, temperature_k, "chemprop-aqsoldb", 0.80)
    except Exception as e:
        logger.error(f"Solubility Chemprop prediction failed for {smiles}: {e}")
        return None


def _predict_esol(smiles: str, temperature_k: float) -> Optional[SolubilityResult]:
    """
    ESOL (Estimated SOLubility) model.
    Delaney 2004: LogS = 0.16 - 0.63*cLogP - 0.0062*MW + 0.066*RB - 0.74*AP

    AP = aromatic proportion, RB = rotatable bonds, MW = molecular weight
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import Descriptors, rdMolDescriptors

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        mw = Descriptors.MolWt(mol)
        logp = Descriptors.MolLogP(mol)
        rb = Descriptors.NumRotatableBonds(mol)
        ap = rdMolDescriptors.CalcFractionCSP3(mol)
        aromatic_atoms = sum(1 for a in mol.GetAtoms() if a.GetIsAromatic())
        total_atoms = mol.GetNumHeavyAtoms()
        aromatic_proportion = aromatic_atoms / max(total_atoms, 1)

        # Delaney ESOL equation
        logs = 0.16 - 0.63 * logp - 0.0062 * mw + 0.066 * rb - 0.74 * aromatic_proportion

        # Simple temperature correction (van't Hoff approximation)
        # Most organic solids become more soluble at higher temperatures
        if temperature_k != 298.15:
            delta_t = temperature_k - 298.15
            logs += 0.002 * delta_t  # ~0.2 log units per 100K

        return _build_result(smiles, logs, temperature_k, "esol-delaney", 0.55)
    except Exception as e:
        logger.error(f"Solubility ESOL prediction failed for {smiles}: {e}")
        return None


def _build_result(
    smiles: str, logs: float, temperature_k: float, method: str, confidence: float
) -> SolubilityResult:
    """Build SolubilityResult from LogS."""
    from rdkit import Chem
    from rdkit.Chem import Descriptors

    # Convert LogS (log10 mol/L) to mg/mL.
    # mol/L × g/mol = g/L, and 1 g/L == 1 mg/mL (the 1000× numerator for
    # g→mg cancels the 1000× denominator for L→mL). No division needed.
    mol = Chem.MolFromSmiles(smiles)
    mw = Descriptors.MolWt(mol) if mol else 300.0
    solubility_mol_l = 10 ** logs
    solubility_mg_ml = solubility_mol_l * mw

    # Categorize
    if logs >= -1:
        category = "highly_soluble"
    elif logs >= -3:
        category = "soluble"
    elif logs >= -5:
        category = "slightly_soluble"
    else:
        category = "insoluble"

    return SolubilityResult(
        smiles=smiles,
        logs=round(logs, 3),
        solubility_mg_ml=round(solubility_mg_ml, 6) if solubility_mg_ml else None,
        temperature_k=temperature_k,
        category=category,
        method=method,
        confidence=confidence,
    )


def is_ready() -> bool:
    # Honest readiness: ESOL-only (missing trained model) is not ready unless
    # ESOL serving is explicitly opted into.
    return is_available()


def get_info() -> dict:
    return {
        "backend": _backend or "not_loaded",
        "ready": is_ready(),
        "weights_loaded": _weights_loaded(),
        "esol_only": _backend == "esol-descriptors",
    }

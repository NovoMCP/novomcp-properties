"""
Bond Dissociation Energy (BDE) Prediction

Uses alfabet (A Learning Framework for Accurate Bond Energy Topologies).
Pre-trained model — works out of the box, no custom training needed.
Predicts homolytic BDE in kcal/mol for all C-H bonds in a molecule.
"""

import logging
from typing import Optional
from dataclasses import dataclass

logger = logging.getLogger("novomcp-properties.bde")

_backend: Optional[str] = None
_alfabet_model = None


@dataclass
class BdeResult:
    smiles: str
    bonds: list[dict]  # [{atom_index: int, bde_kcal_mol: float, bond_type: str}]
    weakest_bond: Optional[dict]
    method: str


def initialize():
    """Load alfabet BDE model."""
    global _backend, _alfabet_model

    try:
        from alfabet import model as alfabet_model
        _alfabet_model = alfabet_model
        # Warm up with a simple molecule
        _alfabet_model.predict(["CCO"])
        _backend = "alfabet"
        logger.info("BDE: alfabet model loaded and warmed up")
        return True
    except Exception as e:
        logger.warning(f"BDE: alfabet not available: {e}")
        _backend = None
        return False


def predict(smiles: str) -> Optional[BdeResult]:
    """Predict bond dissociation energies for all C-H bonds."""
    if _backend is None:
        return None

    try:
        result = _alfabet_model.predict([smiles])

        if result is None or len(result) == 0:
            return None

        df = result[0] if isinstance(result, list) else result

        bonds = []
        if hasattr(df, "iterrows"):
            for _, row in df.iterrows():
                # alfabet returns 'bde_pred' (model prediction) and 'bde' (DFT reference, often NaN)
                bde_val = row.get("bde_pred", row.get("bde", row.get("BDE", None)))
                if bde_val is None or (isinstance(bde_val, float) and bde_val != bde_val):
                    continue  # skip None and NaN
                bonds.append({
                    "atom_index": int(row.get("bond_index", row.get("atom_index", 0))),
                    "bde_kcal_mol": round(float(bde_val), 1),
                    "bond_type": str(row.get("bond_type", "C-H")),
                })
        elif isinstance(df, dict):
            for idx, bde_val in df.items():
                if bde_val is None:
                    continue
                bonds.append({
                    "atom_index": int(idx) if str(idx).isdigit() else 0,
                    "bde_kcal_mol": round(float(bde_val), 1),
                    "bond_type": "C-H",
                })

        weakest = min(bonds, key=lambda b: b["bde_kcal_mol"]) if bonds else None

        return BdeResult(
            smiles=smiles,
            bonds=bonds,
            weakest_bond=weakest,
            method="alfabet",
        )
    except Exception as e:
        logger.error(f"BDE prediction failed for {smiles}: {e}")
        return None


def is_ready() -> bool:
    return _backend is not None


def get_info() -> dict:
    return {"backend": _backend or "not_loaded", "ready": is_ready()}

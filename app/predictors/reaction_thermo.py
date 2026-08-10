"""
Reaction Thermodynamics — predict ΔE, ΔH, ΔG, TΔS, K_eq for chemical reactions.

Uses xTB (GFN2) with Hessian for each species:
  1. Optimize geometry of each reactant and product
  2. Run Hessian → ZPE, enthalpy correction, Gibbs correction, entropy
  3. Assemble reaction thermodynamics from differences
  4. Compute equilibrium constant K_eq = exp(-ΔG / RT)

Scope (v1): Organocatalysis and standard organic reactions.
Transition metals accepted but flagged as low confidence.
No transition state search (kinetics) — thermodynamics only.
"""

import logging
import math
import os
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger("novomcp-properties.reaction_thermo")

# Gas constant
R_KCAL_MOL_K = 1.987204e-3  # kcal/(mol·K)

# Elements considered "high confidence" for xTB organic chemistry
HIGH_CONFIDENCE_ELEMENTS = {"H", "C", "N", "O", "F", "S", "Cl", "Br", "P", "Si", "B", "I"}

# Transition metals (trigger low-confidence flag)
TRANSITION_METALS = {
    "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd",
    "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
    "La", "Ce", "Pr", "Nd", "Sm", "Eu", "Gd",
}

HARTREE_TO_KCAL = 627.509


# =============================================================================
# Result dataclass
# =============================================================================

@dataclass
class SpeciesThermo:
    """Thermochemistry for a single species."""
    smiles: str
    energy_hartree: Optional[float] = None
    energy_kcal: Optional[float] = None
    zpe_kcal: Optional[float] = None
    enthalpy_correction_kcal: Optional[float] = None
    gibbs_correction_kcal: Optional[float] = None
    entropy_cal_mol_k: Optional[float] = None
    n_imaginary: int = 0
    is_true_minimum: bool = True
    error: Optional[str] = None


@dataclass
class ReactionStep:
    """Thermodynamics for a single reaction step."""
    reactants: list[str]
    products: list[str]
    delta_e_kcal: Optional[float] = None
    delta_h_kcal: Optional[float] = None
    delta_g_kcal: Optional[float] = None
    t_delta_s_kcal: Optional[float] = None
    k_eq: Optional[float] = None
    spontaneous: Optional[bool] = None
    temperature_k: float = 298.15


@dataclass
class ReactionThermoResult:
    success: bool
    reactant_smiles: list[str] = field(default_factory=list)
    product_smiles: list[str] = field(default_factory=list)

    # Overall reaction thermodynamics
    delta_e_kcal: Optional[float] = None
    delta_h_kcal: Optional[float] = None
    delta_g_kcal: Optional[float] = None
    t_delta_s_kcal: Optional[float] = None
    k_eq: Optional[float] = None
    spontaneous: Optional[bool] = None
    temperature_k: float = 298.15

    # Per-species data (for expert users)
    species_data: list[dict] = field(default_factory=list)

    # Multi-step pathway (if provided)
    steps: list[dict] = field(default_factory=list)

    # Confidence
    confidence: str = "high"
    confidence_note: str = ""
    has_imaginary_frequencies: bool = False

    # Metadata
    solvent: Optional[str] = None
    method: str = "GFN2-xTB + Hessian thermochemistry"
    wall_time_seconds: Optional[float] = None
    error: Optional[str] = None
    warnings: list[str] = field(default_factory=list)


# =============================================================================
# Element detection for confidence classification
# =============================================================================

def _detect_elements(smiles: str) -> set[str]:
    """Detect elements in a SMILES string via RDKit."""
    try:
        from rdkit import Chem
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return set()
        return {atom.GetSymbol() for atom in mol.GetAtoms()}
    except ImportError:
        # Fallback: crude SMILES parsing for common elements
        elements = set()
        for char in smiles:
            if char.isupper():
                elements.add(char)
        return elements
    except Exception:
        return set()


def _classify_confidence(all_smiles: list[str], has_imaginary: bool) -> tuple[str, str]:
    """Classify confidence based on elements and imaginary frequencies."""
    all_elements = set()
    for s in all_smiles:
        all_elements |= _detect_elements(s)

    has_metals = bool(all_elements & TRANSITION_METALS)
    non_standard = all_elements - HIGH_CONFIDENCE_ELEMENTS

    if has_metals:
        return "low", (
            f"Contains transition metal(s): {all_elements & TRANSITION_METALS}. "
            "xTB accuracy is limited for open-shell transition metal systems. "
            "Validate with DFT for metal-containing catalysts."
        )
    if has_imaginary:
        if non_standard:
            return "low", (
                f"Imaginary frequencies detected AND non-standard elements: {non_standard}. "
                "Structure(s) may not be at true minima. Re-optimize or use tighter convergence."
            )
        return "medium", (
            "Minor imaginary frequencies detected — structure(s) may not be at "
            "true minima. Thermochemistry is approximate. Consider re-optimization."
        )
    if non_standard:
        return "medium", (
            f"Contains elements outside the high-confidence organic set: {non_standard}. "
            "xTB thermochemistry may be less accurate for these elements."
        )
    return "high", "All-organic system, no imaginary frequencies. xTB thermochemistry is reliable for relative ΔG."


# =============================================================================
# Core: compute thermochemistry for a single species
# =============================================================================

def _compute_species_thermo(
    smiles: str,
    charge: int = 0,
    uhf: int = 0,
    solvent: Optional[str] = None,
    temperature: float = 298.15,
) -> SpeciesThermo:
    """Optimize + Hessian for a single species."""
    from app.predictors._xtb_client import get_hessian

    data = get_hessian(
        smiles, charge=charge, uhf=uhf, solvent=solvent,
        temperature=temperature, optimize_first=True,
    )

    if not data or not data.get("energy_hartree"):
        upstream = (data or {}).get("_error") or (data or {}).get("detail")
        suffix = f": {upstream}" if upstream else ""
        return SpeciesThermo(
            smiles=smiles,
            error=f"xTB optimization + Hessian failed for {smiles}{suffix}"
        )

    e_hartree = data["energy_hartree"]
    n_imag = data.get("n_imaginary", 0)

    return SpeciesThermo(
        smiles=smiles,
        energy_hartree=e_hartree,
        energy_kcal=round(e_hartree * HARTREE_TO_KCAL, 3),
        zpe_kcal=data.get("zpe_kcal_mol"),
        enthalpy_correction_kcal=data.get("enthalpy_correction_kcal_mol"),
        gibbs_correction_kcal=data.get("gibbs_correction_kcal_mol"),
        entropy_cal_mol_k=data.get("entropy_cal_mol_k"),
        n_imaginary=n_imag,
        is_true_minimum=n_imag == 0,
    )


# =============================================================================
# Main prediction function
# =============================================================================

def predict(
    reactant_smiles: list[str],
    product_smiles: list[str],
    solvent: Optional[str] = None,
    temperature: float = 298.15,
) -> ReactionThermoResult:
    """Predict reaction thermodynamics: ΔE, ΔH, ΔG, TΔS, K_eq.

    Computes optimize + Hessian for each species (reactants + products),
    then assembles reaction thermodynamics from differences.
    """
    import time
    t0 = time.time()
    warnings = []

    if not reactant_smiles or not product_smiles:
        return ReactionThermoResult(
            success=False,
            error="Both reactant_smiles and product_smiles are required"
        )

    all_smiles = reactant_smiles + product_smiles
    species_results: dict[str, SpeciesThermo] = {}

    # Compute thermochemistry for each unique species
    unique_smiles = list(dict.fromkeys(all_smiles))  # preserve order, deduplicate
    for i, smi in enumerate(unique_smiles):
        logger.info(f"Reaction thermo: species {i+1}/{len(unique_smiles)} — {smi}")
        result = _compute_species_thermo(
            smi, solvent=solvent, temperature=temperature,
        )
        species_results[smi] = result
        if result.error:
            warnings.append(f"{smi}: {result.error}")

    # Check for failures — need all species to compute reaction thermo
    failed = [s for s in all_smiles if species_results[s].error]
    if failed:
        return ReactionThermoResult(
            success=False,
            reactant_smiles=reactant_smiles,
            product_smiles=product_smiles,
            error=f"Thermochemistry failed for: {', '.join(failed)}",
            warnings=warnings,
            wall_time_seconds=round(time.time() - t0, 1),
        )

    # Assemble reaction thermodynamics
    # ΔX = Σ(products) - Σ(reactants) for each thermodynamic quantity

    def _sum_property(smiles_list: list[str], prop: str) -> Optional[float]:
        vals = [getattr(species_results[s], prop) for s in smiles_list]
        if any(v is None for v in vals):
            return None
        return sum(vals)

    # Electronic energy: ΔE
    e_reactants = _sum_property(reactant_smiles, "energy_kcal")
    e_products = _sum_property(product_smiles, "energy_kcal")
    delta_e = round(e_products - e_reactants, 3) if (e_reactants is not None and e_products is not None) else None

    # ZPE-corrected energy
    zpe_reactants = _sum_property(reactant_smiles, "zpe_kcal")
    zpe_products = _sum_property(product_smiles, "zpe_kcal")
    delta_zpe = (zpe_products - zpe_reactants) if (zpe_reactants is not None and zpe_products is not None) else 0.0

    # Enthalpy: ΔH = ΔE + ΔZPE + ΔH_thermal
    h_corr_reactants = _sum_property(reactant_smiles, "enthalpy_correction_kcal")
    h_corr_products = _sum_property(product_smiles, "enthalpy_correction_kcal")
    delta_h_thermal = (h_corr_products - h_corr_reactants) if (h_corr_reactants is not None and h_corr_products is not None) else 0.0

    delta_h = None
    if delta_e is not None:
        delta_h = round(delta_e + delta_zpe + delta_h_thermal, 3)

    # Gibbs free energy: from Gibbs corrections directly
    g_corr_reactants = _sum_property(reactant_smiles, "gibbs_correction_kcal")
    g_corr_products = _sum_property(product_smiles, "gibbs_correction_kcal")

    delta_g = None
    if delta_e is not None and g_corr_reactants is not None and g_corr_products is not None:
        # G = E + G_correction (G_correction includes ZPE + thermal + entropy)
        g_reactants = e_reactants + g_corr_reactants
        g_products = e_products + g_corr_products
        delta_g = round(g_products - g_reactants, 3)

    # Entropy contribution: TΔS = ΔH - ΔG
    t_delta_s = None
    if delta_h is not None and delta_g is not None:
        t_delta_s = round(delta_h - delta_g, 3)

    # Equilibrium constant
    # Cap at 1e200 to avoid JSON serialization failure (float("inf") is not
    # JSON-compliant). For strongly exothermic reactions (ΔG << 0), K_eq
    # overflows — the reaction is effectively irreversible.
    k_eq = None
    spontaneous = None
    if delta_g is not None:
        spontaneous = delta_g < 0
        try:
            exponent = -delta_g / (R_KCAL_MOL_K * temperature)
            if exponent > 460:  # exp(460) ≈ 1e200
                k_eq = 1e200
            elif exponent < -460:
                k_eq = 0.0
            else:
                k_eq = math.exp(exponent)
        except (OverflowError, ValueError):
            k_eq = 1e200 if delta_g < 0 else 0.0

    # Imaginary frequency check
    has_imaginary = any(species_results[s].n_imaginary > 0 for s in all_smiles)
    if has_imaginary:
        imag_species = [s for s in all_smiles if species_results[s].n_imaginary > 0]
        warnings.append(
            f"Imaginary frequencies in: {', '.join(imag_species)}. "
            "These species may not be at true minima — thermochemistry is approximate."
        )

    # Confidence classification
    confidence, confidence_note = _classify_confidence(all_smiles, has_imaginary)

    # Per-species data for the response
    species_data = []
    for smi in unique_smiles:
        sp = species_results[smi]
        species_data.append({
            "smiles": smi,
            "role": "reactant" if smi in reactant_smiles else "product",
            "energy_kcal": sp.energy_kcal,
            "zpe_kcal": sp.zpe_kcal,
            "enthalpy_correction_kcal": sp.enthalpy_correction_kcal,
            "gibbs_correction_kcal": sp.gibbs_correction_kcal,
            "entropy_cal_mol_k": sp.entropy_cal_mol_k,
            "n_imaginary": sp.n_imaginary,
            "is_true_minimum": sp.is_true_minimum,
        })

    wall_time = round(time.time() - t0, 1)

    return ReactionThermoResult(
        success=True,
        reactant_smiles=reactant_smiles,
        product_smiles=product_smiles,
        delta_e_kcal=delta_e,
        delta_h_kcal=delta_h,
        delta_g_kcal=delta_g,
        t_delta_s_kcal=t_delta_s,
        k_eq=k_eq,
        spontaneous=spontaneous,
        temperature_k=temperature,
        species_data=species_data,
        confidence=confidence,
        confidence_note=confidence_note,
        has_imaginary_frequencies=has_imaginary,
        solvent=solvent,
        method="GFN2-xTB + Hessian thermochemistry",
        wall_time_seconds=wall_time,
        warnings=warnings,
    )

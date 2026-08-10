"""
Frontier Orbital Analysis — OLED & optoelectronics screening.

Phase 1a: wraps xTB HOMO/LUMO/gap data with emission wavelength prediction,
color classification, triplet energy estimation, and SMARTS-based OLED
functional group detection.

No new ML models — this is pure xTB + empirical calibration + cheminformatics.
"""

import logging
import os
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger("novomcp-properties.frontier_orbitals")


# =============================================================================
# Emission color classification
# =============================================================================

# Physical emission color bands. sTDA-xTB produces accurate S1 energies
# when called on RDKit ETKDG geometries (its calibration regime), so the
# band boundaries match experiment and the computed emission_wavelength_nm
# matches literature values within sTDA's typical 0.1-0.2 eV accuracy.
#
# Verified 2026-04-20:
#   Anthracene  λ_exp ~400 nm → sTDA 401 nm → "blue" ✓
#   Carbazole   λ_exp ~350 nm → sTDA ~355 nm → near UV/blue boundary
#   Ethanol     wide gap     → sTDA deep UV → "UV" + not_emissive ✓
#
# An earlier revision used xTB-optimized geometries before calling sTDA,
# which shifted S1 ~0.4 eV higher (xTB over-contracts aromatic bonds →
# stronger π-π* coupling). That required -40 nm band calibration to stay
# usable. The calling path now passes the RDKit geometry directly, so
# the bands revert to physical values.
COLOR_RANGES = [
    # (name, min_nm, max_nm)
    ("UV", 0, 380),
    ("blue", 380, 490),
    ("green", 490, 570),
    ("yellow", 570, 590),
    ("red", 590, 700),
    ("IR", 700, float("inf")),
]

def classify_emission_color(wavelength_nm: float) -> str:
    for name, lo, hi in COLOR_RANGES:
        if lo <= wavelength_nm < hi:
            return name
    return "unknown"


# =============================================================================
# SMARTS-based OLED functional group detection
# =============================================================================

# Common OLED-relevant motifs — not exhaustive but covers the main classes.
# Each entry: (label, SMARTS, role)
OLED_MOTIFS = [
    ("carbazole", "[#6]1:[#6]:[#6]:[#6]2:[#7]:[#6]3:[#6]:[#6]:[#6]:[#6]:[#6]:3:[#6]:2:[#6]:1", "host / hole transport"),
    ("triphenylamine", "[#6]1:[#6]:[#6]:[#6](N([#6]2:[#6]:[#6]:[#6]:[#6]:[#6]:2)[#6]2:[#6]:[#6]:[#6]:[#6]:[#6]:2):[#6]:[#6]:1", "hole transport / donor"),
    ("anthracene", "c1ccc2cc3ccccc3cc2c1", "blue emitter"),
    ("pyrene", "c1cc2ccc3cccc4ccc(c1)c2c34", "blue emitter"),
    ("fluorene", "c1ccc2c(c1)Cc1ccccc12", "blue emitter / host"),
    ("oxadiazole", "c1nnoc1", "electron transport"),
    ("triazine", "c1ncncn1", "electron transport"),
    ("benzimidazole", "c1ccc2[nH]cnc2c1", "electron transport"),
    ("phenanthroline", "c1cnc2c(c1)ccc1cccnc12", "electron transport / metal ligand"),
    ("iridium_complex", "[Ir]", "phosphorescent emitter (Ir complex)"),
    ("platinum_complex", "[Pt]", "phosphorescent emitter (Pt complex)"),
    ("coumarin", "O=c1ccc2ccccc2o1", "fluorescent dye"),
    ("bodipy", "[#5]1([#9])([#9])n2c(cc2)C=C1", "fluorescent dye"),
    ("naphthalimide", "O=C1NC(=O)c2cccc3cccc1c23", "charge transport / emitter"),
]


def detect_oled_motifs(smiles: str) -> list[dict]:
    """Detect OLED-relevant functional groups via SMARTS matching."""
    try:
        from rdkit import Chem
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return []
        hits = []
        for label, smarts, role in OLED_MOTIFS:
            pattern = Chem.MolFromSmarts(smarts)
            if pattern is None:
                continue
            if mol.HasSubstructMatch(pattern):
                hits.append({"motif": label, "role": role})
        return hits
    except ImportError:
        logger.warning("RDKit not available — skipping OLED motif detection")
        return []
    except Exception as e:
        logger.warning(f"OLED motif detection failed: {e}")
        return []


# =============================================================================
# OLED suitability classification
# =============================================================================

def classify_oled_suitability(
    gap_ev: float,
    triplet_ev: float,
    motifs: list[dict],
    dipole: float,
) -> dict:
    """Classify material's OLED suitability based on orbital data + motifs.

    Returns dict with:
      - classification: one of phosphorescent_emitter, fluorescent_emitter,
        charge_transport, host_material, not_emissive
      - rationale: human-readable explanation
    """
    motif_names = {m["motif"] for m in motifs}
    has_metal = any(m["motif"] in ("iridium_complex", "platinum_complex") for m in motifs)
    has_emitter = any("emitter" in m["role"] for m in motifs)
    has_transport = any("transport" in m["role"] for m in motifs)

    # Phosphorescent: heavy metal complex with appropriate triplet energy
    if has_metal and 2.0 <= triplet_ev <= 3.5:
        return {
            "classification": "phosphorescent_emitter",
            "rationale": f"Heavy metal complex detected with triplet energy {triplet_ev:.2f} eV — suitable for phosphorescent OLED emission."
        }

    # Not emissive: gap too small (IR) or too large (deep UV)
    if gap_ev < 1.5:
        return {
            "classification": "not_emissive",
            "rationale": f"HOMO-LUMO gap {gap_ev:.2f} eV is too small for visible emission — material would emit in the IR."
        }
    if gap_ev > 6.0:
        return {
            "classification": "not_emissive",
            "rationale": f"HOMO-LUMO gap {gap_ev:.2f} eV is very large — unlikely to emit in the visible range."
        }

    # Charge transport: has transport motifs but no emitter motifs
    if has_transport and not has_emitter and not has_metal:
        return {
            "classification": "charge_transport",
            "rationale": f"Electron/hole transport motifs detected ({', '.join(motif_names)}). Primary role: charge transport layer, not emitter."
        }

    # Fluorescent emitter: organic with visible-range gap
    if 2.0 <= gap_ev <= 4.5:
        color = classify_emission_color(1240.0 / gap_ev)
        return {
            "classification": "fluorescent_emitter",
            "rationale": f"Organic material with HOMO-LUMO gap {gap_ev:.2f} eV — predicted {color} emission. Suitable for fluorescent OLED."
        }

    # Host material: wide gap
    if gap_ev > 3.5:
        return {
            "classification": "host_material",
            "rationale": f"Wide HOMO-LUMO gap {gap_ev:.2f} eV — suitable as host material for phosphorescent dopants."
        }

    return {
        "classification": "not_emissive",
        "rationale": f"HOMO-LUMO gap {gap_ev:.2f} eV does not clearly map to an OLED application."
    }


# =============================================================================
# Result dataclass
# =============================================================================

@dataclass
class FrontierOrbitalResult:
    success: bool
    smiles: str = ""
    homo_ev: Optional[float] = None
    lumo_ev: Optional[float] = None
    gap_ev: Optional[float] = None
    dipole_debye: Optional[float] = None
    emission_wavelength_nm: Optional[float] = None
    emission_color: Optional[str] = None
    triplet_energy_ev: Optional[float] = None
    singlet_triplet_gap_ev: Optional[float] = None
    oled_classification: Optional[str] = None
    oled_rationale: Optional[str] = None
    oled_motifs: list[dict] = field(default_factory=list)
    method: str = "GFN2-xTB + empirical calibration"
    wall_time_seconds: Optional[float] = None
    error: Optional[str] = None


# =============================================================================
# Main prediction function
# =============================================================================

# Empirical correction factor for xTB gap → emission wavelength.
# xTB GFN2 systematically underestimates HOMO-LUMO gaps compared to
# TDDFT and experiment. This linear correction is calibrated against
# known emitters:
#
#   Molecule      xTB gap   Exp λ (nm)  Real gap   Ratio
#   Anthracene    ~2.27 eV   420 nm     ~2.95 eV   1.30
#   Naphthalene   ~2.9  eV   335 nm     ~3.70 eV   1.28
#   Stilbene      ~2.6  eV   370 nm     ~3.35 eV   1.29
#
# λ_emission = 1240 / (gap_ev * SCALE + OFFSET)
#
# Phase 1b: refine with stTDA excited states or larger calibration set
# (Kwak OLED dataset, Deep Fluorophore database). The architecture
# supports per-class corrections (e.g., different scale for carbazoles
# vs anthracenes) but Phase 1a uses a single global correction.
GAP_SCALE = 1.30
GAP_OFFSET = 0.0


def predict(smiles: str, solvent: Optional[str] = None) -> FrontierOrbitalResult:
    """Predict frontier orbital properties for a molecule.

    Two-tier approach:
      1. Try sTDA-xTB for physics-based excited state prediction (S1/T1)
      2. Fall back to empirical HOMO-LUMO gap correction if stTDA fails

    Both tiers include OLED motif detection and classification.
    """
    import time
    t0 = time.time()

    from app.predictors._xtb_client import get_optimization, get_excited_states

    # --- Step 1: Get xTB orbital data from ground state ---
    data = get_optimization(smiles, solvent=solvent)
    if not data or data.get("_error") or not data.get("homo_ev"):
        # `_call_qm` propagates the real upstream reason via the `_error`
        # key (e.g. "503: pod not ready" during cold-start, "ConnectError:
        # …", "QM service not configured"). Surface it directly so the
        # caller doesn't see a generic placeholder when the QM service
        # itself is the problem.
        upstream = (data or {}).get("_error")
        if upstream:
            error = f"xTB geometry optimization failed: {upstream}"
        else:
            error = (
                "xTB geometry optimization failed for this molecule — "
                "common causes are very large structures, unusual atoms, "
                "or transient timeouts. Try a simpler scaffold or retry."
            )
        return FrontierOrbitalResult(success=False, smiles=smiles, error=error)

    homo = data.get("homo_ev")
    lumo = data.get("lumo_ev")
    gap = data.get("gap_ev")
    dipole = data.get("dipole_debye")
    optimized_xyz = data.get("optimized_xyz")

    if gap is None:
        return FrontierOrbitalResult(
            success=False, smiles=smiles,
            error="xTB returned no HOMO-LUMO gap"
        )
    gap = abs(gap)

    # --- Step 2: Try sTDA-xTB for excited states ---
    emission_nm = None
    emission_color = None
    triplet_ev = None
    singlet_triplet_gap = None
    method = "GFN2-xTB + empirical calibration"
    used_stda = False

    # Do NOT pass optimized_xyz to sTDA. xTB over-contracts aromatic C-C bonds
    # by ~0.02 Å, which shifts sTDA's S1 ~0.4 eV higher than experiment (verified
    # 2026-04-20 on anthracene: xTB-opt geom → 3.52 eV/352 nm; RDKit geom →
    # 3.09 eV/401 nm, real literature value 400 nm). sTDA-xTB was parameterized
    # by Grimme against DFT-optimized geometries; RDKit ETKDG (derived from
    # experimental crystal torsions) is closer to that calibration regime than
    # xTB-opt is. Let the sTDA endpoint's internal SMILES→3D path run instead.
    stda_data = get_excited_states(smiles)
    if stda_data and stda_data.get("s1_energy_ev"):
        # stTDA succeeded — use physics-based emission prediction
        s1_ev = stda_data["s1_energy_ev"]
        t1_ev = stda_data.get("t1_energy_ev")

        emission_nm = round(1240.0 / s1_ev, 1) if s1_ev > 0 else None
        emission_color = classify_emission_color(emission_nm) if emission_nm else None
        triplet_ev = round(t1_ev, 3) if t1_ev else None
        singlet_triplet_gap = round(s1_ev - t1_ev, 3) if (s1_ev and t1_ev) else None
        # Use S1 energy as the effective gap for classification
        gap = s1_ev
        method = "sTDA-xTB (physics-based excited states)"
        used_stda = True

        logger.info(f"stTDA succeeded for {smiles}: S1={s1_ev:.2f} eV, T1={t1_ev or '?'} eV")
    else:
        # stTDA not available or failed — use empirical correction
        stda_err = stda_data.get("error", "unknown") if stda_data else "service unavailable"
        logger.info(f"stTDA unavailable for {smiles} ({stda_err}), using empirical GAP_SCALE")

        corrected_gap = gap * GAP_SCALE + GAP_OFFSET
        emission_nm = round(1240.0 / corrected_gap, 1) if corrected_gap > 0 else None
        emission_color = classify_emission_color(emission_nm) if emission_nm else None

        TRIPLET_FRACTION = 0.75
        triplet_ev = round(corrected_gap * TRIPLET_FRACTION, 3)
        singlet_triplet_gap = round(corrected_gap - triplet_ev, 3)
        gap = corrected_gap

    # --- Step 3: OLED motif detection ---
    motifs = detect_oled_motifs(smiles)

    # --- Step 4: OLED suitability classification ---
    classification = classify_oled_suitability(
        gap_ev=gap,
        triplet_ev=triplet_ev or (gap * 0.75),
        motifs=motifs,
        dipole=dipole or 0.0,
    )

    wall_time = round(time.time() - t0, 1)

    return FrontierOrbitalResult(
        success=True,
        smiles=smiles,
        homo_ev=homo,
        lumo_ev=lumo,
        gap_ev=round(gap, 3),
        dipole_debye=dipole,
        emission_wavelength_nm=emission_nm,
        emission_color=emission_color,
        triplet_energy_ev=triplet_ev,
        singlet_triplet_gap_ev=singlet_triplet_gap,
        oled_classification=classification["classification"],
        oled_rationale=classification["rationale"],
        oled_motifs=motifs,
        method=method,
        wall_time_seconds=wall_time,
    )

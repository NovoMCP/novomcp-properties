"""
Shared xTB client — HTTP calls to novomcp-qm.

Used by:
  - pka.py (energy + charges for v8 route)
  - frontier_orbitals.py (optimize + orbital energies)
  - redox.py (optimize + charged species + xyz passthrough)
  - reaction_thermo.py (optimize + hessian)

All functions return dicts (not dataclasses) for simplicity — callers
extract what they need. On persistent failure the returned dict carries
an `_error` key with the real reason (status + body snippet, connection
error, etc.) so callers can surface a useful message instead of "no
response". The dict is still missing the success keys (energy_hartree,
optimized_xyz, …) so existing `if not data.get("energy_hartree"):`
checks continue to trigger the failure branch.
"""

import logging
import os
import time
from typing import Any, Optional

logger = logging.getLogger("novomcp-properties.xtb_client")

_QM_URL: str = ""
_QM_KEY: str = ""


def _ensure_config():
    """Lazy-load env vars (not available at import time in some deploy configs)."""
    global _QM_URL, _QM_KEY
    if not _QM_URL:
        _QM_URL = os.getenv("NOVOMCP_QM_URL", "")
        _QM_KEY = os.getenv("NOVOMCP_QM_API_KEY", "")


def _headers() -> dict:
    return {"X-API-Key": _QM_KEY, "Content-Type": "application/json"}


# =========================================================================
# Core call — all other functions build on this
# =========================================================================

# novomcp-qm runs scale-to-zero on AWS (per feedback_aws_cold_start_defaults).
# The first call after the pod has been idle hits an unready pod and gets back
# 502/503/504 from the ingress, or a connection error mid-activation. Retry a
# couple of times with short backoff to absorb the warm-up transparently.
# Non-retryable statuses (4xx, especially 400 from bad SMILES) fail fast.
_RETRY_STATUSES = {502, 503, 504}
_RETRY_BACKOFFS = [1.0, 3.0]  # seconds before retry attempts 2 and 3


def _call_qm(endpoint: str, payload: dict, timeout: float = 60.0) -> dict:
    """POST to novomcp-qm and return the JSON response.

    On persistent failure returns `{"_error": "<reason>"}` so callers can
    surface the underlying cause. Retries 502/503/504 and connection
    errors (cold-start absorption) but fails fast on 4xx (client error).
    """
    _ensure_config()
    if not _QM_URL:
        logger.warning("NOVOMCP_QM_URL not configured — xTB calls will fail")
        return {"_error": "QM service not configured (NOVOMCP_QM_URL unset)"}

    import httpx

    last_error = "unknown failure"
    max_attempts = 1 + len(_RETRY_BACKOFFS)

    for attempt in range(max_attempts):
        try:
            response = httpx.post(
                f"{_QM_URL}{endpoint}",
                json=payload,
                headers=_headers(),
                timeout=timeout,
            )
            if response.status_code == 200:
                if attempt > 0:
                    logger.info(f"xTB {endpoint} succeeded on attempt {attempt + 1} (cold-start absorbed)")
                return response.json()

            body_snippet = response.text[:200]
            last_error = f"{response.status_code}: {body_snippet}"

            if response.status_code in _RETRY_STATUSES and attempt < max_attempts - 1:
                wait = _RETRY_BACKOFFS[attempt]
                logger.info(
                    f"xTB {endpoint} returned {response.status_code} on attempt "
                    f"{attempt + 1}/{max_attempts}, retrying in {wait}s (cold-start)"
                )
                time.sleep(wait)
                continue

            logger.warning(f"xTB {endpoint} failed: {last_error}")
            return {"_error": last_error}

        except (httpx.ConnectError, httpx.ConnectTimeout, httpx.ReadTimeout) as e:
            last_error = f"{type(e).__name__}: {str(e)[:150]}"
            if attempt < max_attempts - 1:
                wait = _RETRY_BACKOFFS[attempt]
                logger.info(
                    f"xTB {endpoint} {type(e).__name__} on attempt "
                    f"{attempt + 1}/{max_attempts}, retrying in {wait}s"
                )
                time.sleep(wait)
                continue
            logger.warning(f"xTB {endpoint} failed: {last_error}")
            return {"_error": last_error}

        except Exception as e:
            last_error = f"{type(e).__name__}: {str(e)[:150]}"
            logger.warning(f"xTB {endpoint} unexpected error: {last_error}")
            return {"_error": last_error}

    return {"_error": last_error}


# =========================================================================
# High-level functions
# =========================================================================

def get_energy(
    smiles: str,
    charge: int = 0,
    uhf: int = 0,
    solvent: Optional[str] = None,
    xyz_input: Optional[str] = None,
) -> dict:
    """Single-point energy calculation. Returns energy, orbitals, charges."""
    payload: dict[str, Any] = {"smiles": smiles, "calculation": "energy"}
    if charge:
        payload["charge"] = charge
    if uhf:
        payload["uhf"] = uhf
    if solvent:
        payload["solvent"] = solvent
    if xyz_input:
        payload["xyz_input"] = xyz_input
    return _call_qm("/api/qm-calculate", payload, timeout=120.0)


def get_optimization(
    smiles: str,
    charge: int = 0,
    uhf: int = 0,
    solvent: Optional[str] = None,
    xyz_input: Optional[str] = None,
) -> dict:
    """Geometry optimization. Returns optimized_xyz + energy + orbitals.

    Timeout 180s absorbs novomcp-qm cold start. The optimization itself
    completes in <30s once the container is warm; the first call after
    scale-to-zero needs the extra budget for container activation +
    xTB binary initialization.
    """
    payload: dict[str, Any] = {"smiles": smiles, "calculation": "optimize"}
    if charge:
        payload["charge"] = charge
    if uhf:
        payload["uhf"] = uhf
    if solvent:
        payload["solvent"] = solvent
    if xyz_input:
        payload["xyz_input"] = xyz_input
    return _call_qm("/api/qm-calculate", payload, timeout=180.0)


def get_solvation(
    smiles: str,
    solvent: str = "water",
    charge: int = 0,
    uhf: int = 0,
) -> dict:
    """Solvation energy (gas phase vs solvent). Returns solvation_energy_kcal_mol."""
    payload: dict[str, Any] = {
        "smiles": smiles, "calculation": "solvation", "solvent": solvent,
    }
    if charge:
        payload["charge"] = charge
    if uhf:
        payload["uhf"] = uhf
    return _call_qm("/api/qm-calculate", payload, timeout=120.0)


def get_charges(smiles: str) -> list[float]:
    """Get per-atom Mulliken charges. Convenience wrapper for pKa."""
    data = get_energy(smiles)
    return data.get("partial_charges", [])


def get_excited_states(
    smiles: str,
    charge: int = 0,
    num_states: int = 10,
    xyz_input: Optional[str] = None,
) -> dict:
    """Run sTDA-xTB excited state calculation. Returns S1/T1 energies, wavelengths."""
    payload: dict[str, Any] = {"smiles": smiles, "num_states": num_states}
    if charge:
        payload["charge"] = charge
    if xyz_input:
        payload["xyz_input"] = xyz_input
    return _call_qm("/api/qm-excited-states", payload, timeout=120.0)


def get_hessian(
    smiles: str,
    charge: int = 0,
    uhf: int = 0,
    solvent: Optional[str] = None,
    temperature: float = 298.15,
    xyz_input: Optional[str] = None,
    optimize_first: bool = False,
) -> dict:
    """Hessian / frequency calculation. Returns ZPE, H, G, S, frequencies."""
    payload: dict[str, Any] = {"smiles": smiles}
    if charge:
        payload["charge"] = charge
    if uhf:
        payload["uhf"] = uhf
    if solvent:
        payload["solvent"] = solvent
    if abs(temperature - 298.15) > 0.01:
        payload["temperature"] = temperature
    if xyz_input:
        payload["xyz_input"] = xyz_input
    if optimize_first:
        payload["optimize_first"] = True
    return _call_qm("/api/qm-hessian", payload, timeout=180.0)


# =========================================================================
# Legacy-compatible wrapper (drop-in for pka.py's _get_xtb_data)
# =========================================================================

def get_xtb_data(smiles: str) -> dict:
    """Legacy wrapper matching pka.py's _get_xtb_data() return format.

    Returns dict with: energy_hartree, homo_ev, lumo_ev, gap_ev,
    dipole_debye, partial_charges. Empty dict on failure.
    """
    data = get_energy(smiles)
    if not data:
        return {}
    return {
        "energy_hartree": data.get("energy_hartree", 0.0) or 0.0,
        "homo_ev": data.get("homo_ev", 0.0) or 0.0,
        "lumo_ev": data.get("lumo_ev", 0.0) or 0.0,
        "gap_ev": data.get("gap_ev", 0.0) or 0.0,
        "dipole_debye": data.get("dipole_debye", 0.0) or 0.0,
        "partial_charges": data.get("partial_charges", []),
    }

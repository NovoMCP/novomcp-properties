"""
NovoMCP Properties Service

Molecular property prediction: pKa, solubility, bond dissociation energy.
Custom Chemprop models trained on IUPAC pKa (22K) + AqSolDB solubility (10K).
Falls back to empirical models (ESOL, RDKit SMARTS) if checkpoints unavailable.
"""

import os
import logging
import time
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

from app.predictors import pka, solubility, bde, frontier_orbitals, redox, reaction_thermo

# Logging
logging.basicConfig(
    format="[NovoQuantNexus] %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger("novomcp-properties")

# Config
PORT = int(os.getenv("PORT", "8030"))
API_KEY = os.getenv("PROPERTIES_API_KEY", "")

app = FastAPI(
    title="NovoMCP Properties",
    description="Molecular property prediction: pKa, aqueous solubility, bond dissociation energy",
    version="9.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Auth ---

async def validate_key(x_api_key: Optional[str] = Header(None)):
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return x_api_key


# --- Request/Response Models ---

class PkaRequest(BaseModel):
    smiles: str = Field(..., description="SMILES string")

class PkaResponse(BaseModel):
    smiles: str
    pka_values: list[float]
    ionizable_groups: list[str]
    method: str
    confidence: Optional[float] = None
    uncertainty: Optional[float] = None
    model_version: Optional[str] = None

class SolubilityRequest(BaseModel):
    smiles: str = Field(..., description="SMILES string")
    temperature_k: float = Field(298.15, description="Temperature in Kelvin (default 25C)")

class SolubilityResponse(BaseModel):
    smiles: str
    logs: float
    solubility_mg_ml: Optional[float]
    temperature_k: float
    category: str
    method: str
    confidence: Optional[float] = None

class BdeRequest(BaseModel):
    smiles: str = Field(..., description="SMILES string")

class BdeResponse(BaseModel):
    smiles: str
    bonds: list[dict]
    weakest_bond: Optional[dict]
    method: str

class FrontierOrbitalsRequest(BaseModel):
    smiles: str = Field(..., description="SMILES string")
    solvent: Optional[str] = Field(None, description="Solvent for ALPB solvation model (e.g., water, toluene, chloroform)")

class FrontierOrbitalsResponse(BaseModel):
    smiles: str
    homo_ev: Optional[float]
    lumo_ev: Optional[float]
    gap_ev: Optional[float]
    dipole_debye: Optional[float]
    emission_wavelength_nm: Optional[float]
    emission_color: Optional[str]
    triplet_energy_ev: Optional[float]
    singlet_triplet_gap_ev: Optional[float]
    oled_classification: Optional[str]
    oled_rationale: Optional[str]
    oled_motifs: list[dict] = []
    method: str
    wall_time_seconds: Optional[float]

class RedoxRequest(BaseModel):
    smiles: str = Field(..., description="SMILES string")
    solvent: str = Field("water", description="Solvent for ALPB solvation (e.g., water, acetonitrile, ethylene_carbonate)")
    reference_electrode: str = Field("SHE", description="Reference electrode: SHE, Li/Li+, Ag/AgCl, SCE, Fc/Fc+")

class RedoxResponse(BaseModel):
    smiles: str
    ip_adiabatic_ev: Optional[float] = None
    ip_vertical_ev: Optional[float] = None
    ea_adiabatic_ev: Optional[float] = None
    ea_vertical_ev: Optional[float] = None
    oxidation_potential_v: Optional[float] = None
    reduction_potential_v: Optional[float] = None
    reference_electrode: str
    stability_windows: list[dict] = []
    electrochemical_window_v: Optional[float] = None
    solvent: str
    solvent_class: str = "unknown"
    method: str
    accuracy_note: str = ""
    wall_time_seconds: Optional[float]
    warnings: list[str] = []

class ReactionThermoRequest(BaseModel):
    reactant_smiles: list[str] = Field(..., description="SMILES strings of reactants", min_length=1, max_length=10)
    product_smiles: list[str] = Field(..., description="SMILES strings of products", min_length=1, max_length=10)
    solvent: Optional[str] = Field(None, description="Solvent for ALPB solvation model")
    temperature: float = Field(298.15, description="Temperature in K", gt=0)

class ReactionThermoResponse(BaseModel):
    reactant_smiles: list[str]
    product_smiles: list[str]
    delta_e_kcal: Optional[float] = None
    delta_h_kcal: Optional[float] = None
    delta_g_kcal: Optional[float] = None
    t_delta_s_kcal: Optional[float] = None
    k_eq: Optional[float] = None
    spontaneous: Optional[bool] = None
    temperature_k: float
    species_data: list[dict] = []
    confidence: str
    confidence_note: str = ""
    has_imaginary_frequencies: bool = False
    solvent: Optional[str] = None
    method: str
    wall_time_seconds: Optional[float]
    warnings: list[str] = []

class BatchRequest(BaseModel):
    smiles_list: list[str] = Field(..., description="List of SMILES strings", max_length=100)

class BatchPkaResponse(BaseModel):
    results: list[Optional[PkaResponse]]
    count: int
    elapsed_ms: int

class BatchSolubilityResponse(BaseModel):
    results: list[Optional[SolubilityResponse]]
    count: int
    elapsed_ms: int


# --- Model weights ---

MODELS_DIR = Path(os.getenv("MODELS_DIR", "/app/models"))


def _prepare_weights():
    """Make trained model weights available under MODELS_DIR before predictors load.

    Backend is selected with STORAGE_BACKEND:
      HF (default) — pull weights from a public Hugging Face model repo. No cloud
                     credentials required; this is the path self-hosters use.
      S3           — pull from an object-store bucket (set PROPERTIES_BUCKET, and
                     optionally PROPERTIES_PREFIX). For private deployments.
      LOCAL        — weights are already present under MODELS_DIR (baked into the
                     image or mounted); nothing to fetch.

    On any failure the weights are simply absent and the affected predictors report
    themselves unavailable (see pka.is_available) — they never serve a silent fallback.
    """
    backend = os.getenv("STORAGE_BACKEND", "HF").upper()
    if backend == "LOCAL":
        logger.info(f"STORAGE_BACKEND=LOCAL — expecting weights under {MODELS_DIR}")
        return
    if backend == "HF":
        _prepare_weights_hf()
        return
    if backend == "S3":
        _prepare_weights_s3()
        return
    logger.error(f"Unknown STORAGE_BACKEND={backend!r}; expected HF, S3, or LOCAL.")


def _prepare_weights_hf():
    """Download the public weights from Hugging Face into MODELS_DIR (no creds needed)."""
    repo = os.getenv("HF_MODEL_REPO", "NovoMCP/novomcp-properties")
    try:
        from huggingface_hub import snapshot_download
    except Exception as e:
        logger.error(f"huggingface_hub unavailable — cannot fetch weights from {repo}: {e}")
        return
    try:
        snapshot_download(repo_id=repo, repo_type="model", local_dir=str(MODELS_DIR))
        logger.info(f"Weights ready from Hugging Face {repo}")
    except Exception as e:
        logger.error(f"Failed to download weights from Hugging Face {repo}: {e}")


def _prepare_weights_s3():
    """Download weights from an object-store bucket into MODELS_DIR.

    Requires PROPERTIES_BUCKET; PROPERTIES_PREFIX defaults to 'properties-models'.
    The layout under the prefix mirrors the MODELS_DIR tree the predictors expect.
    """
    bucket = os.getenv("PROPERTIES_BUCKET")
    prefix = os.getenv("PROPERTIES_PREFIX", "properties-models")
    if not bucket:
        logger.error("STORAGE_BACKEND=S3 but PROPERTIES_BUCKET is unset — no weights fetched.")
        return
    try:
        import boto3
        client = boto3.client("s3", region_name=os.getenv("AWS_REGION", "us-east-1"))
    except Exception as e:
        logger.error(f"boto3 unavailable — cannot fetch weights from bucket {bucket}: {e}")
        return

    def _get(key: str, local_path: Path) -> bool:
        try:
            local_path.parent.mkdir(parents=True, exist_ok=True)
            client.download_file(bucket, key, str(local_path))
            return True
        except Exception as e:
            logger.warning(f"Checkpoint not found: {key} ({e})")
            return False

    def _get_prefix(key_prefix: str, local_dir: Path, exts: tuple) -> int:
        try:
            paginator = client.get_paginator("list_objects_v2")
            keys = [obj["Key"]
                    for page in paginator.paginate(Bucket=bucket, Prefix=key_prefix)
                    for obj in (page.get("Contents") or [])
                    if obj["Key"].lower().endswith(exts)]
            if not keys:
                return 0
            local_dir.mkdir(parents=True, exist_ok=True)
            return sum(_get(k, local_dir / k.split("/")[-1]) for k in keys)
        except Exception as e:
            logger.warning(f"List/Download failed for {key_prefix} ({e})")
            return 0

    singletons = {
        f"{prefix}/pka-v2/best_model.ckpt":      MODELS_DIR / "pka_chemprop" / "best_model.ckpt",
        f"{prefix}/pka-v3/best_model.ckpt":      MODELS_DIR / "pka-v3" / "best_model.ckpt",
        f"{prefix}/pka-v1/best_model.ckpt":      MODELS_DIR / "pka-v1" / "best_model.ckpt",
        f"{prefix}/solubility/best_model.ckpt":  MODELS_DIR / "solubility_chemprop" / "best_model.ckpt",
    }
    for k, p in singletons.items():
        _get(k, p)
    for sub, ldir in {
        f"{prefix}/pka_v9_nochg/": MODELS_DIR / "pka_v9_nochg",
        f"{prefix}/pka_v9_chg/":   MODELS_DIR / "pka_v9_chg",
    }.items():
        _get_prefix(sub, ldir, (".pt",))
    for sub, ldir in {
        f"{prefix}/pka_v2_ensemble/": MODELS_DIR / "pka_v2_ensemble",
        f"{prefix}/pka_v3_ensemble/": MODELS_DIR / "pka_v3_ensemble",
    }.items():
        _get_prefix(sub, ldir, (".ckpt",))


# --- Startup ---

@app.on_event("startup")
async def startup_event():
    logger.info("Starting NovoMCP Properties Service...")
    t0 = time.time()

    # Fetch trained model weights (Hugging Face by default; see _prepare_weights)
    _prepare_weights()

    # Set model paths via env vars so predictors find them
    pka_ckpt = Path("/app/models/pka_chemprop/best_model.ckpt")
    sol_ckpt = Path("/app/models/solubility_chemprop/best_model.ckpt")
    if pka_ckpt.exists():
        os.environ.setdefault("PKA_MODEL_PATH", str(pka_ckpt))
    if sol_ckpt.exists():
        os.environ.setdefault("SOLUBILITY_MODEL_PATH", str(sol_ckpt))

    pka.initialize()
    pka_ok = pka.is_available()   # honest: empirical-only (missing weights) is not "ready"
    sol_ok = solubility.initialize()
    bde_ok = bde.initialize()

    elapsed = round((time.time() - t0) * 1000)
    logger.info(
        f"Initialization complete in {elapsed}ms — "
        f"pKa: {'ready' if pka_ok else 'FAILED'}, "
        f"Solubility: {'ready' if sol_ok else 'FAILED'}, "
        f"BDE: {'ready' if bde_ok else 'FAILED'}"
    )
    if not pka_ok:
        logger.error(
            "pKa is UNAVAILABLE — trained weights did not load, so /api/predict-pka, "
            "/api/batch-pka and the pKa portion of /api/predict-all will return 503. "
            "Set PKA_ALLOW_EMPIRICAL=1 only if you knowingly want labeled empirical output."
        )


# --- Health ---

@app.get("/health")
async def health():
    predictors = {
        "pka": pka.get_info(),
        "solubility": solubility.get_info(),
        "bde": bde.get_info(),
    }
    ready_count = sum(1 for p in predictors.values() if p["ready"])
    total = len(predictors)

    status = "healthy" if ready_count >= 2 else "degraded" if ready_count >= 1 else "unhealthy"
    code = 200 if status != "unhealthy" else 503

    from fastapi.responses import JSONResponse
    return JSONResponse(
        status_code=code,
        content={
            "status": status,
            "service": "novomcp-properties",
            "version": "1.0.0",
            "port": PORT,
            "predictors": predictors,
            "ready": f"{ready_count}/{total}",
        },
    )


@app.get("/")
async def root():
    return {"service": "novomcp-properties", "version": "1.0.0"}


# --- pKa ---

_PKA_UNAVAILABLE_DETAIL = (
    "pKa model weights are unavailable — the service could not load trained "
    "checkpoints, so it returns no prediction rather than emit a crude estimate "
    "silently. Wire the model weights (see docs/deploying-services/novomcp-properties.md), "
    "or set PKA_ALLOW_EMPIRICAL=1 to explicitly opt into clearly-labeled empirical output."
)


@app.post("/api/predict-pka", response_model=PkaResponse)
async def predict_pka(req: PkaRequest, _key=Header(None, alias="x-api-key")):
    if API_KEY and _key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")

    if not pka.is_available():
        raise HTTPException(status_code=503, detail=_PKA_UNAVAILABLE_DETAIL)

    result = pka.predict(req.smiles)
    if result is None:
        raise HTTPException(status_code=422, detail=f"Could not predict pKa for: {req.smiles}")

    return PkaResponse(
        smiles=result.smiles,
        pka_values=result.pka_values,
        ionizable_groups=result.ionizable_groups,
        method=result.method,
        confidence=result.confidence,
        uncertainty=result.uncertainty,
        model_version=result.model_version,
    )


@app.post("/api/batch-pka", response_model=BatchPkaResponse)
async def batch_pka(req: BatchRequest, _key=Header(None, alias="x-api-key")):
    if API_KEY and _key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")

    if not pka.is_available():
        raise HTTPException(status_code=503, detail=_PKA_UNAVAILABLE_DETAIL)

    t0 = time.time()
    results = []
    for smi in req.smiles_list:
        r = pka.predict(smi)
        if r:
            results.append(PkaResponse(
                smiles=r.smiles, pka_values=r.pka_values,
                ionizable_groups=r.ionizable_groups, method=r.method, confidence=r.confidence,
                uncertainty=r.uncertainty, model_version=r.model_version,
            ))
        else:
            results.append(None)

    return BatchPkaResponse(
        results=results,
        count=len(results),
        elapsed_ms=round((time.time() - t0) * 1000),
    )


# --- Solubility ---

@app.post("/api/predict-solubility", response_model=SolubilityResponse)
async def predict_solubility(req: SolubilityRequest, _key=Header(None, alias="x-api-key")):
    if API_KEY and _key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")

    result = solubility.predict(req.smiles, req.temperature_k)
    if result is None:
        raise HTTPException(status_code=422, detail=f"Could not predict solubility for: {req.smiles}")

    return SolubilityResponse(
        smiles=result.smiles,
        logs=result.logs,
        solubility_mg_ml=result.solubility_mg_ml,
        temperature_k=result.temperature_k,
        category=result.category,
        method=result.method,
        confidence=result.confidence,
    )


@app.post("/api/batch-solubility", response_model=BatchSolubilityResponse)
async def batch_solubility(req: BatchRequest, _key=Header(None, alias="x-api-key")):
    if API_KEY and _key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")

    t0 = time.time()
    results = []
    for smi in req.smiles_list:
        r = solubility.predict(smi)
        if r:
            results.append(SolubilityResponse(
                smiles=r.smiles, logs=r.logs, solubility_mg_ml=r.solubility_mg_ml,
                temperature_k=r.temperature_k, category=r.category,
                method=r.method, confidence=r.confidence,
            ))
        else:
            results.append(None)

    return BatchSolubilityResponse(
        results=results,
        count=len(results),
        elapsed_ms=round((time.time() - t0) * 1000),
    )


# --- BDE ---

@app.post("/api/predict-bde", response_model=BdeResponse)
async def predict_bde(req: BdeRequest, _key=Header(None, alias="x-api-key")):
    if API_KEY and _key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")

    result = bde.predict(req.smiles)
    if result is None:
        raise HTTPException(status_code=422, detail=f"Could not predict BDE for: {req.smiles}")

    return BdeResponse(
        smiles=result.smiles,
        bonds=result.bonds,
        weakest_bond=result.weakest_bond,
        method=result.method,
    )


# --- Combined prediction ---

class CombinedRequest(BaseModel):
    smiles: str
    temperature_k: float = 298.15
    include_pka: bool = True
    include_solubility: bool = True
    include_bde: bool = True

class CombinedResponse(BaseModel):
    smiles: str
    pka: Optional[PkaResponse] = None
    solubility: Optional[SolubilityResponse] = None
    bde: Optional[BdeResponse] = None
    elapsed_ms: int


@app.post("/api/predict-all", response_model=CombinedResponse)
async def predict_all(req: CombinedRequest, _key=Header(None, alias="x-api-key")):
    if API_KEY and _key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")

    t0 = time.time()
    resp = CombinedResponse(smiles=req.smiles, elapsed_ms=0)

    if req.include_pka:
        if not pka.is_available():
            raise HTTPException(status_code=503, detail=_PKA_UNAVAILABLE_DETAIL)
        r = pka.predict(req.smiles)
        if r:
            resp.pka = PkaResponse(
                smiles=r.smiles, pka_values=r.pka_values,
                ionizable_groups=r.ionizable_groups, method=r.method, confidence=r.confidence,
            )

    if req.include_solubility:
        r = solubility.predict(req.smiles, req.temperature_k)
        if r:
            resp.solubility = SolubilityResponse(
                smiles=r.smiles, logs=r.logs, solubility_mg_ml=r.solubility_mg_ml,
                temperature_k=r.temperature_k, category=r.category,
                method=r.method, confidence=r.confidence,
            )

    if req.include_bde:
        r = bde.predict(req.smiles)
        if r:
            resp.bde = BdeResponse(
                smiles=r.smiles, bonds=r.bonds, weakest_bond=r.weakest_bond, method=r.method,
            )

    resp.elapsed_ms = round((time.time() - t0) * 1000)
    return resp


# --- Reaction Thermodynamics (Materials Science — Catalysis Feasibility) ---

@app.post("/api/predict-reaction-thermo", response_model=ReactionThermoResponse)
async def predict_reaction_thermo_endpoint(req: ReactionThermoRequest, _key=Header(None, alias="x-api-key")):
    if API_KEY and _key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")

    result = reaction_thermo.predict(
        reactant_smiles=req.reactant_smiles,
        product_smiles=req.product_smiles,
        solvent=req.solvent,
        temperature=req.temperature,
    )

    if not result.success:
        raise HTTPException(status_code=500, detail=result.error or "Reaction thermodynamics prediction failed")

    return ReactionThermoResponse(
        reactant_smiles=result.reactant_smiles,
        product_smiles=result.product_smiles,
        delta_e_kcal=result.delta_e_kcal,
        delta_h_kcal=result.delta_h_kcal,
        delta_g_kcal=result.delta_g_kcal,
        t_delta_s_kcal=result.t_delta_s_kcal,
        k_eq=result.k_eq,
        spontaneous=result.spontaneous,
        temperature_k=result.temperature_k,
        species_data=result.species_data,
        confidence=result.confidence,
        confidence_note=result.confidence_note,
        has_imaginary_frequencies=result.has_imaginary_frequencies,
        solvent=result.solvent,
        method=result.method,
        wall_time_seconds=result.wall_time_seconds,
        warnings=result.warnings,
    )


# --- Redox Potential (Materials Science — Electrolyte Screening) ---

@app.post("/api/predict-redox-potential", response_model=RedoxResponse)
async def predict_redox_potential_endpoint(req: RedoxRequest, _key=Header(None, alias="x-api-key")):
    if API_KEY and _key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")

    result = redox.predict(req.smiles, solvent=req.solvent, reference_electrode=req.reference_electrode)

    if not result.success:
        raise HTTPException(status_code=500, detail=result.error or "Redox prediction failed")

    return RedoxResponse(
        smiles=result.smiles,
        ip_adiabatic_ev=result.ip_adiabatic_ev,
        ip_vertical_ev=result.ip_vertical_ev,
        ea_adiabatic_ev=result.ea_adiabatic_ev,
        ea_vertical_ev=result.ea_vertical_ev,
        oxidation_potential_v=result.oxidation_potential_v,
        reduction_potential_v=result.reduction_potential_v,
        reference_electrode=result.reference_electrode,
        stability_windows=result.stability_windows,
        electrochemical_window_v=result.electrochemical_window_v,
        solvent=result.solvent,
        solvent_class=result.solvent_class,
        method=result.method,
        accuracy_note=result.accuracy_note,
        wall_time_seconds=result.wall_time_seconds,
        warnings=result.warnings,
    )


# --- Frontier Orbitals (Materials Science Phase 1a) ---

@app.post("/api/predict-frontier-orbitals", response_model=FrontierOrbitalsResponse)
async def predict_frontier_orbitals_endpoint(req: FrontierOrbitalsRequest, _key=Header(None, alias="x-api-key")):
    if API_KEY and _key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")

    result = frontier_orbitals.predict(req.smiles, solvent=req.solvent)

    if not result.success:
        raise HTTPException(status_code=500, detail=result.error or "Frontier orbital prediction failed")

    return FrontierOrbitalsResponse(
        smiles=result.smiles,
        homo_ev=result.homo_ev,
        lumo_ev=result.lumo_ev,
        gap_ev=result.gap_ev,
        dipole_debye=result.dipole_debye,
        emission_wavelength_nm=result.emission_wavelength_nm,
        emission_color=result.emission_color,
        triplet_energy_ev=result.triplet_energy_ev,
        singlet_triplet_gap_ev=result.singlet_triplet_gap_ev,
        oled_classification=result.oled_classification,
        oled_rationale=result.oled_rationale,
        oled_motifs=result.oled_motifs,
        method=result.method,
        wall_time_seconds=result.wall_time_seconds,
    )


if __name__ == "__main__":
    logger.info(f"Starting NovoMCP Properties on port {PORT}")
    uvicorn.run(app, host="0.0.0.0", port=PORT)

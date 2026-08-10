# novomcp-properties

Trained ML models for physicochemical properties — **pKa**, **aqueous solubility**, and
**bond dissociation energy (BDE)** — served over a small FastAPI app. CPU-only. Part of the
open [NovoMCP](https://github.com/NovoMCP) computational-chemistry engine; runs standalone.

Model weights are hosted on Hugging Face and downloaded on first start — **no cloud
credentials required.** Solubility and BDE work out of the box; **pKa weights are
NonCommercial** and opt-in (see [pKa weights](#pka-weights-noncommercial)).

## Quick start

```bash
docker run -p 8030:8030 ghcr.io/novomcp/novomcp-properties:latest
```

On boot the service pulls the permissive (solubility) weights from Hugging Face
([`NovoMCP/novomcp-properties`](https://huggingface.co/NovoMCP/novomcp-properties)) into
`/app/models`, then serves:

```bash
curl -s localhost:8030/health
curl -s -X POST localhost:8030/api/predict-solubility \
  -H 'content-type: application/json' \
  -d '{"smiles":"CC(=O)Oc1ccccc1C(=O)O"}'
```

## Wire into the NovoMCP engine

```bash
export NOVOMCP_PROPERTIES_URL=http://localhost:8030
# For the charge-based pKa routes (sulfonamides / aromatic N–H), also run novomcp-qm:
export NOVOMCP_QM_URL=http://localhost:8031
```

The `predict_pka`, `predict_solubility`, and `predict_bde` tools then light up in the engine.

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| GET  | `/health` | Per-predictor readiness |
| POST | `/api/predict-pka` · `/api/batch-pka` | Acidic/basic ionization constants |
| POST | `/api/predict-solubility` · `/api/batch-solubility` | LogS with temperature dependence |
| POST | `/api/predict-bde` | Bond dissociation energies |
| POST | `/api/predict-all` | All three for one molecule |

## Weights backends

Selected with `STORAGE_BACKEND`:

| Value | Behavior |
|---|---|
| `HF` (default) | Pull public weights from Hugging Face (`HF_MODEL_REPO`, default `NovoMCP/novomcp-properties`). No credentials. |
| `LOCAL` | Use weights already present under `MODELS_DIR` (baked into the image or mounted). |
| `S3` | Pull from an object store — set `PROPERTIES_BUCKET` (and optionally `PROPERTIES_PREFIX`). For private deployments. |

If weights cannot be loaded, the affected predictor reports itself **unavailable** and its
endpoints return `503` — the service never serves a silent low-quality fallback. (A pKa-only
opt-in, `PKA_ALLOW_EMPIRICAL=1`, serves clearly-labeled RDKit empirical estimates on purpose.)

## pKa weights (NonCommercial)

The pKa model is trained primarily on the IUPAC Dissociation Constants (licensed
**CC-BY-NC-4.0**, NonCommercial), and also uses ChEMBL (CC-BY-SA 3.0, ShareAlike). The
pKa weights therefore combine both terms and are published separately under
**CC-BY-NC-SA-4.0** at [`NovoMCP/novomcp-pka`](https://huggingface.co/NovoMCP/novomcp-pka),
**opt-in** so commercial deployments don't pull NonCommercial weights by default.

For **non-commercial** use, enable pKa by pointing at that repo:

```bash
export HF_PKA_MODEL_REPO=NovoMCP/novomcp-pka   # non-commercial use only
```

Without it, the pKa endpoints return `503` (solubility and BDE are unaffected). The
service *code* is Apache-2.0; only the pKa *weights* carry the NonCommercial / ShareAlike terms.

## Env vars

| Var | Default | Purpose |
|---|---|---|
| `PORT` | `8030` | HTTP listen port |
| `STORAGE_BACKEND` | `HF` | `HF` \| `LOCAL` \| `S3` |
| `HF_MODEL_REPO` | `NovoMCP/novomcp-properties` | Hugging Face weights repo (permissive: solubility) |
| `HF_PKA_MODEL_REPO` | – | NonCommercial pKa weights repo (opt-in; e.g. `NovoMCP/novomcp-pka`) |
| `MODELS_DIR` | `/app/models` | Where weights are placed / read |
| `NOVOMCP_QM_URL` | – | novomcp-qm endpoint for per-atom charges (charge-based pKa routes) |
| `BATCH_SIZE` | `64` | Molecules per batch |

## Build from source

```bash
docker build -t novomcp-properties .
docker run -p 8030:8030 novomcp-properties
```

## Models

- **pKa** *(NonCommercial weights, opt-in)* — a routed ensemble: a per-atom-charge specialist
  for sulfonamides / aromatic N–H, and a general model for everything else; each route reports
  an uncertainty estimate. Benchmarked on SAMPL7. The charge-based routes use per-atom charges
  from `novomcp-qm`.
- **Solubility** — pretrained on AqSolDB, fine-tuned on BigSolDB with temperature as an input.
- **BDE** — uses the pretrained `alfabet` network (installed dependency).

## License

- **Code:** Apache-2.0 (`LICENSE`).
- **Solubility weights** (Hugging Face `NovoMCP/novomcp-properties`): Apache-2.0.
- **pKa weights** (Hugging Face `NovoMCP/novomcp-pka`): **CC-BY-NC-SA-4.0 (NonCommercial, ShareAlike)** —
  trained on the IUPAC Dissociation Constants (CC-BY-NC-4.0) and ChEMBL (CC-BY-SA 3.0).

Full training-data attribution in [`NOTICE`](./NOTICE).

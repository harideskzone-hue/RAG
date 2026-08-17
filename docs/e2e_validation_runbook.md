# VISTA AI — Canonical E2E Validation Runbook & Master Release Specification

This document defines the official, non-mocked End-to-End (E2E) validation workflow and operator runbook for VISTA AI Release Candidates.

---

## 1. Master Pipeline Architecture

```text
                 ┌──────────────────────┐
                 │   CCTV .MP4 Video    │
                 └──────────┬───────────┘
                            │
                            ▼
                 ┌──────────────────────┐
                 │ input/watch/         │
                 │ Auto Ingestion Daemon│
                 └──────────┬───────────┘
                            │
                            ▼
              ┌────────────────────────────┐
              │       CV PIPELINE          │
              │                            │
              │ Video Reader               │
              │      ↓                     │
              │ YOLO26n Detection          │
              │      ↓                     │
              │ ByteTrack Tracking         │
              │      ↓                     │
              │ Person Cropping            │
              │      ↓                     │
              │ Crop Quality Gate          │
              │      ↓                     │
              │ OSNet 512-D Re-ID          │
              │      ↓                     │
              │ IdentityResolver           │
              │ MATCHED / NEW / UNRESOLVED │
              └─────────────┬──────────────┘
                            │
             ┌──────────────┼──────────────┐
             ▼              ▼              ▼
        PostgreSQL        MongoDB        Qdrant
        Source Truth     Observations   Embeddings
             │              │              │
             └──────────────┼──────────────┘
                            ▼
                    Object Storage
                  Crops / Video Assets
                            │
                            ▼
                 Cross-Store Audit
                            │
                     PASS / FAILURE
                            │
                            ▼
                  input/completed/
                            │
                            ▼
                  ┌─────────────────┐
                  │  FastAPI /chat  │
                  └────────┬────────┘
                           │
                           ▼
                  ┌─────────────────┐
                  │   Intent LLM    │
                  │ Semantic only   │
                  └────────┬────────┘
                           │
                     QueryIntent
                           │
                           ▼
                  ┌─────────────────┐
                  │ Deterministic   │
                  │ Planner/Retrieval│
                  └────────┬────────┘
                           │
                           ▼
                VerifiedResultContract
                           │
                           ▼
                  ┌─────────────────┐
                  │ Response LLM    │
                  │ Natural language│
                  └────────┬────────┘
                           │
                           ▼
                  Grounding Validator
                           │
                ┌──────────┴──────────┐
                ▼                     ▼
             VALID                  ABSTAIN
                │                     │
                └──────────┬──────────┘
                           ▼
                    FastAPI Response
                           │
                           ▼
                     React UI
```

---

## 2. Infrastructure Setup & Database Bootstrap

```bash
cd /Users/hariharans/Documents/longgraph/vista_agentic_ai

# 1. Start Docker containers (PostgreSQL: 5433, MongoDB: 27017, Qdrant: 6333, MinIO: 9000)
DOCKER_HOST="unix://${HOME}/.colima/default/docker.sock" \
docker compose -f deployment/docker/docker-compose.e2e.yml up -d

# 2. Bootstrap schemas, cameras, and Qdrant collections
MODE=docker \
POSTGRES_URI="postgresql+asyncpg://postgres:postgres@localhost:5433/vista" \
python3 scripts/init_databases.py
```

---

## 3. Database Connectivity Verification

Before processing video, verify each store independently. **Do not continue if any store is offline.**

```bash
# PostgreSQL Check
MODE=docker \
POSTGRES_URI="postgresql+asyncpg://postgres:postgres@localhost:5433/vista" \
python3 -c "import asyncio, asyncpg; asyncio.run(asyncpg.connect('postgresql://postgres:postgres@localhost:5433/vista')); print('PostgreSQL: OK')"

# Qdrant Check
curl -s http://localhost:6333/collections

# Object Storage Check
ls -lah dataset/persons
```

---

## 4. Run the Auto Ingestion Daemon

### Terminal 1: Ingestion Daemon
```bash
MODE=docker \
POSTGRES_URI="postgresql+asyncpg://postgres:postgres@localhost:5433/vista" \
python3 scripts/auto_ingest_daemon.py
```

### Terminal 2: Drop Real CCTV Video
```bash
cp input/completed/VIDEO-2026-08-13-14-20-13.mp4 input/watch/
```

*Lifecycle Guarantee*:
`input/watch/` → `input/processing/` → CV / Re-ID / Multi-Store Insertion → Cross-Store Audit → `input/completed/` (or `input/failed/` on unrecoverable error).

---

## 5. Cross-Store Referential Audit

Verify that PostgreSQL, MongoDB, Qdrant, and Object Storage have synchronized records with **zero orphan data**:

```bash
MODE=docker \
POSTGRES_URI="postgresql+asyncpg://postgres:postgres@localhost:5433/vista" \
python3 scripts/cross_store_audit.py --video-id VIDEO-2026-08-13-14-20-13.mp4
```

---

## 6. Launch FastAPI Backend & React UI

### Terminal 3: FastAPI Backend
```bash
MODE=docker \
POSTGRES_URI="postgresql+asyncpg://postgres:postgres@localhost:5433/vista" \
uvicorn app.app:create_app --factory --host 0.0.0.0 --port 8000 --reload
```

### Terminal 4: React UI
```bash
cd frontend
npm run dev
```
Open **`http://localhost:5173`** in your browser.

---

## 7. Pure Semantic Agentic RAG Testing (Zero Keywords)

### A. Lexical Invariance / Semantic Ablation
Test equivalent questions with different wording. All four must resolve to the identical `VerifiedResultContract`:
1. *"How many people were near the entrance?"*
2. *"Give me the number of individuals observed around the entrance."*
3. *"What was the total human presence at the entrance?"*
4. *"Could you quantify the human presence in that area?"*

### B. Supported Temporal Queries
* *"Show me the movement timeline for the observed person."*
* *"When was person P001 first and last observed?"*

### C. Factual Abstention (Unsupported Metadata)
* *"What is the person's name or social security number?"*
* **Expected Result**: **ABSTAIN** (because the database contract contains no identity names).

### D. Grounding Defense (Hallucination Rejection)
* If the Response LLM claims facts absent from the `VerifiedResultContract`, the **GroundingValidatorNode** intercepts and blocks the statement, preventing fabricated data from ever reaching the React UI.

---

## 8. Master Automated Test Suite & Release Matrix

```bash
# Complete PyTest Suite
PYTHONPATH=. pytest -v

# 31-Point Master Release Validation Matrix
MODE=docker \
POSTGRES_URI="postgresql+asyncpg://postgres:postgres@localhost:5433/vista" \
python3 scripts/master_e2e_validation.py
```

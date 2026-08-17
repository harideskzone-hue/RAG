# 🛡️ VISTA AI — Video Intelligence & Surveillance Thinking Agent

VISTA AI is an enterprise-grade, multi-agent Computer Vision and Agentic RAG platform designed for 24/7 physical security, multi-camera person tracking (Re-ID), automated event metadata extraction, and forensic video investigation.

---

## 🏛️ End-to-End System Architecture

```
                        VISTA AI COMPLETE PIPELINE ARCHITECTURE

  ┌─────────────────────────┐
  │  24/7 RTSP / IP CAM     │
  │     (Hikvision/Dahua)   │
  └────────────┬────────────┘
               │ Live RTSP Feed (H.264/H.265)
               ▼
  ┌─────────────────────────┐
  │  RTSP Stream Chunker    │ ──► [input/recording/temp_*.mp4]
  │  (10-minute slicing)    │ ──► Atomic Promotion (st_size > 1024)
  └────────────┬────────────┘
               │ [input/watch/<cam_id>_<timestamp>.mp4]
               ▼
  ┌─────────────────────────┐
  │  Auto Ingestion Daemon  │ ──► Compute SHA-256 & Telemetry
  │  (State Controller)     │ ──► Transitions: READY ➔ PROCESSING ➔ CV_COMPLETE ➔ ...
  └────────────┬────────────┘
               │
               ▼
  ┌────────────────────────────────────────────────────────────────────────┐
  │                       COMPUTER VISION PIPELINE                         │
  │                                                                        │
  │  ┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐  │
  │  │  YOLO26n /       │ ──►│   ByteTrack /    │ ──►│ Crop Quality Gate│  │
  │  │  PeopleNet       │    │   BoT-SORT       │    │ Blur/Skin/Aspect │  │
  │  └──────────────────┘    └──────────────────┘    └─────────┬────────┘  │
  │                                                            │           │
  │  ┌──────────────────┐    ┌──────────────────┐              │           │
  │  │ IdentityResolver │ ◄──│ OSNet / SOLIDER  │ ◄────────────┘           │
  │  │ (0.82 / 0.05)    │    │ 512-D Embedding  │                          │
  │  └────────┬─────────┘    └──────────────────┘                          │
  └───────────┼────────────────────────────────────────────────────────────┘
              │
              │ (Resolved Canonical Identities: MATCHED / NEW / UNRESOLVED)
              ▼
  ┌────────────────────────────────────────────────────────────────────────┐
  │                     MULTI-STORE PERSISTENCE LAYER                      │
  │                                                                        │
  │  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌───────┐ │
  │  │ PostgreSQL   │    │  MongoDB     │    │ Qdrant /     │    │Object │ │
  │  │ Source Truth │    │  Telemetry   │    │ Vector Store │    │Storage│ │
  │  └──────┬───────┘    └──────┬───────┘    └──────┬───────┘    └───┬───┘ │
  └─────────┼───────────────────┼───────────────────┼────────────────┼─────┘
            │                   │                   │                │
            └───────────────────┴─────────┬─────────┴────────────────┘
                                          │
                                          ▼
                             ┌─────────────────────────┐
                             │    Cross-Store Audit    │
                             │  (Referential Checklist)│
                             └────────────┬────────────┘
                                          │
                               ┌──────────┴──────────┐
                               ▼                     ▼
                        [AUDIT PASS]           [AUDIT FAIL]
                               │                     │
                    Post-Verification Safe       Quarantine to
                     Video Cleanup (retain      [input/failed/]
                      crops & metadata)
                               │
                               ▼
  ┌────────────────────────────────────────────────────────────────────────┐
  │                    AGENTIC RAG REASONING & SEARCH                      │
  │                                                                        │
  │  User Query (React UI / API) ➔ Hybrid Intent Classifier                 │
  │                                      │                                 │
  │                                      ▼                                 │
  │                         Execution Plan (Deterministic)                 │
  │                                      │                                 │
  │                                      ▼                                 │
  │                          Metadata & Evidence Fusion                    │
  │                                      │                                 │
  │                                      ▼                                 │
  │                       VerifiedResultContract (Contract)                │
  │                                      │                                 │
  │                                      ▼                                 │
  │                        LLM Reasoning (Synthesizer)                     │
  │                                      │                                 │
  │                                      ▼                                 │
  │                    Grounding Validator (0 Hallucinations)              │
  └──────────────────────────────────────┬─────────────────────────────────┘
                                         │
                                         ▼
                            FastAPI Server (:8000)
                                         │
                                         ▼
                            React Dashboard (:5173)
```

---

## ⚙️ Computer Vision & Re-ID Stack

| Component | Model / Algorithm | Description |
|---|---|---|
| **Person Detection** | **YOLO26n / NVIDIA PeopleNet** | High-precision bounding box detection tuned for CCTV surveillance. |
| **Multi-Object Tracking** | **ByteTrack / BoT-SORT** | Kalman Filtering + Global Motion Compensation to maintain continuous trajectories (`P001`, `P002`...). |
| **Crop Quality Gate** | **Laplacian Sharpness & Skin-Tone Filter** | Filters out blurry, tiny, or dark artifacts to select top 5–8 high-resolution face crops per person. |
| **Person Re-ID** | **OSNet MSMT17 / SOLIDER TAO Re-ID** | Extracts 512-dimensional L2-normalized feature vectors for multi-camera person search. |
| **Identity Resolver** | **Cosine Matcher (0.82 / 0.05)** | Matches tracklets across cameras/time (`MATCHED`, `NEW`, `UNRESOLVED`). |

---

## 🛠️ Step-by-Step Installation & Run Guidance

### 1. Prerequisites
- **Python**: `3.10+`
- **Node.js**: `18.0+` (for React Dashboard)
- **OS**: macOS, Linux, or Windows (WSL2)

---

### 2. Environment Setup

```bash
# 1. Clone repository
git clone https://github.com/harideskzone-hue/RAG.git
cd RAG

# 2. Create Python virtual environment
python3 -m venv venv
source venv/bin/activate

# 3. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

---

### 3. Configuration (`.env`)

Ensure your `.env` file contains valid model and API settings:

```bash
CV_MODEL_DIR=models
CV_DETECTOR_MODEL=yolo26n.pt
CV_DEVICE=cpu
CV_TRACKER_CONFIG=bytetrack.yaml

LLM_PROVIDER=groq
GROQ_API_KEY=your_groq_api_key_here
REASONING_PROVIDER=groq

POSTGRES_URI=postgresql+asyncpg://vista:vista123@localhost:5433/vista_db
MONGO_URI=mongodb://localhost:27017/
QDRANT_HOST=localhost
QDRANT_PORT=6333
```

---

### 4. Launch VISTA AI Server & Dashboard

To start both the FastAPI backend and React frontend simultaneously:

```bash
./run_vista.sh
```

- **Backend API**: `http://localhost:8000` (Health: `http://localhost:8000/api/v1/health`)
- **React Dashboard**: `http://localhost:5173`

---

## 🚀 24/7 Production Operations & CLI Utilities

### A. Ingest Any Custom Video Immediately
```bash
# Process a video clip with full CV extraction, Re-ID, and Audit:
python3 scripts/run_fast_ingest.py
```

### B. Connect a Live 24/7 RTSP / NVR Stream
```bash
# Stream from IP Camera or NVR and slice into 10-minute segments:
python3 scripts/cctv_stream_chunker.py "rtsp://admin:password@192.168.1.100:554/Streaming/Channels/101" cam_entrance_01
```

### C. Launch 24/7 Auto-Ingestion Watch Daemon
```bash
# Automatically ingests any video dropped into input/watch/:
python3 scripts/auto_ingest_daemon.py
```

### D. Verify Agentic RAG Pipeline (Zero Hallucination)
```bash
python3 scripts/verify_agentic_rag.py
```

### E. Run Cross-Store Referential Audit
```bash
python3 scripts/cross_store_audit.py
```

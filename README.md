# DermWise — AI Skin Lesion Analysis

> A full-stack medical-AI application that classifies dermoscopic skin-lesion images into 7
> diagnostic categories and generates a grounded, structured clinical report. Built as a
> demonstration of an end-to-end **vision → retrieval → language** AI pipeline.

**⚠️ For research and educational use only. Not a medical device and not for diagnosis.**

---

## 🎯 What it does

Upload a dermoscopic image → DermWise runs a **three-stage AI pipeline** and returns a
classification, confidence, top-3 predictions, and an AI-generated clinical report grounded in a
curated medical knowledge base.

1. **Classification** — EfficientNet-B0 (transfer learning) with **Test-Time Augmentation (TTA)**
2. **Knowledge Retrieval** — **RAG** via FAISS + Sentence-Transformers over 53 curated medical chunks
3. **Report Generation** — an instruction-tuned LLM writes a structured 4-section report grounded in the retrieved context

**Supported classes (HAM10000 dataset):**

| Code | Lesion | Severity |
|---|---|---|
| AKIEC | Actinic Keratosis | Pre-cancerous |
| BCC | Basal Cell Carcinoma | Malignant |
| BKL | Benign Keratosis | Benign |
| DF | Dermatofibroma | Benign |
| MEL | Melanoma | Malignant |
| NV | Melanocytic Nevi | Benign |
| VASC | Vascular Lesions | Benign |

---

## 🏗️ Architecture

**Frontend:** Next.js 14 (App Router) + React 18 + TailwindCSS → deployed on **Vercel**
**Backend:** Python + Gradio (3-stage AI pipeline) → deployed on **HuggingFace Spaces (Docker, free CPU)**
**Bridge:** a server-side Next.js proxy (`/api/analyze`) hides the backend URL and avoids CORS.

```mermaid
graph TD
    User["👤 User (Browser)"]
    subgraph Vercel["Vercel — Next.js 14"]
        FE["Dashboard UI<br/>upload + render"]
        Proxy["/api/analyze<br/>(server-side proxy / BFF)"]
    end
    subgraph HF["HuggingFace Space — Docker + Gradio"]
        Pipe["analyze() pipeline"]
        CNN["EfficientNet-B0 + TTA<br/>(best_model.pth)"]
        RAG["FAISS + MiniLM<br/>(faiss_index.bin + knowledge_base.json)"]
    end
    LLM["HF Serverless Inference API<br/>Qwen/Qwen2.5-7B-Instruct"]

    User -->|"JPG/PNG"| FE --> Proxy
    Proxy -->|"Gradio 2-step API (POST + SSE)"| Pipe
    Pipe --> CNN
    Pipe --> RAG
    Pipe -->|"prompt + retrieved context"| LLM
    LLM -->|"report text"| Pipe
    Pipe -->|"JSON result"| Proxy --> FE --> User
```

**Pipeline data flow:**

```mermaid
graph LR
    IMG["Image (RGB)"] --> TTA["EfficientNet-B0<br/>4-view TTA → avg softmax"]
    TTA --> PRED["class + confidence + top-3"]
    PRED --> Q["query = '{class} skin lesion dermoscopy'"]
    Q --> EMB["MiniLM embed (384-d)"]
    EMB --> FAISS["FAISS top-3 chunks"]
    PRED --> PROMPT["LLM prompt"]
    FAISS --> PROMPT
    PROMPT --> REPORT["Qwen2.5-7B report"]
    REPORT --> FIX["medical fact-correction pass"]
    FIX --> OUT["structured clinical report"]
```

> A deeper architectural walkthrough (request lifecycle, design patterns, technical decisions) lives
> in [`PROJECT_DEEP_DIVE.md`](PROJECT_DEEP_DIVE.md).

---

## 🤖 AI pipeline details

### Phase 1 — EfficientNet-B0 classifier
- **Input:** 224×224 dermoscopic image, ImageNet normalization
- **Backbone:** EfficientNet-B0 (transfer learning) with a custom `Dropout(0.3) + Linear(7)` head
- **Test-Time Augmentation:** 4 views (original, h-flip, v-flip, rotate 90°); softmax probabilities
  are averaged across views for a more robust prediction
- **Output:** predicted class + top-3 probabilities

### Phase 2 — FAISS knowledge retrieval (RAG)
- **Embedder:** `all-MiniLM-L6-v2` (384-dim sentence embeddings)
- **Index:** FAISS over 53 curated medical knowledge chunks
- **Design choice:** the retrieval query is built from the **classifier's predicted class**
  (`"{class} skin lesion dermoscopy"`), so the LLM receives facts relevant to what was detected —
  this also incidentally hardens the system against prompt injection (the user supplies no free text
  to the prompt)
- **Output:** top-3 relevant medical facts as grounding context

### Phase 3 — Clinical report generation

> **Design note — TinyLlama (fine-tuned) vs. Qwen (served), read this:**
> TinyLlama-1.1B-Chat was **fine-tuned with QLoRA** for clinical report generation, and the trained
> adapter ships in [`huggingface/models/lora_adapter/`](huggingface/models/lora_adapter/) (training
> code in [`/training`](training/)). **The live demo, however, generates reports via the HuggingFace
> Serverless Inference API using `Qwen/Qwen2.5-7B-Instruct`.**
>
> This was a deliberate **quality/latency trade-off**: a 7B instruction-tuned model produces
> materially better structured medical reports than a 1.1B model running on the free CPU tier, at
> zero hosting cost. The QLoRA adapter is retained as (a) evidence of the fine-tuning work and
> (b) a ready path to fully local inference. See `generate_report()` in
> [`huggingface/app.py`](huggingface/app.py).

- **Served model:** `Qwen/Qwen2.5-7B-Instruct` (`max_tokens=512, temperature=0.2, top_p=0.85`)
- **Fine-tuned model (shipped, not served):** TinyLlama-1.1B + QLoRA (4-bit, r=16, α=32)
- **Guardrail:** a post-generation fact-correction pass scrubs known dangerous misstatements
  (e.g. "melanoma is always benign")
- **Fallback:** if the Inference API is unavailable, a deterministic template-based report is returned
- **Output sections:** Classification Summary · Clinical Description · Risk Assessment · Recommended Actions

---

## 📊 Results & evaluation

> _Evaluation metrics are being added (see roadmap). This section will report overall accuracy,
> macro-F1, per-class precision/recall (with emphasis on melanoma sensitivity), the confusion
> matrix, and the measured TTA vs. no-TTA improvement, computed on the held-out HAM10000 test split._

| Metric | Value |
|---|---|
| Overall accuracy | _TBD_ |
| Macro F1 | _TBD_ |
| Melanoma (MEL) recall | _TBD_ |
| TTA improvement (Δ accuracy) | _TBD_ |

_Confusion matrix and per-class breakdown: see [`huggingface/eval.py`](huggingface/eval.py) (coming soon)._

---

## 🚀 Quick start

### Prerequisites
- Node.js 18+ and npm
- (Optional, to run the AI backend locally) Python 3.10+

### 1. Frontend

```bash
npm install
cp .env.example .env.local        # set HF_SPACE_URL=https://<your-space>.hf.space
npm run dev                        # http://localhost:3000  →  /dashboard
```

### 2. AI backend (optional — to run the Space locally)

```bash
cd huggingface
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt gradio
python app.py                      # Gradio on http://localhost:7860
# then point the frontend at it: HF_SPACE_URL=http://localhost:7860
```

---

## 📦 Project structure

```
DREMWISE/
├── app/
│   ├── page.tsx                 # Landing page
│   ├── dashboard/page.tsx       # Upload & analysis UI
│   └── api/analyze/route.ts     # Server-side proxy → HF Space (hides URL, dodges CORS)
├── components/                  # React UI (ReportDisplay renders the AnalysisResult)
├── lib/api.ts                   # Typed frontend API client + data contract
├── huggingface/                 # AI backend (separate deploy target)
│   ├── app.py                   # Gradio app — 3-stage pipeline
│   ├── Dockerfile               # CPU-only torch build for HF Spaces
│   ├── requirements.txt
│   └── models/                  # best_model.pth · faiss_index.bin · knowledge_base.json · lora_adapter/
├── training/                    # Training & fine-tuning notebooks (classifier, QLoRA, RAG index)  [roadmap]
├── PROJECT_DEEP_DIVE.md         # Full architecture review
└── README.md
```

---

## 🛠️ Tech stack

| Layer | Technology | Purpose |
|---|---|---|
| **Frontend** | Next.js 14, React 18, TailwindCSS, lucide-react | UI + server-side proxy |
| **Backend** | Gradio (on FastAPI/Uvicorn), Docker | Expose the pipeline as a web API |
| **Vision** | PyTorch, torchvision, EfficientNet-B0 | Image classification + TTA |
| **Retrieval** | FAISS, sentence-transformers (MiniLM) | RAG knowledge retrieval |
| **LLM** | Qwen2.5-7B-Instruct (served) · TinyLlama-1.1B + QLoRA (fine-tuned) | Clinical report generation |
| **Deployment** | Vercel (frontend) · HuggingFace Spaces (backend) | Free hosting |

---

## 🔒 Security & privacy

- **API proxy** — the HuggingFace backend URL is server-side only, never exposed to the browser
- **No data storage** — images are processed in-memory and never persisted
- **No free-text prompt path** — the LLM prompt is built from the model's prediction + a curated
  knowledge base, mitigating prompt injection
- **HTTPS** on both hosting platforms

---

## 🧭 Production considerations (future work)

This is a portfolio/demo project and intentionally omits production infrastructure. For a real
deployment the following would be added (and are discussed in
[`PROJECT_DEEP_DIVE.md`](PROJECT_DEEP_DIVE.md)):

- **Result caching** (content-hash or class-bucketed) to cut repeat latency
- **Rate limiting + authentication** on the public analyze endpoint
- **Retry/backoff** around the LLM and Gradio calls; incremental SSE streaming to the client
- **Server-side input re-validation** (magic-bytes/size/dimensions) for defense in depth
- **Always-warm hardware** to eliminate free-tier cold starts
- **Monitoring + an automated evaluation/regression harness**

---

## 📚 Dataset & acknowledgments

- **HAM10000** (10,015 dermoscopic images) — Harvard Dataverse
- **EfficientNet** — Google Research · **TinyLlama** — TinyLlama authors · **Qwen** — Alibaba
- **HuggingFace** (model hosting, Inference API, Gradio, Spaces) · **Vercel** (frontend hosting)

---

## 📝 License

For **research and educational purposes only**. Not approved for medical diagnosis or clinical
decision-making.
```
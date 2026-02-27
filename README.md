# DermAssist — AI Skin Lesion Analysis

A full-stack medical AI application for skin lesion classification and clinical report generation.

## 🎯 Overview

DermAssist uses a three-stage AI pipeline to analyze dermoscopic images:

1. **Classification** — EfficientNet-B0 with Test-Time Augmentation (TTA)
2. **Knowledge Retrieval** — FAISS + Sentence Transformers for medical context
3. **Report Generation** — TinyLlama-1.1B with QLoRA fine-tuning

**Supported Classes** (HAM10000 dataset):
- Actinic Keratosis (AKIEC) — Pre-cancerous
- Basal Cell Carcinoma (BCC) — Malignant
- Benign Keratosis (BKL) — Benign
- Dermatofibroma (DF) — Benign
- Melanoma (MEL) — Malignant
- Melanocytic Nevi (NV) — Benign
- Vascular Lesions (VASC) — Benign

---

## 🏗️ Architecture

**Frontend**: Next.js 14 (App Router) + React + TailwindCSS  
**Backend**: HuggingFace Spaces (Gradio)  
**Deployment**: Vercel (frontend) + HuggingFace Spaces (backend)

```
User Upload → Next.js (localhost/Vercel)
                ↓
          /api/analyze (proxy)
                ↓
    HuggingFace Space (Gradio)
        ↓           ↓           ↓
   EfficientNet   FAISS   TinyLlama
        ↓           ↓           ↓
   Classification → Retrieval → Report
                ↓
        Clinical Report JSON
                ↓
           Frontend Display
```

---

## 🚀 Quick Start

### Prerequisites

- Node.js 18+ and npm
- Python 3.10+ (for HuggingFace Space development)
- Git

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/dermassist.git
cd dermassist
```

### 2. Install Frontend Dependencies

```bash
npm install
```

### 3. Configure Environment Variables

```bash
cp .env.example .env.local
```

Edit `.env.local` and set:
```env
HF_SPACE_URL=https://your-username-dermwise.hf.space
```

### 4. Run Development Server

```bash
npm run dev
```

Visit http://localhost:3000

---

## 📦 Project Structure

```
DREMWISE/
├── app/                      # Next.js App Router
│   ├── page.tsx              # Landing page
│   ├── dashboard/page.tsx    # Upload & analysis
│   └── api/analyze/route.ts  # Server-side API proxy
│
├── components/               # React UI components
│   ├── Navbar.tsx
│   ├── Hero.tsx
│   ├── ReportDisplay.tsx
│   └── ...
│
├── lib/
│   └── api.ts               # Frontend API client
│
├── huggingface/             # AI backend for HF Spaces
│   ├── app.py               # Gradio app (3-stage pipeline)
│   ├── requirements.txt     # Python dependencies
│   └── models/              # Pre-trained models (67 MB)
│       ├── best_model.pth          # EfficientNet-B0
│       ├── faiss_index.bin         # Vector index
│       ├── knowledge_base.json     # 53 medical chunks
│       └── lora_adapter/           # TinyLlama QLoRA
│
├── .env.example             # Environment template
├── .gitignore
└── README.md
```

---

## 🤖 AI Pipeline Details

### Phase 1: EfficientNet-B0 Classifier

- **Input**: 224×224 dermoscopic image
- **TTA**: 4 augmentations (original, h-flip, v-flip, rotate 90°)
- **Output**: Predicted class + top-3 probabilities

### Phase 2: FAISS Knowledge Retrieval

- **Embedder**: all-MiniLM-L6-v2 (384-dim vectors)
- **Index**: 53 medical knowledge chunks
- **Output**: Top-3 relevant medical facts

### Phase 3: TinyLlama Report Generation

- **Model**: TinyLlama-1.1B-Chat-v1.0
- **Fine-tuning**: QLoRA (4-bit NF4 quantization)
- **Output**: Structured clinical report (Classification, Description, Risk, Recommendations)

---

## 🌐 Deployment

### Deploy Backend to HuggingFace Spaces

1. **Create a Space** at https://huggingface.co/new-space
   - Name: `DermWise`
   - SDK: `Gradio`
   - Hardware: `CPU basic` (free)
   - Visibility: `Public`

2. **Upload Files**:
   ```bash
   pip install huggingface_hub
   python -c "from huggingface_hub import login; login()"
   # Paste your write token
   
   python -c "
   from huggingface_hub import upload_folder
   upload_folder(
       folder_path='huggingface',
       repo_id='your-username/DermWise',
       repo_type='space'
   )
   "
   ```

3. **Wait for Build** (5-10 minutes)
   - Monitor at `https://huggingface.co/spaces/your-username/DermWise`
   - Look for "Running" status

4. **Copy Space URL**:
   ```
   https://your-username-dermwise.hf.space
   ```

### Deploy Frontend to Vercel

1. **Push to GitHub** (see below)

2. **Import to Vercel**:
   - Go to https://vercel.com/new
   - Import your GitHub repository
   - Framework: `Next.js`
   - Build Command: `npm run build`
   - Output Directory: `.next`

3. **Add Environment Variable**:
   - Key: `HF_SPACE_URL`
   - Value: `https://your-username-dermwise.hf.space`

4. **Deploy** — Vercel auto-deploys on every push

---

## 🔒 Security & Privacy

- **API Proxy**: HuggingFace URL is hidden from browsers (server-side only)
- **No Data Storage**: Images are processed in-memory and not saved
- **HTTPS Only**: All communications encrypted
- **Educational Use**: Not for medical diagnosis

---

## 🛠️ Technologies

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **Frontend** | Next.js 14 | React framework with SSR |
| | TailwindCSS | Utility-first styling |
| | Lucide React | Icon library |
| **Backend** | Gradio 5.x | Python → Web API |
| | FastAPI | (via Gradio) API server |
| **AI Models** | EfficientNet-B0 | Image classification |
| | FAISS | Vector similarity search |
| | Sentence Transformers | Text embeddings |
| | TinyLlama-1.1B | Language model |
| | QLoRA | Parameter-efficient fine-tuning |
| **Deployment** | Vercel | Frontend hosting |
| | HuggingFace Spaces | AI backend hosting |

---

## 📊 Model Performance

- **Dataset**: HAM10000 (10,015 dermoscopic images)
- **Classes**: 7 types of skin lesions
- **Architecture**: EfficientNet-B0 (5.3M parameters)
- **Training**: TTA (Test-Time Augmentation) for robust predictions
- **Inference Time**: 30-90 seconds (first request), 10-30s (subsequent)

---

## 📝 License

This project is for **research and educational purposes only**.  
Not approved for medical diagnosis or clinical decision-making.

---

## 👨‍💻 Author

Built as a full-stack AI medical assistant demo.

---

## 🙏 Acknowledgments

- **HAM10000 Dataset**: Harvard Dataverse
- **EfficientNet**: Google Research
- **TinyLlama**: TinyLlama Authors
- **HuggingFace**: Model hosting & Gradio framework
- **Vercel**: Frontend deployment platform

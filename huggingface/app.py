"""
DermAssist — HuggingFace Spaces Gradio Backend
================================================
Full inference pipeline:
  Image → EfficientNet-B0 (TTA) → FAISS RAG retrieval → TinyLlama QLoRA → Clinical report

Files expected in models/ directory:
  - best_model.pth          (~16 MB)  EfficientNet-B0 classifier weights
  - faiss_index.bin          (~1 MB)  FAISS index for knowledge retrieval
  - knowledge_base.json      (~1 MB)  JSON list of knowledge chunks
  - lora_adapter/            (~50 MB) QLoRA adapter for TinyLlama
    ├── adapter_config.json
    ├── adapter_model.safetensors (or .bin)
    └── ...

Environment:
  Runs on HuggingFace Spaces (CPU, 16 GB RAM free tier).
  For faster inference, upgrade to T4 GPU in Space settings.
"""

import os
import json
import logging
import traceback
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms, models
import gradio as gr

# ── Logging ──
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("dermassist")

# ── Lazy-import heavy libraries (saves startup memory) ──
faiss = None
SentenceTransformer = None
AutoTokenizer = None
AutoModelForCausalLM = None
PeftModel = None

def _lazy_import_rag():
    global faiss, SentenceTransformer
    if faiss is None:
        import faiss as _faiss
        faiss = _faiss
    if SentenceTransformer is None:
        from sentence_transformers import SentenceTransformer as _ST
        SentenceTransformer = _ST

def _lazy_import_gen():
    global AutoTokenizer, AutoModelForCausalLM, PeftModel
    if AutoTokenizer is None:
        from transformers import AutoTokenizer as _AT, AutoModelForCausalLM as _AM
        AutoTokenizer = _AT
        AutoModelForCausalLM = _AM
    if PeftModel is None:
        from peft import PeftModel as _PM
        PeftModel = _PM

# ── Paths ──
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
CLASSIFIER_PATH = os.path.join(MODEL_DIR, "best_model.pth")
FAISS_INDEX_PATH = os.path.join(MODEL_DIR, "faiss_index.bin")
KNOWLEDGE_PATH = os.path.join(MODEL_DIR, "knowledge_base.json")
LORA_ADAPTER_PATH = os.path.join(MODEL_DIR, "lora_adapter")

# Log what files actually exist so we can debug
logger.info(f"MODEL_DIR: {MODEL_DIR}")
logger.info(f"MODEL_DIR exists: {os.path.exists(MODEL_DIR)}")
if os.path.exists(MODEL_DIR):
    logger.info(f"MODEL_DIR contents: {os.listdir(MODEL_DIR)}")

# ── Class labels (HAM10000, 7 classes) ──
CLASS_NAMES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
CLASS_FULL_NAMES = {
    "akiec": "Actinic Keratosis (AKIEC)",
    "bcc":   "Basal Cell Carcinoma (BCC)",
    "bkl":   "Benign Keratosis (BKL)",
    "df":    "Dermatofibroma (DF)",
    "mel":   "Melanoma (MEL)",
    "nv":    "Melanocytic Nevi (NV)",
    "vasc":  "Vascular Lesions (VASC)",
}

# ── Medical fact corrections (applied post-generation) ──
MEDICAL_CORRECTIONS = {
    "melanoma is always benign": "melanoma is a malignant condition",
    "basal cell carcinoma is benign": "basal cell carcinoma is a malignant neoplasm",
    "actinic keratosis is benign": "actinic keratosis is a pre-cancerous condition",
    "no treatment is needed for melanoma": "melanoma requires urgent medical evaluation",
}

# ── Device ──
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {DEVICE}")


# ═══════════════════════════════════════════════════════════════
# PHASE 1 — EfficientNet-B0 Classifier
# ═══════════════════════════════════════════════════════════════

def load_classifier():
    """Load the EfficientNet-B0 classifier with custom head."""
    model = models.efficientnet_b0(weights=None)
    # Replace classifier head to match 7 classes
    in_features = model.classifier[1].in_features
    model.classifier = torch.nn.Sequential(
        torch.nn.Dropout(p=0.3),
        torch.nn.Linear(in_features, len(CLASS_NAMES)),
    )
    if os.path.exists(CLASSIFIER_PATH):
        state_dict = torch.load(CLASSIFIER_PATH, map_location=DEVICE, weights_only=True)
        model.load_state_dict(state_dict)
        logger.info("Classifier loaded from best_model.pth")
    else:
        logger.warning(f"Classifier weights not found at {CLASSIFIER_PATH} — using random weights")
    model.to(DEVICE)
    model.eval()
    return model


# ── Image preprocessing (ImageNet normalization, 224×224) ──
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

base_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

# ── TTA transforms (4 variants) ──
tta_transforms = [
    base_transform,  # Original
    transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=1.0),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]),
    transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomVerticalFlip(p=1.0),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]),
    transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Lambda(lambda img: img.rotate(90)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]),
]


def classify_with_tta(model, image: Image.Image):
    """
    Classify using Test-Time Augmentation.
    Averages softmax probabilities over 4 augmented views.
    Returns (predicted_class_name, confidence, top_k_list).
    """
    all_probs = []
    with torch.no_grad():
        for t in tta_transforms:
            tensor = t(image).unsqueeze(0).to(DEVICE)
            logits = model(tensor)
            probs = F.softmax(logits, dim=1).cpu().numpy()[0]
            all_probs.append(probs)

    # Average over TTA views
    avg_probs = np.mean(all_probs, axis=0)
    top_indices = np.argsort(avg_probs)[::-1][:3]

    predicted_idx = top_indices[0]
    predicted_class = CLASS_NAMES[predicted_idx]
    confidence = float(avg_probs[predicted_idx])

    top_k = [
        {"class": CLASS_FULL_NAMES[CLASS_NAMES[i]], "prob": float(avg_probs[i])}
        for i in top_indices
    ]

    return CLASS_FULL_NAMES[predicted_class], confidence, top_k


# ═══════════════════════════════════════════════════════════════
# PHASE 2 — FAISS RAG Retrieval
# ═══════════════════════════════════════════════════════════════

def load_rag():
    """Load FAISS index, knowledge chunks, and sentence-transformer."""
    _lazy_import_rag()
    knowledge = []
    index = None
    embedder = None

    if os.path.exists(KNOWLEDGE_PATH):
        with open(KNOWLEDGE_PATH, "r") as f:
            knowledge = json.load(f)
        logger.info(f"Loaded {len(knowledge)} knowledge chunks")
    else:
        logger.warning(f"Knowledge base not found at {KNOWLEDGE_PATH}")

    if os.path.exists(FAISS_INDEX_PATH):
        index = faiss.read_index(FAISS_INDEX_PATH)
        logger.info(f"FAISS index loaded ({index.ntotal} vectors)")
    else:
        logger.warning(f"FAISS index not found at {FAISS_INDEX_PATH}")

    try:
        embedder = SentenceTransformer("all-MiniLM-L6-v2")
        logger.info("Sentence-transformer loaded")
    except Exception as e:
        logger.warning(f"Could not load sentence-transformer: {e}")

    return index, knowledge, embedder


def retrieve_context(query: str, index, knowledge, embedder, top_k=3):
    """Retrieve the top-K relevant knowledge chunks for the predicted class."""
    if not index or not embedder or not knowledge:
        return "No knowledge base available."

    query_vec = embedder.encode([query]).astype("float32")
    distances, indices = index.search(query_vec, top_k)

    chunks = []
    for i in indices[0]:
        if 0 <= i < len(knowledge):
            chunk = knowledge[i]
            # Handle both string and dict formats
            text = chunk if isinstance(chunk, str) else chunk.get("text", str(chunk))
            chunks.append(text)

    return "\n\n".join(chunks) if chunks else "No relevant context found."


# ═══════════════════════════════════════════════════════════════
# PHASE 3 — TinyLlama QLoRA Report Generation
# ═══════════════════════════════════════════════════════════════

def load_generator():
    """Load TinyLlama base model + QLoRA adapter. Uses float16 to save memory."""
    _lazy_import_gen()
    tokenizer = None
    model = None
    base_model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

    try:
        logger.info("Loading TinyLlama tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        logger.info("Loading TinyLlama model (float16)...")
        model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        )
        logger.info("Base model loaded")

        if os.path.exists(LORA_ADAPTER_PATH):
            logger.info(f"Loading LoRA adapter from {LORA_ADAPTER_PATH}...")
            logger.info(f"LoRA adapter dir contents: {os.listdir(LORA_ADAPTER_PATH)}")
            model = PeftModel.from_pretrained(model, LORA_ADAPTER_PATH)
            logger.info("QLoRA adapter loaded")
        else:
            logger.warning(f"LoRA adapter not found at {LORA_ADAPTER_PATH} — using base model")

        model = model.to(DEVICE)
        model.eval()
        logger.info("TinyLlama generator ready")
    except Exception as e:
        logger.error(f"Failed to load generator: {e}")
        logger.error(traceback.format_exc())
        tokenizer = None
        model = None

    return tokenizer, model


def generate_report(
    predicted_class: str,
    confidence: float,
    top_k: list,
    context: str,
    tokenizer,
    model,
):
    """Generate a clinical report using TinyLlama chat template."""
    if not tokenizer or not model:
        return "Report generation unavailable — model not loaded."

    # Build the prompt using TinyLlama chat template
    system_prompt = (
        "You are a medical dermatology AI assistant. Generate a structured clinical "
        "report based on the classification result and medical context provided. "
        "Include: 1) Classification Summary, 2) Clinical Description, "
        "3) Risk Assessment, 4) Recommended Actions. "
        "Be professional, accurate, and note this is for educational purposes only."
    )

    top_k_str = ", ".join(
        [f"{p['class']} ({p['prob']*100:.1f}%)" for p in top_k]
    )

    user_prompt = (
        f"Classification: {predicted_class} (confidence: {confidence*100:.1f}%)\n"
        f"Top predictions: {top_k_str}\n\n"
        f"Medical context:\n{context}\n\n"
        f"Generate a structured clinical report for this skin lesion analysis."
    )

    prompt = (
        f"<|system|>\n{system_prompt}</s>\n"
        f"<|user|>\n{user_prompt}</s>\n"
        f"<|assistant|>\n"
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.2,
            top_p=0.85,
            top_k=40,
            repetition_penalty=1.2,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Decode only the generated tokens (skip prompt)
    generated = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    ).strip()

    # Apply medical fact corrections
    for wrong, correct in MEDICAL_CORRECTIONS.items():
        generated = generated.replace(wrong, correct)

    return generated


# ═══════════════════════════════════════════════════════════════
# Load models at startup — classifier and RAG eagerly,
# generator lazily (on first request) to avoid startup timeout
# ═══════════════════════════════════════════════════════════════

logger.info("=" * 50)
logger.info("Loading Phase 1: Classifier...")
try:
    classifier = load_classifier()
    logger.info("Classifier ready")
except Exception as e:
    logger.error(f"Classifier failed: {e}")
    classifier = None

logger.info("Loading Phase 2: RAG...")
try:
    faiss_index, knowledge_base, sentence_embedder = load_rag()
    logger.info("RAG ready")
except Exception as e:
    logger.error(f"RAG failed: {e}")
    faiss_index, knowledge_base, sentence_embedder = None, [], None

# Generator loaded lazily on first request
gen_tokenizer = None
gen_model = None
_generator_loaded = False

def _ensure_generator():
    global gen_tokenizer, gen_model, _generator_loaded
    if not _generator_loaded:
        logger.info("Loading Phase 3: Generator (first request)...")
        gen_tokenizer, gen_model = load_generator()
        _generator_loaded = True

logger.info("Startup complete — Gradio app ready")
logger.info("=" * 50)


# ═══════════════════════════════════════════════════════════════
# Gradio interface
# ═══════════════════════════════════════════════════════════════

def analyze(image: Image.Image):
    """
    Full end-to-end pipeline:
    1. Classify with TTA
    2. Retrieve relevant medical context
    3. Generate clinical report
    Returns a dict matching the frontend's expected format.
    """
    if image is None:
        return {"error": "No image provided"}

    if classifier is None:
        return {"error": "Classifier model not loaded — check Space logs"}

    try:
        # Ensure RGB
        image = image.convert("RGB")

        # Step 1: Classify
        predicted_class, confidence, top_k = classify_with_tta(classifier, image)
        logger.info(f"Classification: {predicted_class} ({confidence:.3f})")

        # Step 2: Retrieve context
        query = f"{predicted_class} skin lesion dermoscopy"
        context = retrieve_context(query, faiss_index, knowledge_base, sentence_embedder)

        # Step 3: Generate report (lazy-loads generator on first call)
        _ensure_generator()
        report = generate_report(
            predicted_class, confidence, top_k, context,
            gen_tokenizer, gen_model,
        )

        return {
            "predicted_class": predicted_class,
            "confidence": round(confidence, 4),
            "top_k": top_k,
            "report": report,
            "retrieved_context": context,
        }
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        logger.error(traceback.format_exc())
        return {"error": f"Analysis failed: {str(e)}"}


# ── Gradio app ──
demo = gr.Interface(
    fn=analyze,
    inputs=gr.Image(type="pil", label="Dermoscopic Image"),
    outputs=gr.JSON(label="Analysis Result"),
    title="DermAssist — AI Skin Lesion Analysis",
    description="Upload a dermoscopic image for classification and clinical report generation.",
    flagging_mode="never",
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)

"""
DermWise — HuggingFace Spaces Gradio Backend
================================================
Full inference pipeline:
  Image → EfficientNet-B0 (TTA) → FAISS RAG retrieval → HF Inference API → Clinical report

Report generation — design note:
  TinyLlama-1.1B was fine-tuned with QLoRA for this task (adapter shipped in
  models/lora_adapter/ and documented in /training). However, the LIVE pipeline
  generates the clinical report via the HuggingFace Serverless Inference API using
  Qwen/Qwen2.5-7B-Instruct (see generate_report()). This was a deliberate
  quality/latency trade-off: a 7B instruct model produces materially better
  structured medical reports than a 1.1B model running on the free CPU tier, at no
  hosting cost. The QLoRA adapter is therefore NOT loaded at runtime — it is kept
  as evidence of the fine-tuning work and as a path to fully local inference.

Files expected in models/ directory:
  - best_model.pth          (~16 MB)  EfficientNet-B0 classifier weights  [USED]
  - faiss_index.bin          (~1 MB)  FAISS index for knowledge retrieval  [USED]
  - knowledge_base.json      (~1 MB)  JSON list of knowledge chunks        [USED]
  - lora_adapter/            (~50 MB) QLoRA adapter for TinyLlama  [NOT loaded at runtime]

Environment:
  Runs on HuggingFace Spaces (CPU free tier).
  Report generation uses HF Serverless Inference API (free).
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
logger = logging.getLogger("dermwise")

# ── Lazy-import heavy libraries (saves startup memory) ──
faiss = None
SentenceTransformer = None

def _lazy_import_rag():
    global faiss, SentenceTransformer
    if faiss is None:
        import faiss as _faiss
        faiss = _faiss
    if SentenceTransformer is None:
        from sentence_transformers import SentenceTransformer as _ST
        SentenceTransformer = _ST

# ── Paths ──
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
CLASSIFIER_PATH = os.path.join(MODEL_DIR, "best_model.pth")
FAISS_INDEX_PATH = os.path.join(MODEL_DIR, "faiss_index.bin")
KNOWLEDGE_PATH = os.path.join(MODEL_DIR, "knowledge_base.json")

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
DEVICE = torch.device("cpu")
logger.info(f"Using device: {DEVICE}")


# ═══════════════════════════════════════════════════════════════
# PHASE 1 — EfficientNet-B0 Classifier
# ═══════════════════════════════════════════════════════════════

def load_classifier():
    """Load the EfficientNet-B0 classifier with custom head."""
    model = models.efficientnet_b0(weights=None)
    # Replace classifier head to match 7 classes.
    # Dropout p=0.2 matches the training notebook (it kept EfficientNet-B0's default
    # Dropout and only swapped the Linear), so this is identical to the evaluated model.
    # (Dropout is inactive at inference regardless, but we keep it consistent.)
    in_features = model.classifier[1].in_features
    model.classifier = torch.nn.Sequential(
        torch.nn.Dropout(p=0.2),
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
# Views match the training-notebook evaluation exactly (original, h-flip, v-flip,
# both-flip) so the deployed classifier == the model the reported metrics describe.
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
    transforms.Compose([  # both-flip (horizontal + vertical) == torch.flip(x, [2, 3])
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=1.0),
        transforms.RandomVerticalFlip(p=1.0),
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
    # The FAISS index is IndexFlatIP over L2-normalized vectors, so normalizing the
    # query makes the inner product a true cosine similarity (matches the notebook).
    faiss.normalize_L2(query_vec)
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
# PHASE 3 — Report Generation via HF Inference API (free)
# ═══════════════════════════════════════════════════════════════

def generate_report(predicted_class, confidence, top_k, context):
    """Generate a clinical report using HuggingFace free Inference API."""
    from huggingface_hub import InferenceClient

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

    try:
        # Use HF token from environment (auto-set in HF Spaces)
        hf_token = os.environ.get("HF_TOKEN", None)
        client = InferenceClient(token=hf_token)

        # Use chat_completion with a free serverless model
        response = client.chat_completion(
            model="Qwen/Qwen2.5-7B-Instruct",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=512,
            temperature=0.2,
            top_p=0.85,
        )

        report = response.choices[0].message.content.strip()

        # Apply medical fact corrections
        for wrong, correct in MEDICAL_CORRECTIONS.items():
            report = report.replace(wrong, correct)

        return report

    except Exception as e:
        logger.error(f"HF Inference API failed: {e}")
        logger.error(traceback.format_exc())
        # Fallback: generate a template-based report
        return _fallback_report(predicted_class, confidence, top_k, context)


def _fallback_report(predicted_class, confidence, top_k, context):
    """Template-based fallback report if the Inference API is unavailable."""
    severity_map = {
        "Melanoma (MEL)": "High (Malignant)",
        "Basal Cell Carcinoma (BCC)": "High (Malignant)",
        "Actinic Keratosis (AKIEC)": "Moderate (Pre-cancerous)",
        "Benign Keratosis (BKL)": "Low (Benign)",
        "Dermatofibroma (DF)": "Low (Benign)",
        "Melanocytic Nevi (NV)": "Low (Benign)",
        "Vascular Lesions (VASC)": "Low (Benign)",
    }
    severity = severity_map.get(predicted_class, "Unknown")
    top_k_str = "\n".join(
        [f"  - {p['class']}: {p['prob']*100:.1f}%" for p in top_k]
    )

    return (
        f"1) Classification Summary\n"
        f"The lesion has been classified as {predicted_class} "
        f"with a confidence of {confidence*100:.1f}%.\n\n"
        f"Top predictions:\n{top_k_str}\n\n"
        f"2) Clinical Description\n"
        f"Based on the dermoscopic image analysis using EfficientNet-B0 with "
        f"test-time augmentation, the primary classification indicates {predicted_class}.\n\n"
        f"3) Risk Assessment\n"
        f"Risk level: {severity}\n\n"
        f"4) Recommended Actions\n"
        f"This is an AI-generated analysis for educational purposes only. "
        f"Please consult a board-certified dermatologist for proper diagnosis "
        f"and treatment planning.\n\n"
        f"DISCLAIMER: This report is generated by an AI system for research "
        f"and educational purposes only. It should not be used as a substitute "
        f"for professional medical advice, diagnosis, or treatment."
    )


# ═══════════════════════════════════════════════════════════════
# PHASE 3 (alt) — Local fine-tuned TinyLlama-1.1B + QLoRA adapter
# ═══════════════════════════════════════════════════════════════
# This is the model the user fine-tuned with QLoRA. It runs ON THE CPU SPACE
# (no GPU, so we load the base model in fp32 and apply the LoRA adapter — the
# 4-bit quantization used in training is a GPU-only feature). It is slower and
# lower-quality than the hosted Qwen model, so it is opt-in via the UI toggle.
# Loaded lazily on first use so it never slows startup or the Qwen path.

LOCAL_BASE_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
LOCAL_ADAPTER_DIR = os.path.join(MODEL_DIR, "lora_adapter")
_local_llm = {"model": None, "tokenizer": None, "failed": False}


def _load_local_llm():
    """Lazily load TinyLlama base + the fine-tuned QLoRA adapter (CPU, fp32)."""
    if _local_llm["model"] is not None or _local_llm["failed"]:
        return _local_llm["model"], _local_llm["tokenizer"]
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel
        logger.info("Loading local TinyLlama + QLoRA adapter (first use)...")
        tokenizer = AutoTokenizer.from_pretrained(LOCAL_ADAPTER_DIR)
        base = AutoModelForCausalLM.from_pretrained(
            LOCAL_BASE_MODEL, torch_dtype=torch.float32, low_cpu_mem_usage=True
        )
        model = PeftModel.from_pretrained(base, LOCAL_ADAPTER_DIR)
        model.to(DEVICE).eval()
        _local_llm["model"], _local_llm["tokenizer"] = model, tokenizer
        logger.info("Local TinyLlama ready")
    except Exception as e:
        logger.error(f"Local TinyLlama load failed: {e}")
        logger.error(traceback.format_exc())
        _local_llm["failed"] = True
    return _local_llm["model"], _local_llm["tokenizer"]


def generate_report_local(predicted_class, confidence, top_k, context):
    """Generate a report with the fine-tuned TinyLlama (mirrors the training prompt)."""
    model, tokenizer = _load_local_llm()
    if model is None:
        # Loading failed → fall back to the hosted Qwen path so the demo still works.
        logger.warning("Local model unavailable — falling back to Qwen")
        return generate_report(predicted_class, confidence, top_k, context), "qwen-fallback"

    conf_pct = f"{confidence * 100:.1f}"
    prompt = (
        f"<|system|>\nYou are DermWise AI, a dermatology report generator.</s>\n"
        f"<|user|>\nGenerate a dermatology report for a skin lesion classified as "
        f"{predicted_class} with {conf_pct}% confidence.\n"
        f"Medical context: {context[:400]}</s>\n"
        f"<|assistant|>\n"
    )
    try:
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(DEVICE)
        with torch.no_grad():
            output_ids = model.generate(
                **inputs, max_new_tokens=512, temperature=0.2, do_sample=True,
                top_p=0.85, top_k=30, repetition_penalty=1.2,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
        report = tokenizer.decode(
            output_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        ).strip()

        # Trim hallucinated trailing sections the small model tends to add
        for marker in ["<|user|>", "<|system|>", "METHODS:", "BACKGROUND:",
                       "CONCLUSIONS:", "FUNDING", "TRIAL REGISTRATION"]:
            idx = report.find(marker)
            if idx > 0:
                report = report[:idx].strip()

        # Same safety fact-corrections as the Qwen path
        for wrong, correct in MEDICAL_CORRECTIONS.items():
            report = report.replace(wrong, correct)

        return (report or _fallback_report(predicted_class, confidence, top_k, context)), "tinyllama"
    except Exception as e:
        logger.error(f"Local TinyLlama generation failed: {e}")
        logger.error(traceback.format_exc())
        return generate_report(predicted_class, confidence, top_k, context), "qwen-fallback"


# ═══════════════════════════════════════════════════════════════
# Load models at startup — classifier and RAG
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

logger.info("Phase 3: Qwen via HF Inference API (default); local TinyLlama+QLoRA loads on demand")
logger.info("Startup complete — Gradio app ready")
logger.info("=" * 50)


# ═══════════════════════════════════════════════════════════════
# Gradio interface
# ═══════════════════════════════════════════════════════════════

def analyze(image: Image.Image, model_choice: str = "qwen"):
    """
    Full end-to-end pipeline:
    1. Classify with TTA
    2. Retrieve relevant medical context
    3. Generate clinical report — either the hosted Qwen-7B (default, fast) or the
       fine-tuned local TinyLlama-1.1B+QLoRA (opt-in via model_choice="tinyllama").
    Returns a dict matching the frontend's expected format.
    """
    if image is None:
        return {"error": "No image provided"}

    if classifier is None:
        return {"error": "Classifier model not loaded — check Space logs"}

    # Normalize the toggle value (Gradio may pass labels/None)
    choice = "tinyllama" if str(model_choice).strip().lower().startswith("tiny") else "qwen"

    try:
        # Ensure RGB
        image = image.convert("RGB")
        logger.info("Image converted to RGB")

        # Step 1: Classify
        logger.info("Starting classification...")
        predicted_class, confidence, top_k = classify_with_tta(classifier, image)
        logger.info(f"Classification: {predicted_class} ({confidence:.3f})")

        # Step 2: Retrieve context
        logger.info("Retrieving context...")
        query = f"{predicted_class} skin lesion dermoscopy"
        context = retrieve_context(query, faiss_index, knowledge_base, sentence_embedder)
        logger.info("Context retrieved")

        # Step 3: Generate report with the chosen model
        logger.info(f"Generating report (model={choice})...")
        if choice == "tinyllama":
            report, model_used = generate_report_local(predicted_class, confidence, top_k, context)
        else:
            report = generate_report(predicted_class, confidence, top_k, context)
            model_used = "qwen"
        logger.info(f"Report generated (model_used={model_used})")

        return {
            "predicted_class": predicted_class,
            "confidence": round(confidence, 4),
            "top_k": top_k,
            "report": report,
            "retrieved_context": context,
            "model_used": model_used,
        }
    except Exception as e:
        error_msg = f"Analysis failed: {str(e)}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        # Return error as valid JSON so Gradio doesn't swallow it
        return {"error": error_msg, "traceback": traceback.format_exc()}


# ── Gradio app ──
demo = gr.Interface(
    fn=analyze,
    inputs=[
        gr.Image(type="pil", label="Dermoscopic Image"),
        gr.Radio(
            choices=["qwen", "tinyllama"],
            value="qwen",
            label="Report model (qwen = hosted 7B, fast; tinyllama = fine-tuned 1.1B, local/slow)",
        ),
    ],
    outputs=gr.JSON(label="Analysis Result"),
    title="DermWise — AI Skin Lesion Analysis",
    description="Upload a dermoscopic image for classification and clinical report generation.",
    flagging_mode="never",
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)

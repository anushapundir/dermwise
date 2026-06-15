# Training & Fine-Tuning

This folder contains the end-to-end notebook that produced every model artifact shipped in
[`../huggingface/models/`](../huggingface/models/). It is the source of truth for how DermWise was
built; the metrics it reports are documented in [`../MODEL_CARD.md`](../MODEL_CARD.md).

## Contents

- **`dermwise_pipeline.ipynb`** — the full Kaggle notebook (run on GPU), in 4 phases:

  | Phase | What it does | Produces |
  |---|---|---|
  | 1. Classifier | EfficientNet-B0, lesion-level stratified split, 2-stage transfer learning, weighted sampler, label smoothing, early stopping on macro-F1 | `best_model.pth`, `test_split.csv`, training curves |
  | 2. RAG | Builds the 53-chunk dermatology knowledge base, embeds with `all-MiniLM-L6-v2`, builds a cosine-similarity FAISS index | `faiss_index.bin`, `knowledge_base.json` |
  | 3. QLoRA | 4-bit NF4 fine-tune of TinyLlama-1.1B on synthetic templated reports (r=16, α=32, 2.0% trainable params) | `lora_adapter/` |
  | 4. Demo + export | End-to-end image→classify→retrieve→report demo; packages all artifacts | deployment zip |

## Key results (held-out test set, 1,497 images)

- **Accuracy 81.9% · Macro-F1 0.703 · Melanoma recall 73.2%**
- Full per-class breakdown and confusion matrices: [`../MODEL_CARD.md`](../MODEL_CARD.md).

## How the artifacts flow into the app

```
training/dermwise_pipeline.ipynb  ──exports──▶  huggingface/models/
        (Kaggle GPU run)                         ├── best_model.pth      → app.py classifier
                                                 ├── faiss_index.bin     → app.py RAG
                                                 ├── knowledge_base.json → app.py RAG
                                                 └── lora_adapter/       → (fine-tuning evidence;
                                                                            app.py serves Qwen instead)
```

## Reproducing the evaluation offline

You do not need to re-run the notebook to verify the classifier metrics. With `best_model.pth`, the
saved `test_split.csv`, and the HAM10000 images, run
[`../huggingface/eval.py`](../huggingface/eval.py) — see [`../MODEL_CARD.md`](../MODEL_CARD.md) §5.

## Notes

- The notebook targets the Kaggle environment (paths under `/kaggle/input`, GPU + bitsandbytes for
  4-bit QLoRA). It is provided as a reproducible record, not as a script to run on the free CPU Space.
- A few deliberate differences exist between the notebook and the deployed `app.py` (served LLM, TTA
  view, retrieval details) — all listed in [`../MODEL_CARD.md`](../MODEL_CARD.md) §4.

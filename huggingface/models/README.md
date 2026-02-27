# Models Directory

Place your Kaggle-exported model artifacts here:

```
models/
├── best_model.pth              # EfficientNet-B0 classifier (~16 MB)
├── faiss_index.bin             # FAISS index for RAG (~1 MB)
├── knowledge_base.json         # Knowledge chunks for RAG (~1 MB)
└── lora_adapter/               # TinyLlama QLoRA adapter (~50 MB)
    ├── adapter_config.json
    ├── adapter_model.safetensors
    └── ...
```

## How to export from Kaggle

In your Kaggle notebook, run:

```python
# The artifacts should already be saved at:
# /kaggle/working/artifacts/classifier/best_model.pth
# /kaggle/working/artifacts/rag/faiss_index.bin
# /kaggle/working/artifacts/rag/knowledge_base.json
# /kaggle/working/artifacts/tinyllama/lora_adapter/

# Download them from Kaggle's output tab, then place here.
```

## For HuggingFace Spaces deployment

Upload these files directly into your HF Space repo under `models/`.
Large files (>10 MB) will automatically use Git LFS.

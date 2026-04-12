FROM python:3.10-slim

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    git git-lfs build-essential && \
    rm -rf /var/lib/apt/lists/* && \
    git lfs install

# Install CPU-only PyTorch first (saves ~2GB vs CUDA version)
RUN pip install --no-cache-dir \
    torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install remaining deps (including gradio)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt gradio

# Copy app code and models
COPY . .

# Create user for HF Spaces
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user PATH="/home/user/.local/bin:$PATH"

EXPOSE 7860

CMD ["python", "app.py"]

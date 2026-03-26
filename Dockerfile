# Use Python 3.11 slim for smaller image (CPU-only; no CUDA)
FROM python:3.11-slim

# Avoid duplicate OpenMP libs (matches your KMP_DUPLICATE_LIB_OK)
ENV KMP_DUPLICATE_LIB_OK=TRUE

# HF Spaces runs as UID 1000
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user PATH=/home/user/.local/bin:$PATH
WORKDIR /home/user/app

# Install dependencies first (better layer caching)
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY --chown=user . .

# Model: Option A – download from HuggingFace Hub at build time
# (Upload your checkpoint to a HF repo first, e.g. your-username/galaxy-ring-checkpoint)
# ARG HF_CHECKPOINT_REPO
# RUN python -c "from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='${HF_CHECKPOINT_REPO}', filename='stage2-best-epoch=20.ckpt', local_dir='model/version_37/checkpoints')"

# Model: Option B – copy local checkpoint (if you add it to the repo or build context)
COPY --chown=user model/ model/

# Ensure dirs exist (server creates them, but explicit is fine)
RUN mkdir -p images static templates

# HF Spaces expects port 7860
EXPOSE 7860

# Start FastAPI
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "2"]
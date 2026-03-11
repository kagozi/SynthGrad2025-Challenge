# =============================================================================
# SynthRAD2025 Task 1 — Grand Challenge Docker Submission
#
# Target: AWS g4dn.2xlarge (1x NVIDIA T4 16GB, 8 vCPU, 32GB RAM)
# Constraint: 15 min inference per patient
#
# Build:
#   docker build -t synthrad2025_task1 -f docker/Dockerfile .
#
# Test locally:
#   docker run --gpus all \
#     -v /path/to/input:/input \
#     -v /path/to/output:/output \
#     synthrad2025_task1
# =============================================================================

FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

LABEL maintainer="your-email@example.com"
LABEL description="SynthRAD2025 Task 1: MR → Synthetic CT"

# ── System dependencies ────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# ── Python dependencies ────────────────────────────────────────────────────────
WORKDIR /opt/app

COPY requirements.txt .
RUN pip install --no-cache-dir \
    SimpleITK>=2.3.0 \
    numpy>=1.24.0 \
    pandas>=2.0.0 \
    pyyaml>=6.0 \
    tqdm>=4.65.0

# ── Copy project source ────────────────────────────────────────────────────────
COPY src/ /opt/app/src/
COPY docker/process.py /opt/app/process.py

# ── Copy trained model checkpoint ─────────────────────────────────────────────
# Replace with your actual checkpoint path before building
COPY checkpoints/best_model.pth /opt/app/model.pth

# ── Grand Challenge I/O convention ─────────────────────────────────────────────
# Input:  /input/<case_id>/mr.mha  (and mask.mha if available)
# Output: /output/<case_id>/ct.mha (predicted sCT)

ENV INPUT_DIR="/input"
ENV OUTPUT_DIR="/output"
ENV MODEL_PATH="/opt/app/model.pth"

# ── Entrypoint ─────────────────────────────────────────────────────────────────
ENTRYPOINT ["python", "/opt/app/process.py"]

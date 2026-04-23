# =============================================================================
# SynthRAD2025 Task 1 — Grand Challenge Docker Submission
#
# Follows the official algorithm-template structure:
#   https://github.com/SynthRAD2025/algorithm-template
#
# Target: AWS g4dn.2xlarge (NVIDIA T4 16GB, 8 vCPU, 32GB RAM)
# I/O:
#   Input MR   : /input/images/mri/<uuid>.mha
#   Input mask : /input/images/body/<uuid>.mha
#   Region     : /input/region.json
#   Output sCT : /output/images/synthetic-ct/<uuid>.mha
#
# Build (from repo root):
#   docker build -t synthrad_algorithm -f docker/Dockerfile .
#
# Export:
#   docker/export.sh
# =============================================================================

FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

ENV TASK_TYPE=mri
ENV EXECUTE_IN_DOCKER=1

# ── Non-root user (required by Grand Challenge) ────────────────────────────
RUN groupadd -r algorithm && useradd -m --no-log-init -r -g algorithm algorithm

RUN mkdir -p /opt/app /input /output \
    && chown algorithm:algorithm /opt/app /input /output

USER algorithm

WORKDIR /opt/app

ENV PATH="/home/algorithm/.local/bin:${PATH}"

RUN python -m pip install --user -U pip

# ── Python dependencies ────────────────────────────────────────────────────
COPY --chown=algorithm:algorithm docker/requirements.txt /opt/app/
RUN python -m pip install --user -r requirements.txt

# ── Template base class + entrypoint ──────────────────────────────────────
COPY --chown=algorithm:algorithm docker/.env           /opt/app/
COPY --chown=algorithm:algorithm docker/base_algorithm.py /opt/app/
COPY --chown=algorithm:algorithm docker/process.py     /opt/app/

# ── Project source ─────────────────────────────────────────────────────────
COPY --chown=algorithm:algorithm src/ /opt/app/src/

# ── Model weights (baked in at build time) ─────────────────────────────────
# Place fold checkpoints under docker/weights/ before building:
#   docker/weights/fold0_best.pth
#   docker/weights/fold1_best.pth
#   ...
# They are copied to /opt/ml/model/ inside the container.
RUN mkdir -p /opt/ml/model
COPY --chown=algorithm:algorithm docker/weights/ /opt/ml/model/

ENTRYPOINT python -m process $0 $@

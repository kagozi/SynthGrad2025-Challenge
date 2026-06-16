#!/usr/bin/env bash
# =============================================================================
# Submit per-anatomy DynUNet fine-tuning jobs for all folds.
#
# Each job fine-tunes from the matching multi-anatomy fold checkpoint:
#   fold N  →  --finetune_from /pvc/checkpoints/dynunet/foldN_best.pth
#
# Usage:
#   # Submit all 15 jobs (3 anatomies × 5 folds):
#   bash nautilius/submit_per_anatomy.sh
#
#   # Submit one anatomy only:
#   bash nautilius/submit_per_anatomy.sh HN
#
#   # Dry-run — print YAML without applying:
#   DRY_RUN=1 bash nautilius/submit_per_anatomy.sh
# =============================================================================
set -euo pipefail

ANATOMIES=("HN" "TH" "AB")
FOLDS=(0 1 2 3 4)
DRY_RUN="${DRY_RUN:-0}"

# If an anatomy filter is passed as arg, restrict to that one
if [[ $# -ge 1 ]]; then
  ANATOMIES=("$1")
fi

for ANAT in "${ANATOMIES[@]}"; do
  for FOLD in "${FOLDS[@]}"; do

    ANAT_LOWER=$(echo "${ANAT}" | tr '[:upper:]' '[:lower:]')
    JOB_NAME="synthrad2025-dynunet-${ANAT_LOWER}-fold${FOLD}"

    YAML=$(cat <<EOF
apiVersion: batch/v1
kind: Job
metadata:
  name: ${JOB_NAME}
spec:
  backoffLimit: 0
  ttlSecondsAfterFinished: 86400
  template:
    spec:
      restartPolicy: Never
      affinity:
        nodeAffinity:
          requiredDuringSchedulingIgnoredDuringExecution:
            nodeSelectorTerms:
            - matchExpressions:
              - key: kubernetes.io/hostname
                operator: NotIn
                values:
                - k8s-chase-ci-10.calit2.optiputer.net
                - k8s-haosu-02.sdsc.optiputer.net
                - k8s-haosu-03.sdsc.optiputer.net
                - nautilus-it-gpu03.fullerton.edu
              - key: nvidia.com/gpu.compute.major
                operator: Gt
                values:
                - "7"
      containers:
        - name: trainer
          image: ghcr.io/kagozi/synthrad2025-train:latest
          imagePullPolicy: Always
          command: ["python", "training/train_dynunet.py"]
          args:
            - "--config"
            - "training/configs/dynunet_${ANAT}.yaml"
            - "--fold"
            - "${FOLD}"
            - "--finetune_from"
            - "/pvc/checkpoints/dynunet/fold${FOLD}_best.pth"
          env:
            - name: WANDB_API_KEY
              valueFrom:
                secretKeyRef:
                  name: wandb-secret
                  key: api-key
            - name: WANDB_PROJECT
              value: "synthrad2025-task1"
            - name: PYTORCH_CUDA_ALLOC_CONF
              value: "expandable_segments:True"
          resources:
            requests:
              memory: "20Gi"
              cpu: "4"
              nvidia.com/gpu: "1"
            limits:
              memory: "24Gi"
              cpu: "4"
              nvidia.com/gpu: "1"
          volumeMounts:
            - name: synthrad2025-vol
              mountPath: /pvc
            - name: dshm
              mountPath: /dev/shm
      volumes:
        - name: synthrad2025-vol
          persistentVolumeClaim:
            claimName: synthrad2025-pvc
        - name: dshm
          emptyDir:
            medium: Memory
            sizeLimit: 16Gi
      tolerations:
        - key: "nvidia.com/gpu"
          operator: "Exists"
          effect: "NoSchedule"
EOF
)

    if [[ "$DRY_RUN" == "1" ]]; then
      echo "--- ${JOB_NAME} ---"
      echo "$YAML"
      echo ""
    else
      echo "Submitting ${JOB_NAME} ..."
      echo "$YAML" | kubectl apply -f -
    fi

  done
done

echo "Done. ${#ANATOMIES[@]} anatomy/anatomies × ${#FOLDS[@]} folds submitted."

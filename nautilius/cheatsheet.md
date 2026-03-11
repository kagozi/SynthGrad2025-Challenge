# Nautilus NRP — kubectl Cheatsheet for SynthRAD2025

## One-time setup

```bash
# 1. Create PVC (200Gi shared storage)
kubectl apply -f nautilius/pvc-synthgrad.yml
kubectl get pvc synthrad2025-pvc -w

# 2. Create WandB secret (from .env key)
kubectl create secret generic wandb-secret \
    --from-literal=api-key=$(grep WANDB_API_KEY .env | cut -d= -f2)

# 3. Verify secret
kubectl get secret wandb-secret
```

## Upload data to PVC

```bash
# Start uploader pod
kubectl apply -f nautilius/pvc-uploader-pod.yaml
kubectl get pod synthrad2025-uploader -w   # wait for Running

# Upload zip files (run from project root)
kubectl cp ./data/synthRAD2025_Task1_Train.zip   synthrad2025-uploader:/pvc/data/
kubectl cp ./data/synthRAD2025_Task1_Train_D.zip synthrad2025-uploader:/pvc/data/
kubectl cp ./data/synthRAD2025_Task1_Val_Input.zip synthrad2025-uploader:/pvc/data/

# Shell into uploader to extract
kubectl exec -it synthrad2025-uploader -- bash
# Inside pod:
cd /pvc/data
apt-get update && apt-get install -y unzip
unzip -q synthRAD2025_Task1_Train.zip
unzip -q synthRAD2025_Task1_Train_D.zip
unzip -q synthRAD2025_Task1_Val_Input.zip
ls -la
exit

# Delete uploader pod when done
kubectl delete pod synthrad2025-uploader
```

## Build & push Docker image

```bash
# Option A: Manual (local Docker)
docker build -t ghcr.io/kagozi/synthrad2025-train:latest \
    -f docker/Dockerfile.train .
docker push ghcr.io/kagozi/synthrad2025-train:latest

# Option B: GitHub Actions (auto on push to main)
git push origin main   # triggers .github/workflows/docker-build.yml
```

## Prepare CV folds (one-time)

```bash
kubectl apply -f nautilius/jobs/prepare-folds.yaml
kubectl logs -f job/synthrad2025-prepare-folds
```

## Run training jobs

```bash
# Single fold
kubectl apply -f nautilius/jobs/train-fold0.yaml
kubectl logs -f job/synthrad2025-train-fold0

# All 5 folds in parallel
for i in 0 1 2 3 4; do
    kubectl apply -f nautilius/jobs/train-fold${i}.yaml
done

# Monitor all jobs
kubectl get jobs -w
```

## Monitor & debug

```bash
# Check job status
kubectl describe job synthrad2025-train-fold0

# Follow logs
kubectl logs -f job/synthrad2025-train-fold0

# Shell into a running job pod
kubectl exec -it $(kubectl get pods -l job-name=synthrad2025-train-fold0 \
    -o jsonpath='{.items[0].metadata.name}') -- bash

# GPU check inside pod
nvidia-smi

# Check PVC contents
kubectl exec synthrad2025-uploader -- ls -lah /pvc/checkpoints/
kubectl exec synthrad2025-uploader -- ls -lah /pvc/data/splits/
```

## Cleanup

```bash
# Delete completed jobs
kubectl delete job synthrad2025-train-fold0

# Delete all synthrad jobs
kubectl delete jobs -l app=synthrad2025

# Delete PVC (WARNING: destroys all data)
kubectl delete pvc synthrad2025-pvc
```

## Copy results from PVC to local

```bash
# Restart uploader pod first
kubectl apply -f nautilius/pvc-uploader-pod.yaml
kubectl get pod synthrad2025-uploader -w

# Copy checkpoint
kubectl cp synthrad2025-uploader:/pvc/checkpoints/baseline_unet2d/fold0_best.pth \
    ./checkpoints/fold0_best.pth

# Copy splits CSV
kubectl cp synthrad2025-uploader:/pvc/data/splits/folds.csv ./data/splits/folds.csv
```

## PVC directory layout (on pod)

```
/pvc/
├── data/
│   ├── synthRAD2025_Task1_Train/
│   │   └── Task1/
│   │       ├── AB/  1ABA001/ → mr.mha, ct.mha, mask.mha
│   │       ├── HN/
│   │       └── TH/
│   ├── synthRAD2025_Task1_Train_D/
│   │   └── Task1/
│   │       ├── AB/ 1ABD001/ ...
│   │       ├── HN/
│   │       └── TH/
│   ├── synthRAD2025_Task1_Val_Input/
│   │   └── Task1/ {AB,HN,TH}/ (mr.mha + mask.mha only)
│   └── splits/
│       └── folds.csv
└── checkpoints/
    └── baseline_unet2d/
        ├── fold0_best.pth
        ├── fold0_last.pth
        └── ...
```

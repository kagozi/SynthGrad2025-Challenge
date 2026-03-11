#!/usr/bin/env python3
"""
SynthRAD2025 Task 1 — Grand Challenge process.py

This is the Docker entrypoint called by the Grand Challenge evaluation platform.

Grand Challenge I/O convention:
  Input:  /input/<case_id>/mr.mha  (+ mask.mha)
  Output: /output/<case_id>/ct.mha

The script:
  1. Scans /input for all case directories
  2. Determines anatomy from case_id naming (1HN/1TH/1AB)
  3. Runs inference slice-by-slice (fits in 16GB VRAM)
  4. Saves predicted sCT to /output preserving metadata

Inference time budget: 15 min / patient → well within budget for 2D U-Net.
"""

import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import torch
from torch.utils.data import DataLoader

# ── Paths (set by Dockerfile ENV) ─────────────────────────────────────────────
INPUT_DIR  = Path(os.environ.get("INPUT_DIR",  "/input"))
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", "/output"))
MODEL_PATH = Path(os.environ.get("MODEL_PATH", "/opt/app/model.pth"))

sys.path.insert(0, "/opt/app")

from src.dataset import SynthRADInferenceDataset, denormalise_ct, ANATOMY_TO_IDX
from src.models.unet2d import UNet2D, AttentionUNet2D


# ── Anatomy detection ──────────────────────────────────────────────────────────

def detect_anatomy(case_id: str) -> str:
    """Infer anatomy from case ID naming convention."""
    m = re.match(r"\d(HN|TH|AB)", case_id, re.I)
    if m:
        return m.group(1).upper()
    # Fallback: inspect folder name components
    for anat in ["HN", "TH", "AB"]:
        if anat.lower() in case_id.lower():
            return anat
    raise ValueError(f"Cannot determine anatomy for case: {case_id}")


# ── Model loading ──────────────────────────────────────────────────────────────

def load_model(device: torch.device):
    ckpt = torch.load(str(MODEL_PATH), map_location=device)
    cfg  = ckpt["config"]["model"]

    kwargs = dict(
        in_channels=cfg["in_channels"],
        out_channels=cfg["out_channels"],
        base_features=cfg["base_features"],
        depth=cfg["depth"],
        n_anatomy=cfg["n_anatomy"],
        use_anatomy=cfg["use_anatomy"],
    )
    model = AttentionUNet2D(**kwargs) if cfg["name"] == "attention_unet2d" \
            else UNet2D(**kwargs)

    model.load_state_dict(ckpt["model"])
    model.eval()
    model.to(device)
    return model


# ── Inference ──────────────────────────────────────────────────────────────────

@torch.no_grad()
def predict_volume(model, case_path: Path, anatomy: str, device: torch.device) -> np.ndarray:
    ds     = SynthRADInferenceDataset(case_path, anatomy)
    loader = DataLoader(ds, batch_size=16, shuffle=False, num_workers=2,
                        pin_memory=True)

    anat_idx = torch.tensor([ANATOMY_TO_IDX[anatomy]], dtype=torch.long, device=device)
    slices   = []

    for batch in loader:
        mr  = batch["mr"].to(device, non_blocking=True)
        ai  = anat_idx.expand(mr.size(0))
        out = model(mr, ai)
        slices.append(out.squeeze(1).cpu().numpy())

    pred_norm = np.concatenate(slices, axis=0)
    pred_hu   = denormalise_ct(pred_norm)
    return np.clip(pred_hu, -1024, 3000).astype(np.float32)


def save_prediction(pred_hu: np.ndarray, ref_path: Path, out_path: Path):
    ref = sitk.ReadImage(str(ref_path))
    out = sitk.GetImageFromArray(pred_hu)
    out.SetSpacing(ref.GetSpacing())
    out.SetOrigin(ref.GetOrigin())
    out.SetDirection(ref.GetDirection())
    sitk.WriteImage(out, str(out_path))


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    t_start = time.time()
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[process.py] Device: {device}")
    print(f"[process.py] Input:  {INPUT_DIR}")
    print(f"[process.py] Output: {OUTPUT_DIR}")
    print(f"[process.py] Model:  {MODEL_PATH}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Discover input cases
    case_dirs = [d for d in sorted(INPUT_DIR.iterdir())
                 if d.is_dir() and (d / "mr.mha").exists()]

    if not case_dirs:
        print("[process.py] ERROR: No mr.mha files found in input directory.")
        sys.exit(1)

    print(f"[process.py] Cases: {len(case_dirs)}")

    model = load_model(device)
    print("[process.py] Model loaded.")

    errors = []
    for case_dir in case_dirs:
        case_id = case_dir.name
        t_case  = time.time()

        try:
            anatomy = detect_anatomy(case_id)
            print(f"  [{case_id}] anatomy={anatomy}", end=" ... ", flush=True)

            pred_hu  = predict_volume(model, case_dir, anatomy, device)

            out_dir  = OUTPUT_DIR / case_id
            out_dir.mkdir(parents=True, exist_ok=True)
            save_prediction(pred_hu, case_dir / "mr.mha", out_dir / "ct.mha")

            elapsed = time.time() - t_case
            print(f"done ({elapsed:.1f}s)")

        except Exception as e:
            print(f"ERROR: {e}")
            errors.append((case_id, str(e)))

    total = time.time() - t_start
    print(f"\n[process.py] Finished {len(case_dirs) - len(errors)}/{len(case_dirs)} cases "
          f"in {total:.1f}s")

    if errors:
        print("[process.py] Errors:")
        for cid, msg in errors:
            print(f"  {cid}: {msg}")
        sys.exit(1)

    sys.exit(0)


if __name__ == "__main__":
    main()

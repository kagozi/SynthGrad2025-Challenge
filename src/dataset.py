"""
SynthRAD2025 Task 1 — PyTorch Dataset

Supports:
  - 2D slice-based training (default, memory efficient)
  - 3D patch-based training (optional)
  - All 3 anatomies with a single dataset class
  - Anatomy-conditioned via one-hot or integer label
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import SimpleITK as sitk
import torch
from torch.utils.data import Dataset


# ── I/O helpers ────────────────────────────────────────────────────────────────

def load_mha(path: Path) -> np.ndarray:
    """Load .mha file → numpy array (D, H, W), float32."""
    img = sitk.ReadImage(str(path))
    return sitk.GetArrayFromImage(img).astype(np.float32)


def load_mha_with_meta(path: Path):
    img     = sitk.ReadImage(str(path))
    arr     = sitk.GetArrayFromImage(img).astype(np.float32)
    spacing = img.GetSpacing()   # (x, y, z)
    origin  = img.GetOrigin()
    return arr, spacing, origin


# ── Normalisation ───────────────────────────────────────────────────────────────

ANATOMY_MR_PERCENTILES = {
    # Will be filled by running normalisation stats script;
    # these are conservative fallbacks.
    "HN": (0.0, 2500.0),
    "TH": (0.0, 2500.0),
    "AB": (0.0, 2500.0),
}
CT_CLIP = (-1000, 3000)   # HU range to clip before normalising


def normalise_mr(arr: np.ndarray, anatomy: str) -> np.ndarray:
    """Clip to [p1, p99] then scale to [0, 1]."""
    lo, hi = ANATOMY_MR_PERCENTILES.get(anatomy, (0.0, 2500.0))
    arr    = np.clip(arr, lo, hi)
    arr    = (arr - lo) / (hi - lo + 1e-8)
    return arr


def normalise_ct(arr: np.ndarray) -> np.ndarray:
    """Clip HU range then scale to [-1, 1]."""
    lo, hi = CT_CLIP
    arr    = np.clip(arr, lo, hi)
    arr    = (arr - lo) / (hi - lo) * 2.0 - 1.0   # → [-1, 1]
    return arr


def denormalise_ct(arr: np.ndarray) -> np.ndarray:
    """Inverse of normalise_ct → HU."""
    lo, hi = CT_CLIP
    return (arr + 1.0) / 2.0 * (hi - lo) + lo


ANATOMY_TO_IDX = {"HN": 0, "TH": 1, "AB": 2}


# ── 2D Slice Dataset ───────────────────────────────────────────────────────────

class SynthRAD2DDataset(Dataset):
    """
    Yields individual 2D axial slices.

    Each item:
        mr    : (1, H, W) float32 in [0, 1]
        ct    : (1, H, W) float32 in [-1, 1]
        mask  : (1, H, W) bool
        anatomy_idx : int  (0=HN, 1=TH, 2=AB)
        case_id     : str
        slice_idx   : int
    """

    def __init__(
        self,
        case_list: List[Dict],          # list of dicts with keys: case_id, path, anatomy
        fold_df: Optional[pd.DataFrame] = None,
        fold: Optional[int] = None,     # if set, this fold is validation
        split: str = "train",           # "train" or "val"
        slice_axis: int = 0,            # 0=axial (D), 1=coronal, 2=sagittal
        augment: bool = True,
        skip_empty_slices: bool = True,
        empty_threshold: float = 0.01,  # min mask voxel fraction to keep slice
        pad_to: Optional[Tuple[int, int]] = (512, 512),  # pad slices to fixed H×W
    ):
        self.slice_axis         = slice_axis
        self.augment            = augment and (split == "train")
        self.skip_empty         = skip_empty_slices
        self.empty_threshold    = empty_threshold
        self.pad_to             = pad_to

        # Filter by fold
        if fold_df is not None and fold is not None:
            if split == "train":
                cases = fold_df[fold_df["fold"] != fold]["case_id"].tolist()
            else:
                cases = fold_df[fold_df["fold"] == fold]["case_id"].tolist()
            case_set  = set(cases)
            case_list = [c for c in case_list if c["case_id"] in case_set]

        self.case_list = case_list
        self._build_slice_index()

    def _build_slice_index(self):
        """Pre-scan volumes to build (case_idx, slice_idx) index."""
        print(f"[Dataset] Building slice index for {len(self.case_list)} cases...")
        self.index: List[Tuple[int, int]] = []

        for c_idx, case in enumerate(self.case_list):
            mask_path = Path(case["path"]) / "mask.mha"
            mr_path   = Path(case["path"]) / "mr.mha"

            # Use mask to determine which slices have content
            if mask_path.exists():
                mask = load_mha(mask_path)
            else:
                # Fallback: read MR and treat non-zero as mask
                mr   = load_mha(mr_path)
                mask = (mr > 0).astype(np.float32)

            n_slices = mask.shape[self.slice_axis]
            for s in range(n_slices):
                if self.slice_axis == 0:
                    sl = mask[s, :, :]
                elif self.slice_axis == 1:
                    sl = mask[:, s, :]
                else:
                    sl = mask[:, :, s]

                frac = sl.sum() / (sl.size + 1e-8)
                if not self.skip_empty or frac >= self.empty_threshold:
                    self.index.append((c_idx, s))

        print(f"[Dataset] Total slices: {len(self.index)}")

    def __len__(self):
        return len(self.index)

    def _get_slice(self, arr: np.ndarray, s: int) -> np.ndarray:
        if self.slice_axis == 0:
            return arr[s, :, :]
        elif self.slice_axis == 1:
            return arr[:, s, :]
        else:
            return arr[:, :, s]

    def _pad_slice(self, mr: np.ndarray, ct: np.ndarray, mask: np.ndarray):
        """Pad H and W dimensions to self.pad_to with anatomy-appropriate fill."""
        th, tw = self.pad_to
        h,  w  = mr.shape
        ph = max(0, th - h)
        pw = max(0, tw - w)
        if ph == 0 and pw == 0:
            return mr, ct, mask
        pad = ((0, ph), (0, pw))
        mr   = np.pad(mr,   pad, mode="constant", constant_values=0.0)
        ct   = np.pad(ct,   pad, mode="constant", constant_values=-1.0)  # air = -1000 HU
        mask = np.pad(mask, pad, mode="constant", constant_values=0.0)
        return mr, ct, mask

    def _augment(self, mr: np.ndarray, ct: np.ndarray, mask: np.ndarray):
        """Simple 2D augmentation applied identically to MR, CT, mask."""
        # Random horizontal flip
        if random.random() < 0.5:
            mr   = np.fliplr(mr).copy()
            ct   = np.fliplr(ct).copy()
            mask = np.fliplr(mask).copy()

        # Random vertical flip
        if random.random() < 0.3:
            mr   = np.flipud(mr).copy()
            ct   = np.flipud(ct).copy()
            mask = np.flipud(mask).copy()

        # Random brightness/contrast for MR only (not CT — HU must be preserved)
        if random.random() < 0.5:
            factor  = random.uniform(0.85, 1.15)
            shift   = random.uniform(-0.05, 0.05)
            mr      = np.clip(mr * factor + shift, 0.0, 1.0)

        return mr, ct, mask

    def __getitem__(self, idx: int) -> Dict:
        c_idx, s = self.index[idx]
        case     = self.case_list[c_idx]
        anatomy  = case["anatomy"]
        path     = Path(case["path"])

        mr_arr   = load_mha(path / "mr.mha")
        ct_arr   = load_mha(path / "ct.mha")
        mask_arr = load_mha(path / "mask.mha") if (path / "mask.mha").exists() \
                   else np.ones_like(mr_arr)

        mr_arr   = normalise_mr(mr_arr, anatomy)
        ct_arr   = normalise_ct(ct_arr)
        mask_arr = (mask_arr > 0).astype(np.float32)

        mr   = self._get_slice(mr_arr,   s)
        ct   = self._get_slice(ct_arr,   s)
        mask = self._get_slice(mask_arr, s)

        if self.pad_to is not None:
            mr, ct, mask = self._pad_slice(mr, ct, mask)

        if self.augment:
            mr, ct, mask = self._augment(mr, ct, mask)

        return {
            "mr":          torch.from_numpy(mr[None]),        # (1, H, W)
            "ct":          torch.from_numpy(ct[None]),        # (1, H, W)
            "mask":        torch.from_numpy(mask[None]),      # (1, H, W)
            "anatomy_idx": torch.tensor(ANATOMY_TO_IDX[anatomy], dtype=torch.long),
            "case_id":     case["case_id"],
            "slice_idx":   s,
        }


# ── 3D Patch Dataset ───────────────────────────────────────────────────────────

class SynthRAD3DDataset(Dataset):
    """
    Yields random 3D patches — useful for patch-based U-Net or nnUNet-style.

    Each item:
        mr    : (1, D, H, W) float32
        ct    : (1, D, H, W) float32
        mask  : (1, D, H, W) bool
        anatomy_idx : int
    """

    def __init__(
        self,
        case_list: List[Dict],
        fold_df: Optional[pd.DataFrame] = None,
        fold: Optional[int] = None,
        split: str = "train",
        patch_size: Tuple[int, int, int] = (32, 256, 256),
        samples_per_volume: int = 8,
        augment: bool = True,
    ):
        self.patch_size         = patch_size
        self.samples_per_volume = samples_per_volume
        self.augment            = augment and (split == "train")

        if fold_df is not None and fold is not None:
            if split == "train":
                cases = fold_df[fold_df["fold"] != fold]["case_id"].tolist()
            else:
                cases = fold_df[fold_df["fold"] == fold]["case_id"].tolist()
            case_set  = set(cases)
            case_list = [c for c in case_list if c["case_id"] in case_set]

        self.case_list = case_list

    def __len__(self):
        return len(self.case_list) * self.samples_per_volume

    def _random_crop(self, arr: np.ndarray, d: int, h: int, w: int) -> np.ndarray:
        D, H, W = arr.shape
        pd, ph, pw = self.patch_size
        dd = random.randint(0, max(0, D - pd))
        hh = random.randint(0, max(0, H - ph))
        ww = random.randint(0, max(0, W - pw))
        return arr[dd:dd+pd, hh:hh+ph, ww:ww+pw]

    def __getitem__(self, idx: int) -> Dict:
        c_idx   = idx // self.samples_per_volume
        case    = self.case_list[c_idx]
        anatomy = case["anatomy"]
        path    = Path(case["path"])

        mr_arr   = load_mha(path / "mr.mha")
        ct_arr   = load_mha(path / "ct.mha")
        mask_arr = load_mha(path / "mask.mha") if (path / "mask.mha").exists() \
                   else np.ones_like(mr_arr)

        mr_arr   = normalise_mr(mr_arr, anatomy)
        ct_arr   = normalise_ct(ct_arr)
        mask_arr = (mask_arr > 0).astype(np.float32)

        D, H, W = mr_arr.shape
        pd_, ph, pw = self.patch_size

        # Ensure minimum size by padding if needed
        def pad_if_needed(a, target_d, target_h, target_w):
            dd = max(0, target_d - a.shape[0])
            hh = max(0, target_h - a.shape[1])
            ww = max(0, target_w - a.shape[2])
            return np.pad(a, ((0, dd), (0, hh), (0, ww)))

        mr_arr   = pad_if_needed(mr_arr,   pd_, ph, pw)
        ct_arr   = pad_if_needed(ct_arr,   pd_, ph, pw)
        mask_arr = pad_if_needed(mask_arr, pd_, ph, pw)

        d0 = random.randint(0, max(0, mr_arr.shape[0] - pd_))
        h0 = random.randint(0, max(0, mr_arr.shape[1] - ph))
        w0 = random.randint(0, max(0, mr_arr.shape[2] - pw))

        mr   = mr_arr  [d0:d0+pd_, h0:h0+ph, w0:w0+pw]
        ct   = ct_arr  [d0:d0+pd_, h0:h0+ph, w0:w0+pw]
        mask = mask_arr[d0:d0+pd_, h0:h0+ph, w0:w0+pw]

        return {
            "mr":          torch.from_numpy(mr[None]),
            "ct":          torch.from_numpy(ct[None]),
            "mask":        torch.from_numpy(mask[None]),
            "anatomy_idx": torch.tensor(ANATOMY_TO_IDX[anatomy], dtype=torch.long),
            "case_id":     case["case_id"],
        }


# ── Inference Dataset ──────────────────────────────────────────────────────────

class SynthRADInferenceDataset(Dataset):
    """
    Yields full 2D slices for a single case (no shuffling, no augmentation).
    Used during validation/test prediction.
    """

    def __init__(self, case_path: Path, anatomy: str):
        self.case_path = case_path
        self.anatomy   = anatomy

        mr_arr      = load_mha(case_path / "mr.mha")
        self.mr_arr = normalise_mr(mr_arr, anatomy)

        mask_path   = case_path / "mask.mha"
        self.mask   = load_mha(mask_path) if mask_path.exists() \
                      else np.ones_like(self.mr_arr)

        self.n_slices = self.mr_arr.shape[0]
        self.shape    = self.mr_arr.shape   # (D, H, W)

    def __len__(self):
        return self.n_slices

    def __getitem__(self, s: int) -> Dict:
        mr   = self.mr_arr[s, :, :]
        mask = self.mask[s, :, :]
        return {
            "mr":   torch.from_numpy(mr[None].copy()),
            "mask": torch.from_numpy(mask[None].copy()),
            "slice_idx": s,
        }


# ── Factory ────────────────────────────────────────────────────────────────────

def build_case_list(data_dirs) -> List[Dict]:
    """
    Scan one or more data root directories and return a unified case list.

    Handles both the flat layout (data/raw/1HNA001/) and the actual PVC layout:
      {root}/synthRAD2025_Task1_Train/Task1/{AB|HN|TH}/{case_id}/
      {root}/synthRAD2025_Task1_Train_D/Task1/{AB|HN|TH}/{case_id}/

    Args:
        data_dirs: str | Path | list of str/Path — one or more root dirs to scan.
    """
    import re

    if isinstance(data_dirs, (str, Path)):
        data_dirs = [data_dirs]

    case_id_re = re.compile(r"(\d)(HN|TH|AB)([A-E])(\d+)", re.I)
    seen  = set()
    cases = []

    for root in data_dirs:
        root = Path(root)
        if not root.exists():
            continue

        # Collect all candidate directories (any depth up to 4 levels)
        candidates = []
        for pattern in [
            # PVC layout: root/*/Task1/{ANAT}/{case_id}
            "*/Task1/*/*",
            # Flat layout: root/{case_id}
            "*",
            # One level: root/{ANAT}/{case_id}
            "*/*",
        ]:
            hits = [d for d in root.glob(pattern)
                    if d.is_dir() and case_id_re.match(d.name)]
            if hits:
                candidates = hits
                break

        for d in sorted(candidates):
            m = case_id_re.match(d.name)
            if not m:
                continue
            if d.name in seen:
                continue           # deduplicate across roots
            if not (d / "mr.mha").exists():
                continue           # skip incomplete cases

            seen.add(d.name)
            cases.append({
                "case_id": d.name,
                "anatomy": m.group(2).upper(),
                "center":  m.group(3).upper(),
                "path":    str(d),
            })

    return cases

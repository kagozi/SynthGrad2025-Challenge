#!/usr/bin/env python3
"""
SynthRAD2025 Task 1 — 3D MONAI Diffusion Training

Architecture: MONAI DiffusionModelUNet (3D) wrapped in GaussianDiffusion3D.
Conditioning: MR patch channel-concatenated with noisy CT patch.
Loss: noise MSE + Min-SNR weighted MAE + SSIM.
EMA: exponential moving average for stable inference.

Usage:
    python training/train_monai_diffusion.py \
        --config training/configs/monai_diffusion_3d.yaml --fold 0
    python training/train_monai_diffusion.py \
        --config training/configs/monai_diffusion_3d.yaml --fold 0 \
        --resume /pvc/checkpoints/monai_diffusion_3d/fold0_last.pth
"""

import argparse
import os
import random
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import wandb
import yaml
from dotenv import load_dotenv
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.dataset import (
    SynthRAD3DDataset,
    build_case_list,
    denormalise_ct,
    ANATOMY_TO_IDX,
)
from src.metrics import compute_mae, compute_psnr, compute_ms_ssim
from src.models.monai_diffusion_3d import MonaiDiffusion3D, GaussianDiffusion3D, EMA

try:
    from monai.inferers import sliding_window_inference
except ImportError as e:
    raise ImportError("MONAI required: pip install 'monai[all]>=1.4'") from e

load_dotenv()


# ── Helpers ────────────────────────────────────────────────────────────────────

def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark     = True
    torch.backends.cudnn.deterministic = False


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ── Model factory ─────────────────────────────────────────────────────────────

def build_model(cfg: dict) -> GaussianDiffusion3D:
    mc = cfg["model"]
    dc = cfg["diffusion"]
    lc = cfg["loss"]

    backbone = MonaiDiffusion3D(
        in_channels       = mc["in_channels"],
        out_channels      = mc["out_channels"],
        channels          = tuple(mc["channels"]),
        attention_levels  = tuple(mc["attention_levels"]),
        num_res_blocks    = mc["num_res_blocks"],
        num_head_channels = mc["num_head_channels"],
        norm_num_groups   = mc["norm_num_groups"],
        norm_eps          = mc.get("norm_eps", 1e-6),
        n_anatomy         = mc.get("n_anatomy", 3),
        dropout_cattn     = mc.get("dropout_cattn", 0.0),
        use_flash_attention = mc.get("use_flash_attention", True),
    )
    diffusion = GaussianDiffusion3D(
        model            = backbone,
        T                = dc["T"],
        cosine_s         = dc.get("cosine_s", 0.008),
        noise_mse_weight = lc.get("noise_mse_weight", 1.0),
        mae_weight       = lc.get("mae_weight", 0.25),
        ssim_weight      = lc.get("ssim_weight", 0.25),
        snr_gamma        = lc.get("snr_gamma", 5.0),
        vlb_weight       = lc.get("vlb_weight", 0.0),
        vs_step_buckets  = dc.get("vs_step_buckets", None),
    )
    return diffusion


# ── Data loaders ──────────────────────────────────────────────────────────────

def build_loaders(cfg: dict, fold: int):
    dc    = cfg["data"]
    cases = build_case_list(dc["data_dirs"])

    folds_df  = pd.read_csv(dc["folds_csv"])
    train_ids = set(folds_df.loc[folds_df["fold"] != fold, "case_id"])
    val_ids   = set(folds_df.loc[folds_df["fold"] == fold,  "case_id"])

    train_cases = [c for c in cases if c["case_id"] in train_ids]
    val_cases   = [c for c in cases if c["case_id"] in val_ids]

    patch_size = tuple(dc["patch_size"])
    spv        = dc.get("samples_per_volume", 8)

    train_ds = SynthRAD3DDataset(
        train_cases, patch_size=patch_size,
        samples_per_volume=spv, augment=True,
    )
    val_ds = SynthRAD3DDataset(
        val_cases[:20], patch_size=patch_size,
        samples_per_volume=2, augment=False,
    )

    tc = cfg["training"]
    train_loader = DataLoader(
        train_ds,
        batch_size  = tc["batch_size"],
        shuffle     = True,
        num_workers = dc.get("num_workers", 8),
        pin_memory  = dc.get("pin_memory", True),
        drop_last   = True,
        persistent_workers = True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size  = tc["batch_size"],
        shuffle     = False,
        num_workers = max(1, dc.get("num_workers", 8) // 2),
        pin_memory  = dc.get("pin_memory", True),
        drop_last   = False,
    )
    return train_loader, val_loader


# ── Sample figure ─────────────────────────────────────────────────────────────

def make_sample_figure(mr, ct_gt, ct_pred, n: int = 4):
    n   = min(n, mr.shape[0])
    fig, axes = plt.subplots(3, n, figsize=(n * 3, 9))
    for col in range(n):
        mid = mr.shape[2] // 2   # middle depth slice
        for row, (vol, title) in enumerate([
            (mr,      "MR input"),
            (ct_gt,   "CT (GT)"),
            (ct_pred, "sCT (Diffusion)"),
        ]):
            axes[row, col].imshow(vol[col, 0, mid].cpu().numpy(), cmap="gray")
            axes[row, col].axis("off")
            if col == 0:
                axes[row, col].set_ylabel(title, fontsize=9)
    plt.tight_layout()
    return fig


# ── LR scheduler factory ──────────────────────────────────────────────────────

def build_scheduler(optimizer, cfg: dict, max_steps: int):
    sc = cfg.get("scheduler", {})
    name = sc.get("name", "constant_with_warmup")
    warmup = sc.get("warmup_steps", 2500)

    if name == "constant_with_warmup":
        def _lr(step):
            if step < warmup:
                return float(step) / max(warmup, 1)
            return 1.0
        return torch.optim.lr_scheduler.LambdaLR(optimizer, _lr)
    elif name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max   = max_steps - warmup,
            eta_min = sc.get("eta_min", 1e-6),
        )
    return None


# ── Validation ────────────────────────────────────────────────────────────────

@torch.no_grad()
def validate(diffusion, ema, loader, device, ddim_steps: int, use_amp: bool):
    # Use EMA weights for validation
    ema_model = MonaiDiffusion3D.__new__(MonaiDiffusion3D)
    ema_model.__dict__ = diffusion.model.__dict__.copy()
    ema.apply_to(diffusion.model)
    diffusion.eval()

    all_mae, all_psnr, all_ssim = [], [], []
    _autocast = getattr(torch.amp, "autocast", torch.cuda.amp.autocast)

    for batch in tqdm(loader, desc="  val", leave=False):
        mr        = batch["mr"].to(device)
        ct        = batch["ct"].to(device)
        mask      = batch["mask"].to(device)
        anat_idx  = batch["anatomy_idx"].to(device)

        with _autocast("cuda", enabled=use_amp):
            pred = diffusion.ddim_sample(mr, anat_idx, steps=ddim_steps)

        for i in range(pred.shape[0]):
            pred_hu = denormalise_ct(pred[i, 0].cpu().numpy())
            ct_hu   = denormalise_ct(ct[i, 0].cpu().numpy())
            m       = mask[i, 0].cpu().numpy().astype(bool)
            if m.sum() == 0:
                continue
            all_mae.append(compute_mae(pred_hu, ct_hu, m))
            all_psnr.append(compute_psnr(pred_hu, ct_hu, m))
            all_ssim.append(compute_ms_ssim(pred_hu, ct_hu))

    diffusion.train()
    return (
        float(np.mean(all_mae))  if all_mae  else float("inf"),
        float(np.mean(all_psnr)) if all_psnr else 0.0,
        float(np.mean(all_ssim)) if all_ssim else 0.0,
    )


# ── Main training loop ────────────────────────────────────────────────────────

def train(cfg: dict, fold: int, resume: str = None):
    seed_everything(cfg["experiment"]["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    exp_name = f"{cfg['experiment']['name']}_fold{fold}"
    wandb.init(
        project = os.getenv("WANDB_PROJECT", "synthrad2025-task1"),
        entity  = os.getenv("WANDB_ENTITY",  None),
        name    = exp_name,
        config  = cfg,
        resume  = "allow",
        id      = exp_name,
    )

    diffusion = build_model(cfg).to(device)
    n_params  = sum(p.numel() for p in diffusion.model.parameters()) / 1e6
    print(f"MonaiDiffusion3D params: {n_params:.1f}M")

    ema_cfg = cfg.get("ema", {})
    ema     = EMA(diffusion.model, decay=ema_cfg.get("decay", 0.999)) \
              if ema_cfg.get("enabled", True) else None

    train_loader, val_loader = build_loaders(cfg, fold)

    oc = cfg["optimizer"]
    optimizer = torch.optim.AdamW(
        diffusion.parameters(),
        lr           = oc["lr"],
        weight_decay = oc.get("weight_decay", 1e-4),
        betas        = tuple(oc.get("betas", [0.9, 0.999])),
    )

    tc          = cfg["training"]
    max_steps   = tc["max_steps"]
    scheduler   = build_scheduler(optimizer, cfg, max_steps)
    grad_accum  = tc.get("gradient_accumulation", 1)
    use_amp     = tc.get("use_amp", True)

    log_every   = tc.get("steps_per_log",  100)
    val_every   = tc.get("steps_per_val",  2500)
    save_every  = tc.get("steps_per_save", 999999)
    val_steps   = tc.get("val_ddim_steps", 10)

    ckpt_dir    = Path(cfg["output"]["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    _GradScaler = getattr(torch.amp, "GradScaler", torch.cuda.amp.GradScaler)
    scaler      = _GradScaler("cuda", enabled=use_amp)
    _autocast   = getattr(torch.amp, "autocast", torch.cuda.amp.autocast)

    global_step = 0
    best_mae    = float("inf")

    # ── Resume ────────────────────────────────────────────────────────────────
    if resume:
        ckpt        = torch.load(resume, map_location=device)
        diffusion.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        if scheduler and "scheduler" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler"])
        if ema and "ema" in ckpt:
            ema.load_state_dict(ckpt["ema"])
        global_step = ckpt.get("global_step", 0)
        best_mae    = ckpt.get("best_mae", float("inf"))
        print(f"Resumed from {resume} (step {global_step})")

    # ── Training loop ─────────────────────────────────────────────────────────
    diffusion.train()
    loader_iter  = iter(train_loader)
    accum_loss   = 0.0
    accum_parts: dict = {}

    pbar = tqdm(total=max_steps, initial=global_step, desc="Training")

    optimizer.zero_grad(set_to_none=True)

    while global_step < max_steps:
        try:
            batch = next(loader_iter)
        except StopIteration:
            loader_iter = iter(train_loader)
            batch       = next(loader_iter)

        mr       = batch["mr"].to(device)
        ct       = batch["ct"].to(device)
        mask     = batch["mask"].to(device)
        anat_idx = batch["anatomy_idx"].to(device)

        t = diffusion.sample_t(mr.shape[0], device)   # VS-DDPM aware

        with _autocast("cuda", enabled=use_amp):
            loss, parts = diffusion.p_loss(ct, mr, t, anat_idx, mask)
            loss = loss / grad_accum

        scaler.scale(loss).backward()

        accum_loss += loss.item() * grad_accum
        for k, v in parts.items():
            accum_parts[k] = accum_parts.get(k, 0.0) + v / grad_accum

        # Gradient accumulation step
        if (global_step + 1) % grad_accum == 0 or (global_step + 1) == max_steps:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(diffusion.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            if scheduler:
                scheduler.step()
            if ema:
                ema.update(diffusion.model)

        global_step += 1
        pbar.update(1)

        # ── Logging ───────────────────────────────────────────────────────────
        if global_step % log_every == 0:
            log_dict = {
                "train/loss": accum_loss / log_every,
                "step": global_step,
                "lr":   optimizer.param_groups[0]["lr"],
            }
            for k, v in accum_parts.items():
                log_dict[f"train/{k}"] = v / log_every
            wandb.log(log_dict)
            accum_loss  = 0.0
            accum_parts = {}

        # ── Validation ────────────────────────────────────────────────────────
        if global_step % val_every == 0:
            mae, psnr, ssim = validate(
                diffusion, ema or EMA(diffusion.model),
                val_loader, device, val_steps, use_amp,
            )
            print(f"\n  Step {global_step}: MAE={mae:.2f} HU | PSNR={psnr:.2f} dB | MS-SSIM={ssim:.4f}")
            wandb.log({"val/mae": mae, "val/psnr": psnr, "val/ms_ssim": ssim, "step": global_step})

            # Sample visualisation
            try:
                batch_v = next(iter(val_loader))
                mr_v    = batch_v["mr"][:4].to(device)
                ct_v    = batch_v["ct"][:4].to(device)
                ai_v    = batch_v["anatomy_idx"][:4].to(device)
                ema.apply_to(diffusion.model)
                with torch.no_grad(), _autocast("cuda", enabled=use_amp):
                    pred_v = diffusion.ddim_sample(mr_v, ai_v, steps=val_steps)
                fig = make_sample_figure(mr_v, ct_v, pred_v)
                wandb.log({"val/samples": wandb.Image(fig), "step": global_step})
                plt.close(fig)
            except Exception:
                pass

            if mae < best_mae:
                best_mae = mae
                try:
                    _save(ckpt_dir / f"fold{fold}_best.pth",
                          diffusion, optimizer, scheduler, ema, global_step, best_mae, cfg)
                    print(f"  New best MAE: {best_mae:.2f} HU — saved.")
                except Exception as e:
                    print(f"  WARNING: failed to save best checkpoint: {e}")

    pbar.close()
    try:
        _save(ckpt_dir / f"fold{fold}_last.pth",
              diffusion, optimizer, scheduler, ema, global_step, best_mae, cfg)
    except Exception as e:
        print(f"  WARNING: failed to save last checkpoint: {e}")
    wandb.finish()
    print(f"Training complete. Best val MAE: {best_mae:.2f} HU")


def _save(path, diffusion, optimizer, scheduler, ema, step, best_mae, cfg):
    ckpt = {
        "model":       diffusion.state_dict(),
        "optimizer":   optimizer.state_dict(),
        "global_step": step,
        "best_mae":    best_mae,
        "cfg":         cfg,
    }
    if scheduler:
        ckpt["scheduler"] = scheduler.state_dict()
    if ema:
        ckpt["ema"] = ema.state_dict()
    torch.save(ckpt, path)


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--fold",   type=int, default=None)
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()

    cfg  = load_config(args.config)
    fold = args.fold if args.fold is not None else cfg["experiment"].get("fold", 0)

    print(f"Training 3D MONAI Diffusion  |  config: {args.config}  |  fold: {fold}")
    train(cfg, fold, args.resume)


if __name__ == "__main__":
    main()

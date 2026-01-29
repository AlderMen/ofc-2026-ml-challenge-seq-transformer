"""
OFC 2026 ML Challenge: Sequence Transformer (v3)
Solution for EDFA Gain Profile Prediction

This script implements a Transformer-based sequence model to predict EDFA gain profiles
from input spectra and device parameters. It includes features like multi-scale token 
pooling, local attention masking, and ensemble training.

Authors: Koopman Lab
Date: Jan 2026
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants and Path Configuration
ROOT = Path(__file__).resolve().parent
# Default data path - users can override this via symlinks or script modification
DATA_ROOT = ROOT / "ofc-ml-challenge-data-code-main"

@dataclass
class Scaler:
    """Standard Scaler for data normalization."""
    mean: np.ndarray
    std: np.ndarray

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / self.std

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        return x * self.std + self.mean

def _fit_scaler(x: np.ndarray, eps: float = 1e-8) -> Scaler:
    """Fits a scaler to the provided data."""
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    std = np.where(std < eps, 1.0, std)
    return Scaler(mean=mean, std=std)

def _find_channel_cols(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """Identifies input spectra and WSS activation columns."""
    in_cols = sorted([c for c in df.columns if c.lower().startswith("edfa_input_spectra_")])
    wss_cols = sorted([c for c in df.columns if "dut_wss_activated_channel_index" in c.lower()])
    if len(in_cols) != 95 or len(wss_cols) != 95:
        raise ValueError(f"Expected 95 channels, got input={len(in_cols)} wss={len(wss_cols)}")
    return in_cols, wss_cols

def _coord95() -> np.ndarray:
    """Generates normalized coordinates [-1, 1] for 95 channels."""
    idx = np.arange(95, dtype=np.float32)
    center = (95 - 1) / 2.0
    return ((idx - center) / center).astype(np.float32)

def _gain_prior_from_targets(df: pd.DataFrame) -> np.ndarray:
    """Computes initial gain prior based on target gain and tilt."""
    tg = df["target_gain"].values.astype(np.float32) if "target_gain" in df.columns else np.zeros((len(df),), np.float32)
    tt = df["target_gain_tilt"].values.astype(np.float32) if "target_gain_tilt" in df.columns else np.zeros((len(df),), np.float32)
    coord = _coord95()
    return (tg[:, None] + tt[:, None] * coord[None, :]).astype(np.float32)

def _build_channel_features(
    df: pd.DataFrame,
    category_levels: Optional[List[str]] = None,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], List[str]]:
    """
    Constructs the input feature matrix (N, 95, Cin) for the Transformer.
    """
    df = df.copy()
    in_cols, wss_cols = _find_channel_cols(df)
    in_spec = df[in_cols].values.astype(np.float32)
    mask = df[wss_cols].values.astype(np.float32)

    def get_col(name: str, default: float = 0.0) -> np.ndarray:
        return df[name].values.astype(np.float32) if name in df.columns else np.full((len(df),), default, np.float32)

    target_gain = get_col("target_gain")
    target_tilt = get_col("target_gain_tilt")
    p_in = get_col("EDFA_input_power_total")
    p_out = get_col("EDFA_output_power_total")
    edfa_index = get_col("edfa_index", default=-1.0)

    if "EDFA_type" in df.columns:
        edfa_type = (df["EDFA_type"].astype(str).str.lower() == "booster").astype(np.float32).values
    else:
        edfa_type = np.zeros((len(df),), np.float32)

    used_levels: List[str] = []
    if "Category" in df.columns:
        cat = df["Category"].astype(str)
        used_levels = sorted(cat.unique()) if category_levels is None else list(category_levels)
        cat_oh = np.stack([(cat == name).astype(np.float32).values for name in used_levels], axis=1)
    else:
        used_levels = [] if category_levels is None else list(category_levels)
        cat_oh = np.zeros((len(df), len(used_levels)), dtype=np.float32)

    coord = _coord95()
    ch_idx = (np.arange(95, dtype=np.float32) / 94.0).astype(np.float32)
    tilt = (target_tilt[:, None] * coord[None, :]).astype(np.float32)
    gain_per_ch = (target_gain[:, None] + tilt).astype(np.float32)

    N = len(df)
    p_in_rep = np.repeat(p_in[:, None], 95, axis=1)
    p_out_rep = np.repeat(p_out[:, None], 95, axis=1)
    edfa_type_rep = np.repeat(edfa_type[:, None], 95, axis=1)
    edfa_index_rep = np.repeat(edfa_index[:, None], 95, axis=1)
    coord_rep = np.repeat(coord[None, :], N, axis=0)
    ch_idx_rep = np.repeat(ch_idx[None, :], N, axis=0)

    # Load block features
    loaded = mask > 0.5
    load_starts = np.zeros((N, 95), dtype=np.float32)
    load_ends = np.zeros((N, 95), dtype=np.float32)
    for i in range(N):
        ld = loaded[i]
        if not np.any(ld): continue
        starts = np.where(np.diff(np.concatenate([[False], ld])) == 1)[0]
        ends = np.where(np.diff(np.concatenate([ld, [False]])) == -1)[0]
        for s, e in zip(starts, ends):
            load_starts[i, s:e] = s / 94.0
            load_ends[i, s:e] = e / 94.0

    # Spectrum stats
    spec_mean = np.mean(in_spec, axis=1, keepdims=True).astype(np.float32)
    spec_std = np.std(in_spec, axis=1, keepdims=True).astype(np.float32)
    spec_min = np.min(in_spec, axis=1, keepdims=True).astype(np.float32)
    spec_max = np.max(in_spec, axis=1, keepdims=True).astype(np.float32)
    
    spec_mean_rep = np.repeat(spec_mean, 95, axis=1)
    spec_std_rep = np.repeat(spec_std, 95, axis=1)
    spec_min_rep = np.repeat(spec_min, 95, axis=1)
    spec_max_rep = np.repeat(spec_max, 95, axis=1)

    # Power density
    power_density = (in_spec / (p_in_rep + 1e-8)).astype(np.float32)

    # EDFA one-hot (assuming 8 device types)
    edfa_onehot = np.zeros((N, 95, 8), dtype=np.float32)
    edfa_idx_int = np.round(edfa_index).astype(np.int32)
    for i in range(N):
        idx = int(edfa_idx_int[i])
        if 0 <= idx < 8:
            edfa_onehot[i, :, idx] = 1.0

    feats = [
        in_spec, mask, gain_per_ch, tilt, coord_rep, ch_idx_rep,
        load_starts, load_ends, spec_mean_rep, spec_std_rep,
        spec_min_rep, spec_max_rep, power_density, p_in_rep, p_out_rep,
        edfa_type_rep, edfa_index_rep
    ]

    if cat_oh.shape[1] > 0:
        for k in range(cat_oh.shape[1]):
            feats.append(np.repeat(cat_oh[:, k : k + 1], 95, axis=1))

    for j in range(8):
        feats.append(edfa_onehot[:, :, j])

    X = np.stack(feats, axis=-1).astype(np.float32)
    ids = df["ID"].values.astype(np.int64) if "ID" in df.columns else None
    return X, mask, ids, used_levels

def _build_labels(df_y: pd.DataFrame) -> np.ndarray:
    """Extracts target gain labels from dataframe."""
    y_cols = sorted([c for c in df_y.columns if "calculated_gain_spectra" in c.lower()])
    if len(y_cols) != 95:
        raise ValueError(f"Expected 95 gain labels, got {len(y_cols)}")
    return df_y[y_cols].values.astype(np.float32)

# --- Loss Functions ---

def masked_mse_db(pred_n: torch.Tensor, target_n: torch.Tensor, mask: torch.Tensor, y_std_db: torch.Tensor) -> torch.Tensor:
    """Computes Masked Mean Squared Error in dB."""
    err_db = (pred_n - target_n) * y_std_db
    se = (err_db ** 2) * mask
    denom = torch.clamp(mask.sum(dim=1), min=1.0)
    return (se.sum(dim=1) / denom).mean()

def masked_mse_db_per_row(pred_n: torch.Tensor, target_n: torch.Tensor, mask: torch.Tensor, y_std_db: torch.Tensor) -> torch.Tensor:
    """Computes Masked MSE per sample row."""
    err_db = (pred_n - target_n) * y_std_db
    se = (err_db ** 2) * mask
    denom = torch.clamp(mask.sum(dim=1), min=1.0)
    return se.sum(dim=1) / denom

def kaggle_like_loss_db(
    pred_n: torch.Tensor,
    target_n: torch.Tensor,
    mask: torch.Tensor,
    y_std_db: torch.Tensor,
    p95_weight: float = 0.3,
    max_weight: float = 0.1,
    std_weight: float = 0.15,
    tau: float = 0.5,
    beta: float = 0.7,
) -> torch.Tensor:
    """
    Kaggle-inspired loss function combining MAE, Standard Deviation, P95, and Max error.
    Optimized for the competition evaluation metric.
    """
    err_db = torch.abs(pred_n - target_n) * y_std_db * mask
    denom = torch.clamp(mask.sum(dim=1), min=1.0)
    row_mae = err_db.sum(dim=1) / denom
    mae = row_mae.mean()

    row_mean = row_mae[:, None]
    row_var = ((err_db - row_mean) ** 2).sum(dim=1) / denom
    row_std = torch.sqrt(torch.clamp(row_var, min=0.0))
    std_term = row_std.mean()

    flat = err_db[mask > 0.5]
    if flat.numel() == 0:
        return mae + std_weight * std_term
    
    p95 = torch.quantile(flat, 0.95)
    t95 = torch.relu(p95 - mae.detach() - tau)
    mx = torch.max(flat)
    tmax = torch.relu(mx - p95.detach() - beta)
    
    return mae + std_weight * std_term + p95_weight * t95 + max_weight * tmax

def _dilate_mask(mask: np.ndarray, k: int) -> np.ndarray:
    """Expands activation mask by k channels on each side."""
    if k <= 0: return mask > 0.5
    m = (mask > 0.5)
    vis = m.copy()
    for d in range(1, k + 1):
        vis[:, d:] |= m[:, :-d]
        vis[:, :-d] |= m[:, d:]
    return vis

class SeqTransformerV3(nn.Module):
    """
    Sequence Transformer architecture for spectral data.
    Supports global and pooled segment tokens.
    """
    def __init__(
        self,
        in_channels: int,
        d_model: int = 256,
        nhead: int = 8,
        depth: int = 8,
        dim_ff: int = 768,
        dropout: float = 0.1,
        use_pos_emb: bool = True,
        n_pool_tokens: int = 0,
    ):
        super().__init__()
        self.d_model = d_model
        self.use_pos_emb = use_pos_emb
        self.n_pool_tokens = int(max(0, n_pool_tokens))

        self.proj = nn.Linear(in_channels, d_model)
        self.global_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pool_token_bias = nn.Parameter(torch.zeros(1, self.n_pool_tokens, d_model)) if self.n_pool_tokens > 0 else None

        self.seq_len = 1 + self.n_pool_tokens + 95
        self.pos_emb = nn.Parameter(torch.zeros(1, self.seq_len, d_model)) if use_pos_emb else None

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.enc = nn.TransformerEncoder(enc_layer, num_layers=depth)

        self.head = nn.Sequential(
            nn.LayerNorm(d_model * 2),
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(d_model, 1),
        )

        nn.init.normal_(self.global_token, std=0.02)
        if self.pool_token_bias is not None:
            nn.init.normal_(self.pool_token_bias, std=0.02)
        if self.pos_emb is not None:
            nn.init.normal_(self.pos_emb, std=0.02)

    def forward(self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B = x.shape[0]
        h = self.proj(x)
        g = self.global_token.expand(B, -1, -1)

        toks = [g]
        if self.n_pool_tokens > 0:
            segs = torch.chunk(h, chunks=self.n_pool_tokens, dim=1)
            pooled = torch.stack([s.mean(dim=1) for s in segs], dim=1)
            pooled = pooled + self.pool_token_bias.expand(B, -1, -1)
            toks.append(pooled)
        toks.append(h)
        
        z = torch.cat(toks, dim=1)
        if self.pos_emb is not None:
            z = z + self.pos_emb
        
        z = self.enc(z, src_key_padding_mask=key_padding_mask)

        g2 = z[:, :1, :]
        h2 = z[:, (1 + self.n_pool_tokens) :, :]
        g_rep = g2.expand(-1, 95, -1)
        y = self.head(torch.cat([h2, g_rep], dim=-1)).squeeze(-1)
        return y

# --- Utilities ---

def _make_out_dirs(out_dir: Path) -> dict[str, Path]:
    """Ensures output directories exist."""
    d = Path(out_dir)
    ckpt = d / "checkpoints"
    sc = d / "scalers"
    sub = d / "submissions"
    for p in [ckpt, sc, sub]: p.mkdir(parents=True, exist_ok=True)
    return {"root": d, "checkpoints": ckpt, "scalers": sc, "submissions": sub}

def write_submission(ids: np.ndarray, pred_db: np.ndarray, mask: np.ndarray, out_path: Path, unloaded_value: float = -18.0) -> None:
    """Formats and writes prediction to CSV for Kaggle."""
    pred_db = np.where(mask > 0.5, pred_db, unloaded_value).astype(np.float32)
    pred_db = np.nan_to_num(pred_db, nan=unloaded_value, posinf=unloaded_value, neginf=unloaded_value)
    cols = ["ID"] + [f"calculated_gain_spectra_{i:02d}" for i in range(95)]
    df = pd.DataFrame(np.concatenate([ids[:, None], pred_db], axis=1), columns=cols)
    df["ID"] = df["ID"].astype(int)
    df.to_csv(out_path, index=False)

def build_kpm(mb: torch.Tensor, n_pool: int, mode: str, k: int) -> torch.Tensor:
    if mode == "full":
        ch_keep = torch.ones_like(mb, dtype=torch.bool)
    elif mode == "pad_unloaded":
        ch_keep = (mb > 0.5)
    else:
        m = (mb > 0.5).cpu().numpy()
        vis = _dilate_mask(m, int(k))
        ch_keep = torch.from_numpy(vis).to(mb.device).bool()
    tok_keep = torch.ones((mb.shape[0], 1 + int(n_pool)), device=mb.device, dtype=torch.bool)
    keep = torch.cat([tok_keep, ch_keep], dim=1)
    return (~keep).bool()

def main() -> None:
    p = argparse.ArgumentParser(description="OFC 2026 ML Challenge SeqTransformer Training & Prediction")
    # Basic Controls
    p.add_argument("--train", action="store_true")
    p.add_argument("--predict", action="store_true")
    p.add_argument("--out_dir", type=str, default="runs/default")
    p.add_argument("--tag", type=str, default="model")
    
    # Training Hyperparams
    p.add_argument("--epochs", type=int, default=250)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--amp", action="store_true", help="Enable Mixed Precision")
    p.add_argument("--save_best", action="store_true")
    
    # Ensemble
    p.add_argument("--n_ensemble", type=int, default=1)
    p.add_argument("--ensemble", action="store_true", help="Predict using ensemble")
    
    # Model Config
    p.add_argument("--d_model", type=int, default=256)
    p.add_argument("--nhead", type=int, default=8)
    p.add_argument("--depth", type=int, default=8)
    p.add_argument("--dim_ff", type=int, default=768)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--n_pool_tokens", type=int, default=0)
    p.add_argument("--attn_mode", type=str, default="pad_unloaded", choices=["pad_unloaded", "context_k", "full"])
    p.add_argument("--unloaded_context_k", type=int, default=2)
    
    # Dataset specific
    p.add_argument("--cosmos_max", type=int, default=10000)
    p.add_argument("--train_mode", type=str, default="two_stage", choices=["two_stage", "mixed"])
    p.add_argument("--residual_from_target", action="store_true")
    p.add_argument("--loss", type=str, default="mse", choices=["mse", "kaggle_proxy"])

    # Fine-tuning
    p.add_argument("--finetune_kaggle_epochs", type=int, default=120)
    p.add_argument("--finetune_lr", type=float, default=2e-4)
    p.add_argument("--finetune_loss", type=str, default="kaggle_proxy")
    p.add_argument("--finetune_val_frac", type=float, default=0.2)

    args = p.parse_args()
    out_dirs = _make_out_dirs(Path(args.out_dir))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Data Paths
    train_feat_path = DATA_ROOT / "Features" / "Train" / "train_features.csv"
    train_lab_path = DATA_ROOT / "Features" / "Train" / "train_labels.csv"
    test_feat_path = DATA_ROOT / "Features" / "Test" / "test_features.csv"

    def get_loss_fn(name: str):
        if name == "mse": return masked_mse_db
        return lambda p, t, m, s: kaggle_like_loss_db(p, t, m, s, p95_weight=0.3, max_weight=0.1, std_weight=0.15)

    if args.train:
        logger.info("Starting Training...")
        X_df = pd.read_csv(train_feat_path)
        y_df = pd.read_csv(train_lab_path)
        
        is_cosmos_row = None
        if "Category" in X_df.columns:
            is_cosmos_row = X_df["Category"].eq("COSMOS").values
            if args.cosmos_max >= 0:
                cosmos_idx = np.where(is_cosmos_row)[0]
                other_idx = np.where(~is_cosmos_row)[0]
                if len(cosmos_idx) > args.cosmos_max:
                    rng = np.random.default_rng(args.seed)
                    keep_cosmos = rng.choice(cosmos_idx, size=args.cosmos_max, replace=False)
                    keep = np.sort(np.concatenate([other_idx, keep_cosmos]))
                    X_df, y_df = X_df.iloc[keep].reset_index(drop=True), y_df.iloc[keep].reset_index(drop=True)
                    is_cosmos_row = X_df["Category"].eq("COSMOS").values
                    logger.info(f"COSMOS downsampled to {args.cosmos_max} samples.")

        X, mask, _, cat_levels = _build_channel_features(X_df)
        y = _build_labels(y_df)
        prior = _gain_prior_from_targets(X_df) if args.residual_from_target else 0
        y_used = y - prior

        # Normalization
        x_scaler = _fit_scaler(X.reshape(-1, X.shape[-1]))
        Xn = x_scaler.transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)
        
        y_loaded = y_used[mask > 0.5]
        y_scaler = _fit_scaler(y_loaded[:, None])
        yn = y_scaler.transform(y_used.reshape(-1, 1)).reshape(y_used.shape)
        y_std_db = float(y_scaler.std.flatten()[0])

        # Train/Val Split
        rng = np.random.default_rng(args.seed)
        if is_cosmos_row is not None and np.any(is_cosmos_row):
            kaggle_idx = np.where(~is_cosmos_row)[0]
            cosmos_idx = np.where(is_cosmos_row)[0]
            rng.shuffle(kaggle_idx)
            n_val = int(0.2 * len(kaggle_idx))
            val_idx, tr_idx = kaggle_idx[:n_val], np.concatenate([kaggle_idx[n_val:], cosmos_idx])
        else:
            indices = rng.permutation(len(Xn))
            n_val = int(0.2 * len(Xn))
            val_idx, tr_idx = indices[:n_val], indices[n_val:]

        def create_loader(ii, shuffle, bs):
            ds = TensorDataset(torch.from_numpy(Xn[ii]), torch.from_numpy(yn[ii]), torch.from_numpy(mask[ii]))
            return DataLoader(ds, batch_size=bs, shuffle=shuffle)

        tr_loader = create_loader(tr_idx, True, args.batch_size)
        va_loader = create_loader(val_idx, False, args.batch_size)

        # Save Metadata
        np.savez(out_dirs["scalers"] / f"{args.tag}.npz", 
                 x_mean=x_scaler.mean, x_std=x_scaler.std, 
                 y_mean=y_scaler.mean, y_std=y_scaler.std,
                 cat_levels=np.array(cat_levels, dtype=object),
                 args=vars(args))

        for ens_idx in range(args.n_ensemble):
            seed = args.seed + ens_idx * 10
            torch.manual_seed(seed)
            logger.info(f"Training Ensemble Model {ens_idx+1}/{args.n_ensemble} (Seed {seed})")

            model = SeqTransformerV3(in_channels=X.shape[-1], d_model=args.d_model, nhead=args.nhead, 
                                     depth=args.depth, dim_ff=args.dim_ff, dropout=args.dropout, 
                                     n_pool_tokens=args.n_pool_tokens).to(device)
            
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
            scaler = torch.cuda.amp.GradScaler(enabled=args.amp)
            
            y_std_t = torch.tensor(y_std_db, device=device)
            loss_fn = get_loss_fn(args.loss)
            best_val = float('inf')
            best_state = None

            # Stage 1: Full Training
            for epoch in range(args.epochs):
                model.train()
                for xb, yb, mb in tr_loader:
                    xb, yb, mb = xb.to(device), yb.to(device), mb.to(device)
                    optimizer.zero_grad()
                    kpm = build_kpm(mb, args.n_pool_tokens, args.attn_mode, args.unloaded_context_k)
                    with torch.cuda.amp.autocast(enabled=args.amp):
                        pred = model(xb, key_padding_mask=kpm)
                        loss = loss_fn(pred, yb, mb, y_std_t)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                scheduler.step()
                
                model.eval()
                val_loss = 0
                with torch.no_grad():
                    for xb, yb, mb in va_loader:
                        xb, yb, mb = xb.to(device), yb.to(device), mb.to(device)
                        kpm = build_kpm(mb, args.n_pool_tokens, args.attn_mode, args.unloaded_context_k)
                        pred = model(xb, key_padding_mask=kpm)
                        val_loss += loss_fn(pred, yb, mb, y_std_t).item()
                val_loss /= len(va_loader)

                if val_loss < best_val:
                    best_val = val_loss
                    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                if (epoch + 1) % 50 == 0:
                    logger.info(f"Epoch {epoch+1}/{args.epochs} - Val Loss: {val_loss:.4f}")

            # Stage 2: Optional Two-Stage Finetuning on Kaggle Data Only
            if args.train_mode == "two_stage" and args.finetune_kaggle_epochs > 0 and is_cosmos_row is not None:
                logger.info(f"Stage 2: Finetuning on Kaggle Data (Seed {seed})")
                if best_state is not None: model.load_state_dict(best_state)
                
                kaggle_only_idx = np.where(~is_cosmos_row)[0]
                rng_ft = np.random.default_rng(seed)
                rng_ft.shuffle(kaggle_only_idx)
                n_val_ft = int(args.finetune_val_frac * len(kaggle_only_idx))
                val_ft, tr_ft = kaggle_only_idx[:n_val_ft], kaggle_only_idx[n_val_ft:]
                
                ft_loader = create_loader(tr_ft, True, args.batch_size)
                ft_val_loader = create_loader(val_ft, False, args.batch_size)
                
                optimizer_ft = torch.optim.AdamW(model.parameters(), lr=args.finetune_lr, weight_decay=args.weight_decay)
                scheduler_ft = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_ft, T_max=args.finetune_kaggle_epochs)
                loss_fn_ft = get_loss_fn(args.finetune_loss)
                best_val_ft = float('inf')

                for epoch in range(args.finetune_kaggle_epochs):
                    model.train()
                    for xb, yb, mb in ft_loader:
                        xb, yb, mb = xb.to(device), yb.to(device), mb.to(device)
                        optimizer_ft.zero_grad()
                        kpm = build_kpm(mb, args.n_pool_tokens, args.attn_mode, args.unloaded_context_k)
                        with torch.cuda.amp.autocast(enabled=args.amp):
                            pred = model(xb, key_padding_mask=kpm)
                            loss = loss_fn_ft(pred, yb, mb, y_std_t)
                        scaler.scale(loss).backward()
                        scaler.step(optimizer_ft)
                        scaler.update()
                    scheduler_ft.step()

                    model.eval()
                    v_loss = 0
                    with torch.no_grad():
                        for xb, yb, mb in ft_val_loader:
                            xb, yb, mb = xb.to(device), yb.to(device), mb.to(device)
                            kpm = build_kpm(mb, args.n_pool_tokens, args.attn_mode, args.unloaded_context_k)
                            pred = model(xb, key_padding_mask=kpm)
                            v_loss += loss_fn_ft(pred, yb, mb, y_std_t).item()
                    v_loss /= len(ft_val_loader)
                    if v_loss < best_val_ft:
                        best_val_ft = v_loss
                        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

            if best_state is not None and args.save_best:
                model.load_state_dict(best_state)
                path = out_dirs["checkpoints"] / (f"{args.tag}.pth" if args.n_ensemble == 1 else f"{args.tag}_{ens_idx}.pth")
                torch.save(model.state_dict(), path)
                logger.info(f"Model saved to {path} (Best Val: {min(best_val, best_val_ft) if 'best_val_ft' in locals() else best_val:.4f})")

    if args.predict:
        logger.info("Starting Prediction...")
        meta = np.load(out_dirs["scalers"] / f"{args.tag}.npz", allow_pickle=True)
        test_df = pd.read_csv(test_feat_path)
        X, mask, ids, _ = _build_channel_features(test_df, category_levels=meta['cat_levels'].tolist())
        prior = _gain_prior_from_targets(test_df)

        x_scaler = Scaler(meta['x_mean'], meta['x_std'])
        y_scaler = Scaler(meta['y_mean'], meta['y_std'])
        Xn = x_scaler.transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)

        ens_count = args.n_ensemble if args.ensemble else 1
        preds = []
        
        for i in range(ens_count):
            path = out_dirs["checkpoints"] / (f"{args.tag}.pth" if ens_count == 1 else f"{args.tag}_{i}.pth")
            if not path.exists():
                logger.warning(f"Checkpoint {path} not found, skipping.")
                continue
            model = SeqTransformerV3(in_channels=X.shape[-1], d_model=args.d_model, nhead=args.nhead, 
                                     depth=args.depth, dim_ff=args.dim_ff, dropout=args.dropout, 
                                     n_pool_tokens=args.n_pool_tokens).to(device)
            model.load_state_dict(torch.load(path, map_location=device))
            model.eval()
            
            p_batch = []
            with torch.no_grad():
                for s in range(0, len(Xn), 256):
                    e = s + 256
                    xb = torch.from_numpy(Xn[s:e]).to(device)
                    mb = torch.from_numpy(mask[s:e]).to(device)
                    kpm = build_kpm(mb, args.n_pool_tokens, args.attn_mode, args.unloaded_context_k)
                    p_batch.append(model(xb, key_padding_mask=kpm).cpu().numpy())
            preds.append(np.concatenate(p_batch, axis=0))

        if not preds: raise RuntimeError("No models found for prediction.")
        pred_n = np.mean(preds, axis=0)
        pred_db = y_scaler.inverse_transform(pred_n.reshape(-1, 1)).reshape(pred_n.shape)
        if args.residual_from_target: pred_db += prior
        
        out_path = out_dirs["submissions"] / f"submission_{args.tag}.csv"
        write_submission(ids, pred_db, mask, out_path)
        logger.info(f"Submission saved to {out_path}")

if __name__ == "__main__":
    main()

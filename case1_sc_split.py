"""
Case 1: SC-wise Channel Prediction (Same Bandwidth)
=====================================================
- Subcarrier 64개를 각각 독립 sample로 취급
- 각 sample: input (PAST_LEN=16, 2*N_RX=32), target (32,)
- Train/Val 분리: 시간 기준 75% / 25% (겹침 없음)
- Models: LSTM, LWM

Usage:
    python case1_sc_split.py
    python case1_sc_split.py --scenes 1000 --cuda 0 --epochs 200
"""

import os
import sys
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset as TorchDataset, DataLoader

from deepverse import ParameterManager, Dataset
from lwm_model1 import lwm

# ─────────────────────────────────────────────────────────────
# 0. Argument
# ─────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--scenes",  type=int,   default=1000)
parser.add_argument("--cuda",    type=int,   default=0)
parser.add_argument("--epochs",  type=int,   default=200)
parser.add_argument("--batch",   type=int,   default=256)
parser.add_argument("--lr",      type=float, default=1e-4)
parser.add_argument("--past_len",type=int,   default=16)
args = parser.parse_args()

N_SCENES  = args.scenes
PAST_LEN  = args.past_len
EPOCHS    = args.epochs
BATCH     = args.batch
LR        = args.lr
N_SC      = 64
N_RX      = 16
F_SC      = 2 * N_RX   # 32: real(16) + imag(16) of Rx antennas per SC
DEVICE    = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")

print(f"Device: {DEVICE} | Scenes: {N_SCENES} | Epochs: {EPOCHS} | Batch: {BATCH}")

# ─────────────────────────────────────────────────────────────
# 1. Dataset Loading
# ─────────────────────────────────────────────────────────────
scenarios_name = "DT31"
config_path    = f"scenarios/{scenarios_name}/param/config.m"

param_manager = ParameterManager(config_path)
param_manager.params["scenes"] = list(range(N_SCENES))
param_manager.params["comm"]["OFDM"]["selected_subcarriers"] = list(range(N_SC))
param_manager.params["radar"]["enable"] = False

dataset = Dataset(param_manager)

def flatten_comm_frames(comm):
    frames = []
    for row in comm.data:
        for d in row:
            frames.append(d)
    return frames

def get_coeffs_from_frame(frame, ue_idx=0):
    ue_obj = frame["ue"]
    ch_obj = ue_obj[ue_idx] if isinstance(ue_obj, (list, tuple)) else ue_obj
    if hasattr(ch_obj, "coeffs"):
        return ch_obj.coeffs
    if isinstance(ch_obj, dict) and "coeffs" in ch_obj:
        return ch_obj["coeffs"]
    raise TypeError(f"Cannot get coeffs: {type(ue_obj)}, {type(ch_obj)}")

comm_frames = flatten_comm_frames(dataset.comm_dataset)
N = len(comm_frames)
print(f"Total comm frames: {N}")

# channel shape per frame: (N_RX=16, N_TX=1, N_SC=64) complex
sample = get_coeffs_from_frame(comm_frames[0])
print(f"Channel shape per frame: {sample.shape}")  # (16, 1, 64)

# ─────────────────────────────────────────────────────────────
# 2. Precompute channel tensor for efficiency
#    raw_data: (N, N_SC, N_RX, 2)  [last dim: real/imag]
# ─────────────────────────────────────────────────────────────
print("Precomputing raw channel data...")
raw_data = np.zeros((N, N_SC, N_RX, 2), dtype=np.float32)
for t in range(N):
    c = get_coeffs_from_frame(comm_frames[t])  # (16, 1, 64)
    for sc in range(N_SC):
        sc_vals = c[:, 0, sc]          # (16,) complex
        raw_data[t, sc, :, 0] = sc_vals.real
        raw_data[t, sc, :, 1] = sc_vals.imag
print("Done.")

# ─────────────────────────────────────────────────────────────
# 3. Time-based Train/Val Split
# ─────────────────────────────────────────────────────────────
valid_start = PAST_LEN - 1   # 15
valid_end   = N - 2          # need t+1 for target

all_t    = list(range(valid_start, valid_end + 1))
n_total  = len(all_t)
n_train  = int(n_total * 0.75)

train_t  = all_t[:n_train]
val_t    = all_t[n_train:]

print(f"Time windows — total: {n_total} | train: {len(train_t)} | val: {len(val_t)}")
print(f"Samples      — train: {len(train_t)*N_SC:,} | val: {len(val_t)*N_SC:,}")

# ─────────────────────────────────────────────────────────────
# 4. Min-Max normalization (train statistics only)
# ─────────────────────────────────────────────────────────────
eps = 1e-16
channel_data = np.zeros((N, N_SC, F_SC), dtype=np.float32)
print("Computing global min/max from train set...")
train_real = raw_data[train_t, :, :, 0]   # (n_train, N_SC, N_RX)
train_imag = raw_data[train_t, :, :, 1]

r_min, r_max = float(train_real.min()), float(train_real.max())
i_min, i_max = float(train_imag.min()), float(train_imag.max())
print(f"Global scale real: [{r_min:.6f}, {r_max:.6f}]")
print(f"Global scale imag: [{i_min:.6f}, {i_max:.6f}]")

channel_data[:, :, :N_RX] = (raw_data[:, :, :, 0] - r_min) / max(r_max - r_min, eps)
channel_data[:, :, N_RX:] = (raw_data[:, :, :, 1] - i_min) / max(i_max - i_min, eps)

channel_tensor = torch.from_numpy(channel_data).to(DEVICE)  # (N, N_SC, F_SC=32)
print(f"channel_tensor: {channel_tensor.shape} on {channel_tensor.device}")

# ─────────────────────────────────────────────────────────────
# 5. Dataset
# ─────────────────────────────────────────────────────────────
class SCwiseDataset(TorchDataset):
    """
    각 sample = (SC, time_window) 쌍
    Input:  (PAST_LEN, F_SC) = (16, 32)
    Target: (F_SC,)          = (32,)
    """
    def __init__(self, channel_tensor, sc_list, time_indices, past_len=16):
        self.ch       = channel_tensor   # (N, N_SC, F_SC) — already on GPU
        self.past_len = past_len
        # 모든 (t, sc) 조합을 sample로 등록
        self.samples  = [(t, sc) for t in time_indices for sc in sc_list]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        t, sc = self.samples[idx]
        ch_past = self.ch[t - self.past_len + 1 : t + 1, sc, :]  # (16, 32)
        target  = self.ch[t + 1, sc, :]                           # (32,)
        return ch_past, target

all_sc       = list(range(N_SC))
train_dataset = SCwiseDataset(channel_tensor, all_sc, train_t, PAST_LEN)
val_dataset   = SCwiseDataset(channel_tensor, all_sc, val_t,   PAST_LEN)

train_loader = DataLoader(train_dataset, batch_size=BATCH, shuffle=True,  num_workers=0, pin_memory=False)
val_loader   = DataLoader(val_dataset,   batch_size=BATCH, shuffle=False, num_workers=0, pin_memory=False)

ch_s, y_s = next(iter(train_loader))
print(f"Batch — ch: {ch_s.shape}, y: {y_s.shape}")   # (B, 16, 32), (B, 32)

# ─────────────────────────────────────────────────────────────
# 6. Models
# ─────────────────────────────────────────────────────────────

class LSTMForecaster(nn.Module):
    """
    Input : ch (B, T, F_in=32)
    Output: yhat (B, F_out=32)
    """
    def __init__(self, F_in, F_out, hidden=256, num_layers=2, dropout=0.1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=F_in, hidden_size=hidden, num_layers=num_layers,
            batch_first=True, dropout=(dropout if num_layers > 1 else 0.0)
        )
        self.head = nn.Linear(hidden, F_out)

    def forward(self, ch):
        out, _ = self.lstm(ch)      # (B, T, hidden)
        return self.head(out[:, -1, :])  # (B, F_out)


class LWMForecaster(nn.Module):
    """
    LWM backbone을 직접 호출 (forward 수정 없이 embedding + layers 사용)
    Input : ch (B, T, F_in=32)
    Output: yhat (B, F_out=32)
    """
    def __init__(self, base_lwm, F_in, F_out, freeze_backbone=False):
        super().__init__()
        self.base = base_lwm
        n_dim   = self.base.embedding.element_length  # 64
        d_model = self.base.embedding.d_model          # 128

        # F_in=32 → element_length=64 projection
        self.in_proj = nn.Linear(F_in, n_dim)
        self.head    = nn.Linear(d_model, F_out)

        if freeze_backbone:
            for p in self.base.parameters():
                p.requires_grad = False

    def forward(self, ch):
        x = self.in_proj(ch)                  # (B, T, 64)
        out = self.base.embedding(x)           # (B, T, 128)
        for layer in self.base.layers:
            out, _ = layer(out)               # (B, T, 128)
        z = out[:, -1, :]                     # (B, 128)
        return self.head(z)                   # (B, 32)

# ─────────────────────────────────────────────────────────────
# 7. Training utilities
# ─────────────────────────────────────────────────────────────

@torch.no_grad()
def nmse_db(yhat, y, eps=1e-16):
    num = torch.sum((yhat - y) ** 2, dim=1)
    den = torch.sum(y ** 2, dim=1).clamp_min(eps)
    return (10.0 * torch.log10((num / den).clamp_min(eps))).mean().item()


def train_one_epoch(model, loader, optimizer, grad_clip=1.0):
    model.train()
    total_loss, total_nmse, n = 0.0, 0.0, 0
    for ch, y in loader:
        optimizer.zero_grad(set_to_none=True)
        yhat = model(ch)
        loss = F.mse_loss(yhat, y)
        loss.backward()
        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        total_loss += loss.item()
        total_nmse += nmse_db(yhat.detach(), y)
        n += 1
    return total_loss / max(n, 1), total_nmse / max(n, 1)


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    total_loss, total_nmse, n = 0.0, 0.0, 0
    for ch, y in loader:
        yhat = model(ch)
        loss = F.mse_loss(yhat, y)
        total_loss += loss.item()
        total_nmse += nmse_db(yhat, y)
        n += 1
    return total_loss / max(n, 1), total_nmse / max(n, 1)


def run_experiment(model_name, model, log_path, ckpt_path):
    print(f"\n{'='*55}")
    print(f" {model_name} | params: {sum(p.numel() for p in model.parameters()):,}")
    print(f"{'='*55}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_val, best_nmse, best_epoch = float("inf"), None, None
    t0_total = time.time()

    with open(log_path, "w", encoding="utf-8") as f:
        header = (
            f"=== {model_name} | Case1 SC-wise | scenes={N_SCENES} | "
            f"train={len(train_dataset):,} | val={len(val_dataset):,} ===\n"
        )
        print(header.strip()); f.write(header)

        for epoch in range(1, EPOCHS + 1):
            t0 = time.time()
            tr_loss, tr_nmse = train_one_epoch(model, train_loader, optimizer)
            va_loss, va_nmse = evaluate(model, val_loader)
            scheduler.step()

            line = (
                f"[{epoch:03d}/{EPOCHS}] "
                f"train loss={tr_loss:.6f} nmse={tr_nmse:.4f}dB | "
                f"val loss={va_loss:.6f} nmse={va_nmse:.4f}dB | "
                f"{time.time()-t0:.1f}s"
            )
            print(line); f.write(line + "\n"); f.flush()

            if va_loss < best_val:
                best_val, best_nmse, best_epoch = va_loss, va_nmse, epoch
                torch.save({
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "F_in": F_SC, "F_out": F_SC,
                    "best_val_loss": best_val,
                    "best_val_nmse_db": best_nmse,
                    "n_scenes": N_SCENES,
                }, ckpt_path)
                ckpt_line = f"  -> saved {ckpt_path} | best@{epoch}: val nmse={best_nmse:.4f}dB"
                print(ckpt_line); f.write(ckpt_line + "\n"); f.flush()

        total = time.time() - t0_total
        h, m, s = int(total//3600), int((total%3600)//60), total%60
        summary = (
            f"\n=== Done === {h}h {m}m {s:.0f}s | "
            f"Best epoch {best_epoch}: val loss={best_val:.6f} nmse={best_nmse:.4f}dB"
        )
        print(summary); f.write(summary + "\n")

    return best_val, best_nmse

# ─────────────────────────────────────────────────────────────
# 8. Run Experiments
# ─────────────────────────────────────────────────────────────
results = {}

# --- LSTM ---
lstm_model = LSTMForecaster(F_in=F_SC, F_out=F_SC, hidden=256, num_layers=2).to(DEVICE)
lstm_loss, lstm_nmse = run_experiment(
    model_name="LSTM",
    model=lstm_model,
    log_path=f"case1_lstm_scene{N_SCENES}.txt",
    ckpt_path=f"case1_lstm_scene{N_SCENES}_best.pt",
)
results["LSTM"] = lstm_nmse

# --- LWM ---
backbone  = lwm().to(DEVICE)   # 사전학습 없이 초기화 (model_weights.pth가 있다면 아래 주석 해제)
# backbone = lwm.from_pretrained("model_weights.pth", device=DEVICE)
lwm_model = LWMForecaster(base_lwm=backbone, F_in=F_SC, F_out=F_SC).to(DEVICE)
lwm_loss, lwm_nmse = run_experiment(
    model_name="LWM",
    model=lwm_model,
    log_path=f"case1_lwm_scene{N_SCENES}.txt",
    ckpt_path=f"case1_lwm_scene{N_SCENES}_best.pt",
)
results["LWM"] = lwm_nmse

# ─────────────────────────────────────────────────────────────
# 9. Final Summary
# ─────────────────────────────────────────────────────────────
print(f"\n{'='*55}")
print(f" Final Results — Case 1 SC-wise | scenes={N_SCENES}")
print(f"{'='*55}")
for model_name, nmse in results.items():
    print(f"  {model_name:6s}: best val NMSE = {nmse:.4f} dB")
print(f"{'='*55}")

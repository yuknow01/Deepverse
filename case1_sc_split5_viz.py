"""
case1_sc_split5_viz.py
─────────────────────
Case 1 SC-wise split5 (N_SC=512, scenes=1000, batch=256) 학습 결과 시각화

사용법:
    python case1_sc_split5_viz.py              # N_SCENES=1000 (default)
    python case1_sc_split5_viz.py --scenes 1000
"""

import re
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

SPLIT_TAG = "split5"
N_SC      = 512

COLORS = {
    "LSTM":            "#2196F3",
    "lwm_fromScratch": "#F44336",
    "lwm_finetune":    "#4CAF50",
    "lwm_freeze":      "#FF9800",
}
LINESTYLES = {
    "LSTM":            "-",
    "lwm_fromScratch": "--",
    "lwm_finetune":    "-.",
    "lwm_freeze":      ":",
}

EPOCH_RE = re.compile(
    r"\[(\d+)/\d+\].*?"
    r"train loss=([\d.]+) nmse=([-\d.]+)dB.*?"
    r"val loss=([\d.]+) nmse=([-\d.]+)dB"
)
BEST_RE = re.compile(r"Best epoch (\d+):.*?nmse=([-\d.]+)dB")


def parse_log(path: Path):
    epochs, tr_loss, va_loss, tr_nmse, va_nmse = [], [], [], [], []
    best_epoch, best_nmse = None, None
    with open(path, encoding="utf-8") as f:
        for line in f:
            m = EPOCH_RE.search(line)
            if m:
                epochs.append(int(m.group(1)))
                tr_loss.append(float(m.group(2)))
                tr_nmse.append(float(m.group(3)))
                va_loss.append(float(m.group(4)))
                va_nmse.append(float(m.group(5)))
                continue
            m = BEST_RE.search(line)
            if m:
                best_epoch = int(m.group(1))
                best_nmse  = float(m.group(2))
    return {
        "epochs":     np.array(epochs),
        "tr_loss":    np.array(tr_loss),
        "va_loss":    np.array(va_loss),
        "tr_nmse":    np.array(tr_nmse),
        "va_nmse":    np.array(va_nmse),
        "best_epoch": best_epoch,
        "best_nmse":  best_nmse,
    }


def plot_curves(results, title_suffix, out_prefix):
    # ── Figure 1: 2×2 학습 곡선 ─────────────────────────────────
    fig1, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig1.suptitle(f"Case 1 SC-wise  |  {title_suffix}", fontsize=14, fontweight="bold")

    panel_cfg = [
        (axes[0, 0], "tr_loss", "Train Loss (MSE)",  False),
        (axes[0, 1], "va_loss", "Val Loss (MSE)",     False),
        (axes[1, 0], "tr_nmse", "Train NMSE (dB)",    True),
        (axes[1, 1], "va_nmse", "Val NMSE (dB)",      True),
    ]
    for ax, key, title, is_nmse in panel_cfg:
        for name, d in results.items():
            ax.plot(d["epochs"], d[key],
                    color=COLORS[name], ls=LINESTYLES[name],
                    lw=1.5, label=name, alpha=0.9)
            if key in ("va_loss", "va_nmse") and d["best_epoch"]:
                be = d["best_epoch"] - 1
                if be < len(d[key]):
                    ax.scatter(d["epochs"][be], d[key][be],
                               color=COLORS[name], s=60, zorder=5,
                               marker="*", edgecolors="black", linewidths=0.5)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("dB" if is_nmse else "MSE")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    fig1.tight_layout()

    # ── Figure 2: Best Val NMSE 바 차트 ─────────────────────────
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    fig2.suptitle(f"Best Val NMSE (dB)  |  {title_suffix}", fontsize=13, fontweight="bold")
    names = list(results.keys())
    vals  = [results[n]["best_nmse"] for n in names]
    bars  = ax2.bar(names, vals, color=[COLORS[n] for n in names],
                    width=0.5, edgecolor="black", linewidth=0.7)
    for bar, val in zip(bars, vals):
        if val is not None:
            ax2.text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() - 0.3,
                     f"{val:.3f} dB",
                     ha="center", va="top", fontsize=10, fontweight="bold", color="white")
    ax2.set_ylabel("NMSE (dB)")
    valid_vals = [v for v in vals if v is not None]
    if valid_vals:
        ax2.set_ylim(min(valid_vals) - 2, max(valid_vals) + 1)
    ax2.axhline(0, color="gray", lw=0.8, ls="--")
    ax2.grid(True, axis="y", alpha=0.3)
    fig2.tight_layout()

    # ── Figure 3: Val NMSE 단독 곡선 ────────────────────────────
    fig3, ax3 = plt.subplots(figsize=(10, 5))
    fig3.suptitle(f"Val NMSE (dB) — All Models  |  {title_suffix}", fontsize=13, fontweight="bold")
    for name, d in results.items():
        best_s = f"{d['best_nmse']:.3f} dB" if d["best_nmse"] is not None else "N/A"
        ax3.plot(d["epochs"], d["va_nmse"],
                 color=COLORS[name], ls=LINESTYLES[name],
                 lw=2, label=f"{name}  (best {best_s})")
        if d["best_epoch"]:
            be = d["best_epoch"] - 1
            if be < len(d["va_nmse"]):
                ax3.scatter(d["epochs"][be], d["va_nmse"][be],
                            color=COLORS[name], s=80, zorder=5,
                            marker="*", edgecolors="black", linewidths=0.6)
    ax3.set_xlabel("Epoch", fontsize=11)
    ax3.set_ylabel("Val NMSE (dB)", fontsize=11)
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)
    ax3.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    fig3.tight_layout()

    # ── 저장 ─────────────────────────────────────────────────────
    for fig, tag in [(fig1, "curves"), (fig2, "bar"), (fig3, "val_nmse")]:
        p = f"{out_prefix}_{tag}.png"
        fig.savefig(p, dpi=150, bbox_inches="tight")
        print(f"  저장: {p}")
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenes",  type=int, default=1000)
    parser.add_argument("--log_dir", type=str, default=".")
    parser.add_argument("--out",     type=str, default=None)
    args = parser.parse_args()

    N       = args.scenes
    log_dir = Path(args.log_dir)

    model_files = {
        "LSTM":            log_dir / f"case1_{SPLIT_TAG}_lstm_scene{N}.txt",
        "lwm_fromScratch": log_dir / f"case1_{SPLIT_TAG}_lwm_fromScratch_scene{N}.txt",
        "lwm_finetune":    log_dir / f"case1_{SPLIT_TAG}_lwm_finetune_scene{N}.txt",
        "lwm_freeze":      log_dir / f"case1_{SPLIT_TAG}_lwm_freeze_scene{N}.txt",
    }

    results = {}
    for name, path in model_files.items():
        if path.exists():
            results[name] = parse_log(path)
            d = results[name]
            best_s = f"{d['best_nmse']:.4f} dB" if d["best_nmse"] is not None else "N/A"
            print(f"[✓] {name:20s}  best val NMSE = {best_s}  (epoch {d['best_epoch']})")
        else:
            print(f"[✗] {name:20s}  파일 없음: {path}")

    if not results:
        print("파싱할 로그 파일이 없습니다.")
        return

    out_prefix = args.out or str(log_dir / f"case1_sc_{SPLIT_TAG}_viz_scene{N}")
    title      = f"{SPLIT_TAG}  |  N_SC={N_SC}  |  batch=256  |  scenes={N}"
    plot_curves(results, title, out_prefix)


if __name__ == "__main__":
    main()

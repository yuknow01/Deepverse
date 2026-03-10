"""
case1_sc_split2_viz.py
─────────────────────
Case 1 SC-wise 학습 결과 시각화 스크립트

사용법:
    python case1_sc_split2_viz.py              # N_SCENES=1000
    python case1_sc_split2_viz.py --scenes 500
"""

import re
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# ────────────────────────────────────────────────
# Config
# ────────────────────────────────────────────────
COLORS = {
    "LSTM":           "#2196F3",   # blue
    "lwm_fromScratch": "#F44336",  # red
    "lwm_finetune":    "#4CAF50",  # green
    "lwm_freeze":      "#FF9800",  # orange
}
LINESTYLES = {
    "LSTM":           "-",
    "lwm_fromScratch": "--",
    "lwm_finetune":    "-.",
    "lwm_freeze":      ":",
}


# ────────────────────────────────────────────────
# Parser
# ────────────────────────────────────────────────
# [001/200] train loss=0.050307 nmse=-7.3044dB | val loss=0.017240 nmse=-12.0050dB | 1.1s
EPOCH_RE = re.compile(
    r"\[(\d+)/\d+\].*?"
    r"train loss=([\d.]+) nmse=([-\d.]+)dB.*?"
    r"val loss=([\d.]+) nmse=([-\d.]+)dB"
)
# Best epoch line
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
        "epochs":    np.array(epochs),
        "tr_loss":   np.array(tr_loss),
        "va_loss":   np.array(va_loss),
        "tr_nmse":   np.array(tr_nmse),
        "va_nmse":   np.array(va_nmse),
        "best_epoch": best_epoch,
        "best_nmse":  best_nmse,
    }


# ────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenes", type=int, default=1000)
    parser.add_argument("--log_dir", type=str, default=".")
    parser.add_argument("--out",  type=str, default=None,
                        help="저장 경로 (미지정 시 자동)")
    args = parser.parse_args()

    N = args.scenes
    log_dir = Path(args.log_dir)

    model_files = {
        "LSTM":            log_dir / f"case1_lstm_scene{N}.txt",
        "lwm_fromScratch": log_dir / f"case1_lwm_fromScratch_scene{N}.txt",
        "lwm_finetune":    log_dir / f"case1_lwm_finetune_scene{N}.txt",
        "lwm_freeze":      log_dir / f"case1_lwm_freeze_scene{N}.txt",
    }

    # 존재하는 파일만 파싱
    results = {}
    for name, path in model_files.items():
        if path.exists():
            results[name] = parse_log(path)
            print(f"[✓] {name:20s}  best val NMSE = {results[name]['best_nmse']:.4f} dB  (epoch {results[name]['best_epoch']})")
        else:
            print(f"[✗] {name:20s}  파일 없음: {path}")

    if not results:
        print("파싱할 로그 파일이 없습니다.")
        return

    # ────────────────────────────────────────────
    # Figure 1: 학습 곡선 (2×2)
    # ────────────────────────────────────────────
    fig1, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig1.suptitle(f"Case 1 SC-wise  |  scenes={N}", fontsize=14, fontweight="bold")

    ax_tr_loss, ax_va_loss = axes[0]
    ax_tr_nmse, ax_va_nmse = axes[1]

    panel_cfg = [
        (ax_tr_loss, "tr_loss",  "Train Loss (MSE)",      False),
        (ax_va_loss, "va_loss",  "Val Loss (MSE)",         False),
        (ax_tr_nmse, "tr_nmse",  "Train NMSE (dB)",        True),
        (ax_va_nmse, "va_nmse",  "Val NMSE (dB)",          True),
    ]

    for ax, key, title, is_nmse in panel_cfg:
        for name, d in results.items():
            ax.plot(
                d["epochs"], d[key],
                color=COLORS[name], ls=LINESTYLES[name],
                lw=1.5, label=name, alpha=0.9,
            )
            # best epoch 마커 (val 패널만)
            if key in ("va_loss", "va_nmse") and d["best_epoch"] is not None:
                be = d["best_epoch"] - 1   # 0-indexed
                if be < len(d[key]):
                    ax.scatter(
                        d["epochs"][be], d[key][be],
                        color=COLORS[name], s=60, zorder=5,
                        marker="*", edgecolors="black", linewidths=0.5,
                    )
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("dB" if is_nmse else "MSE")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    fig1.tight_layout()

    # ────────────────────────────────────────────
    # Figure 2: Best Val NMSE 비교 바 차트
    # ────────────────────────────────────────────
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    fig2.suptitle(f"Best Val NMSE (dB) Comparison  |  scenes={N}",
                  fontsize=13, fontweight="bold")

    names  = list(results.keys())
    nmse_vals = [results[n]["best_nmse"] for n in names]
    colors = [COLORS[n] for n in names]

    bars = ax2.bar(names, nmse_vals, color=colors, width=0.5, edgecolor="black", linewidth=0.7)
    for bar, val in zip(bars, nmse_vals):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() - 0.3,
            f"{val:.3f} dB",
            ha="center", va="top", fontsize=10, fontweight="bold", color="white",
        )

    ax2.set_ylabel("NMSE (dB)")
    ax2.set_ylim(min(nmse_vals) - 2, max(nmse_vals) + 1)
    ax2.axhline(0, color="gray", lw=0.8, ls="--")
    ax2.grid(True, axis="y", alpha=0.3)
    fig2.tight_layout()

    # ────────────────────────────────────────────
    # Figure 3: Val NMSE 곡선 단독 (크게)
    # ────────────────────────────────────────────
    fig3, ax3 = plt.subplots(figsize=(10, 5))
    fig3.suptitle(f"Val NMSE (dB) — All Models  |  scenes={N}",
                  fontsize=13, fontweight="bold")

    for name, d in results.items():
        ax3.plot(
            d["epochs"], d["va_nmse"],
            color=COLORS[name], ls=LINESTYLES[name],
            lw=2, label=f"{name}  (best {d['best_nmse']:.3f} dB)",
        )
        if d["best_epoch"] is not None:
            be = d["best_epoch"] - 1
            if be < len(d["va_nmse"]):
                ax3.scatter(
                    d["epochs"][be], d["va_nmse"][be],
                    color=COLORS[name], s=80, zorder=5,
                    marker="*", edgecolors="black", linewidths=0.6,
                )

    ax3.set_xlabel("Epoch", fontsize=11)
    ax3.set_ylabel("Val NMSE (dB)", fontsize=11)
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)
    ax3.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    fig3.tight_layout()

    # ────────────────────────────────────────────
    # Save
    # ────────────────────────────────────────────
    out_prefix = args.out if args.out else str(log_dir / f"case1_sc_split2_viz_scene{N}")
    fig1.savefig(f"{out_prefix}_curves.png", dpi=150, bbox_inches="tight")
    fig2.savefig(f"{out_prefix}_bar.png",    dpi=150, bbox_inches="tight")
    fig3.savefig(f"{out_prefix}_val_nmse.png", dpi=150, bbox_inches="tight")
    print(f"\n저장 완료:")
    print(f"  {out_prefix}_curves.png")
    print(f"  {out_prefix}_bar.png")
    print(f"  {out_prefix}_val_nmse.png")

    plt.show()


if __name__ == "__main__":
    main()

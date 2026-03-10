"""
case1_sc_compare_viz.py
────────────────────────
split2 / split3 / split4 통합 비교 시각화

  split2 : scenes=1000, batch=256, N_SC=64   (baseline)
  split3 : scenes=1000, batch=64,  N_SC=64
  split4 : scenes=7000, batch=256, N_SC=64
  split5 : scenes=1000, batch=256, N_SC=512

사용법:
    python case1_sc_compare_viz.py
    python case1_sc_compare_viz.py --splits split2 split4   # 원하는 split만 선택
"""

import re
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# ── 실험 메타 정보 ────────────────────────────────────────────────
SPLIT_META = {
    "split2": {"scenes": 1000, "batch": 256, "n_sc":  64, "color": "#1565C0"},   # 진파랑
    "split3": {"scenes": 1000, "batch":  64, "n_sc":  64, "color": "#B71C1C"},   # 진빨강
    "split4": {"scenes": 7000, "batch": 256, "n_sc":  64, "color": "#1B5E20"},   # 진초록
    "split5": {"scenes": 1000, "batch": 256, "n_sc": 512, "color": "#4A148C"},   # 진보라
}

MODEL_NAMES   = ["LSTM", "lwm_fromScratch", "lwm_finetune", "lwm_freeze"]
MODEL_DISPLAY = ["LSTM", "Scratch", "Finetune", "Freeze"]

MODEL_COLORS = {
    "LSTM":            "#2196F3",
    "lwm_fromScratch": "#F44336",
    "lwm_finetune":    "#4CAF50",
    "lwm_freeze":      "#FF9800",
}
MODEL_LS = {
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


def get_log_path(log_dir: Path, split_key: str, model_key: str):
    model_file = {
        "LSTM":            "lstm",
        "lwm_fromScratch": "lwm_fromScratch",
        "lwm_finetune":    "lwm_finetune",
        "lwm_freeze":      "lwm_freeze",
    }[model_key]
    N = SPLIT_META[split_key]["scenes"]
    # split2 는 태그 없는 원본 파일명 사용
    if split_key == "split2":
        return log_dir / f"case1_{model_file}_scene{N}.txt"
    return log_dir / f"case1_{split_key}_{model_file}_scene{N}.txt"


def split_label(split_key):
    m = SPLIT_META[split_key]
    return f"{split_key} (sc={m['scenes']}, b={m['batch']}, SC={m['n_sc']})"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits",  nargs="+", default=list(SPLIT_META.keys()),
                        help="비교할 split 목록 (예: split2 split4)")
    parser.add_argument("--log_dir", type=str, default=".")
    parser.add_argument("--out",     type=str, default=None)
    args = parser.parse_args()

    log_dir      = Path(args.log_dir)
    active_splits = args.splits

    # ── 데이터 로드 ────────────────────────────────────────────────
    data = {}
    for sp in active_splits:
        data[sp] = {}
        for mk in MODEL_NAMES:
            path = get_log_path(log_dir, sp, mk)
            if path.exists():
                data[sp][mk] = parse_log(path)
                d = data[sp][mk]
                nmse_str = f"{d['best_nmse']:.4f} dB" if d["best_nmse"] is not None else "N/A"
                print(f"[✓] {sp} / {mk:20s}  best={nmse_str}  (ep {d['best_epoch']})")
            else:
                print(f"[✗] {sp} / {mk:20s}  없음: {path.name}")

    loaded_splits = [sp for sp in active_splits if data[sp]]
    if not loaded_splits:
        print("로드된 데이터가 없습니다.")
        return

    tag = "_".join(active_splits)
    out_prefix = args.out or str(log_dir / f"case1_sc_compare_{tag}")

    # ── Figure 1: 모델별 Val NMSE 비교 (2×2) ─────────────────────
    fig1, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig1.suptitle("Val NMSE (dB) per Model  —  " + " vs ".join(active_splits),
                  fontsize=13, fontweight="bold")

    for i, (mk, md) in enumerate(zip(MODEL_NAMES, MODEL_DISPLAY)):
        ax = axes.flatten()[i]
        for sp in loaded_splits:
            if mk not in data[sp]:
                continue
            d   = data[sp][mk]
            col = SPLIT_META[sp]["color"]
            best_s = f"{d['best_nmse']:.3f}dB" if d["best_nmse"] is not None else "N/A"
            lbl = f"{split_label(sp)}  best={best_s}"
            ax.plot(d["epochs"], d["va_nmse"], color=col, lw=1.8, label=lbl)
            if d["best_epoch"]:
                be = d["best_epoch"] - 1
                if be < len(d["va_nmse"]):
                    ax.scatter(d["epochs"][be], d["va_nmse"][be],
                               color=col, s=70, zorder=5,
                               marker="*", edgecolors="black", linewidths=0.5)
        ax.set_title(md, fontsize=11)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Val NMSE (dB)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    fig1.tight_layout()
    p1 = f"{out_prefix}_val_nmse_per_model.png"
    fig1.savefig(p1, dpi=150, bbox_inches="tight")
    print(f"\n저장: {p1}")

    # ── Figure 2: Best Val NMSE 그룹 바 차트 ─────────────────────
    fig2, ax2 = plt.subplots(figsize=(11, 5))
    fig2.suptitle("Best Val NMSE (dB) Comparison  —  " + " vs ".join(active_splits),
                  fontsize=13, fontweight="bold")

    x     = np.arange(len(MODEL_NAMES))
    n_sp  = len(loaded_splits)
    width = 0.7 / n_sp

    for si, sp in enumerate(loaded_splits):
        vals = []
        for mk in MODEL_NAMES:
            v = data[sp].get(mk, {}).get("best_nmse")
            vals.append(v if v is not None else float("nan"))
        offset = (si - (n_sp - 1) / 2) * width
        bars = ax2.bar(x + offset, vals, width,
                       label=split_label(sp),
                       color=SPLIT_META[sp]["color"],
                       edgecolor="black", linewidth=0.7, alpha=0.85)
        for bar, val in zip(bars, vals):
            if not np.isnan(val):
                ax2.text(bar.get_x() + bar.get_width() / 2,
                         bar.get_height() - 0.15,
                         f"{val:.3f}", ha="center", va="top",
                         fontsize=8, fontweight="bold", color="white")

    ax2.set_xticks(x)
    ax2.set_xticklabels(MODEL_DISPLAY, fontsize=11)
    ax2.set_ylabel("NMSE (dB)")
    all_vals = [data[s][m]["best_nmse"] for s in loaded_splits
                for m in MODEL_NAMES if m in data[s] and data[s][m]["best_nmse"] is not None]
    if all_vals:
        ax2.set_ylim(min(all_vals) - 2, max(all_vals) + 1)
    ax2.axhline(0, color="gray", lw=0.8, ls="--")
    ax2.legend(fontsize=10)
    ax2.grid(True, axis="y", alpha=0.3)
    fig2.tight_layout()
    p2 = f"{out_prefix}_bar_compare.png"
    fig2.savefig(p2, dpi=150, bbox_inches="tight")
    print(f"저장: {p2}")

    # ── Figure 3: 전체 Val NMSE 한 플롯 (모델=색, split=선종류) ───
    fig3, ax3 = plt.subplots(figsize=(13, 6))
    fig3.suptitle("Val NMSE — All Models & Splits  —  " + " vs ".join(active_splits),
                  fontsize=13, fontweight="bold")

    # split linestyle 매핑 (3개)
    split_ls = {}
    ls_pool  = ["-", "--", "-."]
    for idx, sp in enumerate(loaded_splits):
        split_ls[sp] = ls_pool[idx % len(ls_pool)]

    for sp in loaded_splits:
        for mk in MODEL_NAMES:
            if mk not in data[sp]:
                continue
            d   = data[sp][mk]
            best_s2 = f"{d['best_nmse']:.3f}dB" if d["best_nmse"] is not None else "N/A"
            lbl = f"[{sp}] {mk} ({best_s2})"
            ax3.plot(d["epochs"], d["va_nmse"],
                     color=MODEL_COLORS[mk], ls=split_ls[sp],
                     lw=1.8, label=lbl, alpha=0.85)

    ax3.set_xlabel("Epoch", fontsize=11)
    ax3.set_ylabel("Val NMSE (dB)", fontsize=11)
    ax3.grid(True, alpha=0.3)
    ax3.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    # 범례 분리: 모델색 + split 선종류
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    model_handles = [Patch(color=MODEL_COLORS[mk], label=md)
                     for mk, md in zip(MODEL_NAMES, MODEL_DISPLAY)]
    split_handles = [Line2D([0],[0], color="gray", ls=split_ls[sp], lw=2,
                            label=split_label(sp)) for sp in loaded_splits]
    leg1 = ax3.legend(handles=model_handles, title="Model",
                      fontsize=9, loc="lower left", bbox_to_anchor=(0.01, 0.01))
    ax3.add_artist(leg1)
    ax3.legend(handles=split_handles, title="Split",
               fontsize=9, loc="lower left", bbox_to_anchor=(0.18, 0.01))
    fig3.tight_layout()
    p3 = f"{out_prefix}_val_nmse_all.png"
    fig3.savefig(p3, dpi=150, bbox_inches="tight")
    print(f"저장: {p3}")

    plt.show()

    # ── 텍스트 요약 테이블 ─────────────────────────────────────────
    col_w = 16
    print(f"\n{'='*(20 + col_w * len(loaded_splits))}")
    header = f"{'Model':<20}" + "".join(f"{split_label(sp):>{col_w}}" for sp in loaded_splits)
    print(header)
    print("-" * len(header))
    for mk, md in zip(MODEL_NAMES, MODEL_DISPLAY):
        row = f"{md:<20}"
        for sp in loaded_splits:
            v = data[sp].get(mk, {}).get("best_nmse")
            row += f"{(f'{v:.4f} dB') if v is not None else 'N/A':>{col_w}}"
        print(row)
    print("=" * len(header))


if __name__ == "__main__":
    main()

"""
根据 results 下各模型的 *_history.jsonl 绘制训练曲线：
- x 轴：step
- train_loss（蓝色）
- eval_loss（橙色虚线）
- eval_macro_f1（绿色圆点，右侧 y 轴 0~1）

用法（在 results 目录下）：
  python generate_figure.py
  python generate_figure.py --model bert_base_uncased
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_history(path: Path) -> tuple[list, list, list, list, list]:
    train_steps, train_losses = [], []
    eval_steps, eval_losses, eval_f1s = [], [], []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        if "loss" in row and "step" in row and "eval_loss" not in row:
            train_steps.append(row["step"])
            train_losses.append(row["loss"])
        if "eval_loss" in row and "step" in row:
            eval_steps.append(row["step"])
            eval_losses.append(row["eval_loss"])
            eval_f1s.append(row.get("eval_macro_f1", float("nan")))
    return train_steps, train_losses, eval_steps, eval_losses, eval_f1s


def plot_one(
    name: str,
    train_steps, train_losses, eval_steps, eval_losses, eval_f1s,
    out_path: Path,
) -> None:
    import matplotlib.pyplot as plt

    fig, ax1 = plt.subplots(figsize=(11, 5))
    if train_steps:
        ax1.plot(train_steps, train_losses, label="train_loss", color="tab:blue", linewidth=1.2)
    if eval_steps:
        ax1.plot(eval_steps, eval_losses, label="eval_loss", color="tab:orange", linestyle="--", linewidth=1.5)
    ax1.set_xlabel("step")
    ax1.set_ylabel("loss")
    ax1.grid(alpha=0.3)

    ax2 = ax1.twinx()
    if eval_steps:
        ax2.plot(eval_steps, eval_f1s, label="eval_macro_f1", color="tab:green", marker="o", markersize=4)
    ax2.set_ylabel("eval_macro_f1")
    ax2.set_ylim(0, 1)

    ax1.set_title(f"{name} — train loss / eval loss / eval macro-F1")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--model", type=str, default=None)
    args = parser.parse_args()
    root = args.results_dir

    dirs = [root / args.model] if args.model else sorted(p for p in root.iterdir() if p.is_dir())

    for sub in dirs:
        if not sub.is_dir():
            continue
        for hist in sorted(sub.glob("*_history.jsonl")):
            tag = sub.name
            t_s, t_l, e_s, e_el, e_f1 = parse_history(hist)
            if not t_s and not e_s:
                print(f"Skip: {hist}")
                continue
            out_png = sub / f"{tag}_train_eval_curves.png"
            plot_one(tag, t_s, t_l, e_s, e_el, e_f1, out_png)


if __name__ == "__main__":
    main()
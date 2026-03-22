import argparse
import datetime
import json
import os
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)


class EmotionDataset(Dataset):
    def __init__(self, encodings, labels=None):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        if self.labels is not None:
            item["labels"] = torch.tensor(int(self.labels[idx]))
        return item

    def __len__(self):
        return len(next(iter(self.encodings.values())))


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    macro_f1 = f1_score(labels, preds, average="macro")
    acc = (preds == labels).mean()
    return {"macro_f1": macro_f1, "accuracy": acc}


def save_training_logs_and_curve(
    trainer_log_history: List[dict],
    logs_dir: Path,
    model_tag: str,
    valid_macro_f1: float,
) -> None:
    """
    将 Trainer 的日志历史保存为 jsonl，并输出训练曲线（loss vs step）。
    """
    logs_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # 1) 原始日志，便于后续复盘
    history_path = logs_dir / f"{model_tag}_{stamp}_history.jsonl"
    with history_path.open("w", encoding="utf-8") as f:
        for row in trainer_log_history:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    # 2) 训练曲线（优先 loss）
    loss_steps = []
    losses = []
    eval_steps = []
    eval_f1s = []
    for row in trainer_log_history:
        if "loss" in row and "step" in row:
            loss_steps.append(row["step"])
            losses.append(row["loss"])
        if "eval_macro_f1" in row and "step" in row:
            eval_steps.append(row["step"])
            eval_f1s.append(row["eval_macro_f1"])

    try:
        import matplotlib.pyplot as plt

        fig, ax1 = plt.subplots(figsize=(9, 5))
        if loss_steps:
            ax1.plot(loss_steps, losses, label="train_loss")
            ax1.set_xlabel("step")
            ax1.set_ylabel("loss")
            ax1.grid(alpha=0.3)

        # 若训练过程里有 eval_macro_f1，叠加第二坐标轴；没有则只标注最终 F1
        if eval_steps:
            ax2 = ax1.twinx()
            ax2.plot(eval_steps, eval_f1s, color="tab:orange", label="eval_macro_f1")
            ax2.set_ylabel("macro_f1")

        ax1.set_title(f"{model_tag} training curve | final_valid_macro_f1={valid_macro_f1:.4f}")
        curve_path = logs_dir / f"{model_tag}_{stamp}_curve.png"
        fig.tight_layout()
        fig.savefig(curve_path, dpi=150)
        plt.close(fig)
    except Exception as e:
        # 不阻塞训练流程；仅记录绘图失败信息
        err_path = logs_dir / f"{model_tag}_{stamp}_curve_error.txt"
        with err_path.open("w", encoding="utf-8") as f:
            f.write(str(e))


def train_single_model(
    model_name: str,
    data_dir: Path,
    run_root: Path,
    model_tag: str,
    logs_dir: Optional[Path] = None,
    num_labels: int = 7,
    num_epochs: int = 3,
    train_batch_size: int = 16,
    eval_batch_size: int = 64,
) -> Dict[str, Path]:
    """
    训练单个 Transformer 模型，并在 run_root/model_tag 下保存：
    - best model checkpoints（由 Trainer 管理）
    - valid_pred.csv
    - test_pred.csv
    - test_logits.npy （用于 ensemble）
    """
    run_dir = run_root / model_tag
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n==== 训练 {model_tag} ({model_name}) ====")
    print(f"数据目录: {data_dir}")
    print(f"输出目录: {run_dir}")

    train_df = pd.read_csv(data_dir / "train.csv")
    valid_df = pd.read_csv(data_dir / "valid.csv")
    test_df = pd.read_csv(data_dir / "test_no_label.csv")

    # DeBERTa-v3 的 fast tokenizer 与部分 transformers 版本不兼容，用 slow 避免 vocab_file None 报错
    if model_name == "microsoft/deberta-v3-base":
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name)

    train_encodings = tokenizer(
        train_df["text"].tolist(),
        truncation=True,
        padding=True,
        max_length=128,
    )
    valid_encodings = tokenizer(
        valid_df["text"].tolist(),
        truncation=True,
        padding=True,
        max_length=128,
    )
    test_encodings = tokenizer(
        test_df["text"].tolist(),
        truncation=True,
        padding=True,
        max_length=128,
    )

    train_dataset = EmotionDataset(train_encodings, train_df["label"].tolist())
    valid_dataset = EmotionDataset(valid_encodings, valid_df["label"].tolist())
    test_dataset = EmotionDataset(test_encodings, labels=None)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
    )

    # 只使用老版本 transformers 也支持的 TrainingArguments 参数，避免报
    # “got an unexpected keyword argument 'evaluation_strategy'” 之类错误。
    training_args = TrainingArguments(
        output_dir=str(run_dir / "checkpoints"),
        num_train_epochs=num_epochs,
        per_device_train_batch_size=train_batch_size,
        per_device_eval_batch_size=eval_batch_size,
        learning_rate=2e-5,
        weight_decay=0.01,
        logging_steps=50,
        save_strategy="steps",
        save_steps=2000,
        save_total_limit=5,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    # 验证集预测（用于和 evaluate_baseline_model.py 对齐）
    valid_outputs = trainer.predict(valid_dataset)
    valid_preds = np.argmax(valid_outputs.predictions, axis=-1)
    valid_macro_f1 = f1_score(valid_df["label"].values, valid_preds, average="macro")

    # 将当前模型在验证集上的 Macro-F1 记录到 log 文件中
    if logs_dir is not None:
        logs_dir.mkdir(parents=True, exist_ok=True)
        log_path = logs_dir / f"{model_tag}.log"
        with log_path.open("a", encoding="utf-8") as log_f:
            now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            log_f.write(
                f"[{now_str}] model={model_tag}, valid_macro_f1={valid_macro_f1:.6f}\n"
            )
        save_training_logs_and_curve(
            trainer_log_history=trainer.state.log_history,
            logs_dir=logs_dir,
            model_tag=model_tag,
            valid_macro_f1=valid_macro_f1,
        )

    valid_pred_path = run_dir / f"{model_tag}_valid_pred.csv"
    pd.DataFrame(
        {"id": valid_df["id"].values, "label": valid_preds},
    ).to_csv(valid_pred_path, index=False)

    # 测试集预测
    test_outputs = trainer.predict(test_dataset)
    test_logits = test_outputs.predictions
    test_preds = np.argmax(test_logits, axis=-1)
    test_pred_path = run_dir / f"{model_tag}_test_pred.csv"
    pd.DataFrame(
        {"id": test_df["id"].values, "label": test_preds},
    ).to_csv(test_pred_path, index=False)

    # 保存 logits 供 ensemble 使用
    logits_path = run_dir / f"{model_tag}_test_logits.npy"
    np.save(logits_path, test_logits)

    # 记录 best dev macro-f1
    metrics_path = run_dir / "metrics.txt"
    with metrics_path.open("w", encoding="utf-8") as f:
        f.write(f"valid_macro_f1={valid_macro_f1:.6f}\n")

    print(f"{model_tag} 训练完成，验证集 Macro-F1 = {valid_macro_f1:.4f}")
    print(f"验证集预测保存到: {valid_pred_path}")
    print(f"测试集预测保存到: {test_pred_path}")

    return {
        "run_dir": run_dir,
        "valid_pred": valid_pred_path,
        "test_pred": test_pred_path,
        "logits": logits_path,
    }


def ensemble_by_logits(
    test_ids: List[int],
    logits_paths: List[Path],
    out_path: Path,
):
    """
    根据多个模型的 test logits 做概率平均 ensemble。
    """
    all_logits = [np.load(p) for p in logits_paths]
    stacked = np.stack(all_logits, axis=0)  # (M, N, C)
    mean_logits = stacked.mean(axis=0)  # (N, C)
    final_preds = np.argmax(mean_logits, axis=-1)

    pd.DataFrame({"id": test_ids, "label": final_preds}).to_csv(out_path, index=False)
    print(f"集成模型测试集预测已保存到: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=str,
        default=os.path.join("..", "Processed_data"),
        help="预处理后的数据目录（默认使用 SiruiChen/Processed_data）",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default=os.path.join("..", "training_output"),
        help="所有训练运行结果的根目录",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="每个模型的训练 epoch 数",
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=16,
        help="训练时的 per-device batch size",
    )
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=64,
        help="验证/测试时的 per-device batch size",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_root = output_root / timestamp
    run_root.mkdir(parents=True, exist_ok=True)

    # 本次运行的 log 目录，用于记录各模型的验证集 F1
    logs_dir = run_root / "logs"

    print(f"本次运行输出目录: {run_root}")

    # 训练三个 Transformer 模型
    model_configs = [
        ("bert-base-uncased", "bert_base_uncased"),
        ("roberta-base", "roberta_base"),
        ("microsoft/deberta-v3-base", "deberta_v3_base"),
    ]

    model_artifacts: Dict[str, Dict[str, Path]] = {}
    for model_name, tag in model_configs:
        artifacts = train_single_model(
            model_name=model_name,
            data_dir=data_dir,
            run_root=run_root,
            model_tag=tag,
            logs_dir=logs_dir,
            num_labels=7,
            num_epochs=args.epochs,
            train_batch_size=args.train_batch_size,
            eval_batch_size=args.eval_batch_size,
        )
        model_artifacts[tag] = artifacts

    # 使用 logits 做 ensemble（概率平均）
    test_df = pd.read_csv(data_dir / "test_no_label.csv")
    ensemble_out_path = run_root / "ensemble_test_pred.csv"
    ensemble_by_logits(
        test_ids=test_df["id"].values,
        logits_paths=[model_artifacts[tag]["logits"] for _, tag in model_configs],
        out_path=ensemble_out_path,
    )

    print("\n全部模型训练 + 集成完成。")
    print(f"你可以使用 evaluate_baseline_model.py 对某个模型或 ensemble 的验证集预测做评估，示例：")
    print(
        f"  python evaluate_baseline_model.py --pred {run_root / 'bert_base_uncased' / 'bert_base_uncased_valid_pred.csv'}"
    )


if __name__ == "__main__":
    main()


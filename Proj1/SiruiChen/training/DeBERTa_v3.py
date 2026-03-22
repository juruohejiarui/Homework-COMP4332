import argparse
import datetime
from pathlib import Path

import numpy as np
import pandas as pd
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
        item = {key: np.array(val[idx]) for key, val in self.encodings.items()}
        if self.labels is not None:
            item["labels"] = int(self.labels[idx])
        return item

    def __len__(self):
        return len(next(iter(self.encodings.values())))


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    macro_f1 = f1_score(labels, preds, average="macro")
    acc = (preds == labels).mean()
    return {"macro_f1": macro_f1, "accuracy": acc}


def main():
    parser = argparse.ArgumentParser(description="Train DeBERTa-v3 for emotion classification.")
    parser.add_argument("--data_dir", type=str, default="../Processed_data")
    parser.add_argument("--output_root", type=str, default="../training_output")
    parser.add_argument("--model_name", type=str, default="microsoft/deberta-v3-base")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--train_batch_size", type=int, default=32)
    parser.add_argument("--eval_batch_size", type=int, default=128)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--tb_logdir", type=str, default="../training/logs/tb/deberta_v3")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_root / f"deberta_v3_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"==== 训练 {args.model_name} ====")
    print(f"数据目录: {data_dir}")
    print(f"输出目录: {run_dir}")

    train_df = pd.read_csv(data_dir / "train.csv")
    valid_df = pd.read_csv(data_dir / "valid.csv")
    test_df = pd.read_csv(data_dir / "test_no_label.csv")

    # 避免 DeBERTa-v3 fast tokenizer 在部分 transformers 版本下报错
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=False)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=7,
    )

    train_enc = tokenizer(
        train_df["text"].tolist(),
        truncation=True,
        padding=True,
        max_length=args.max_length,
    )
    valid_enc = tokenizer(
        valid_df["text"].tolist(),
        truncation=True,
        padding=True,
        max_length=args.max_length,
    )
    test_enc = tokenizer(
        test_df["text"].tolist(),
        truncation=True,
        padding=True,
        max_length=args.max_length,
    )

    train_ds = EmotionDataset(train_enc, train_df["label"].tolist())
    valid_ds = EmotionDataset(valid_enc, valid_df["label"].tolist())
    test_ds = EmotionDataset(test_enc, labels=None)

    training_args = TrainingArguments(
        output_dir=str(run_dir / "checkpoints"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        learning_rate=2e-5,
        weight_decay=0.01,
        logging_steps=50,
        save_strategy="steps",
        save_steps=2000,
        save_total_limit=5,
        logging_dir=str(Path(args.tb_logdir)),
        report_to=["tensorboard"],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=valid_ds,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    valid_outputs = trainer.predict(valid_ds)
    valid_preds = np.argmax(valid_outputs.predictions, axis=-1)
    valid_macro_f1 = f1_score(valid_df["label"].values, valid_preds, average="macro")

    valid_pred_path = run_dir / "deberta_v3_valid_pred.csv"
    pd.DataFrame({"id": valid_df["id"].values, "label": valid_preds}).to_csv(valid_pred_path, index=False)

    test_outputs = trainer.predict(test_ds)
    test_preds = np.argmax(test_outputs.predictions, axis=-1)
    test_pred_path = run_dir / "deberta_v3_test_pred.csv"
    pd.DataFrame({"id": test_df["id"].values, "label": test_preds}).to_csv(test_pred_path, index=False)

    metrics_path = run_dir / "metrics.txt"
    with metrics_path.open("w", encoding="utf-8") as f:
        f.write(f"model={args.model_name}\n")
        f.write(f"valid_macro_f1={valid_macro_f1:.6f}\n")

    print(f"训练完成，验证集 Macro-F1 = {valid_macro_f1:.4f}")
    print(f"验证集预测保存到: {valid_pred_path}")
    print(f"测试集预测保存到: {test_pred_path}")
    print(f"指标保存到: {metrics_path}")
    print(f"TensorBoard 日志目录: {Path(args.tb_logdir)}")


if __name__ == "__main__":
    main()
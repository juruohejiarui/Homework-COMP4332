import pandas as pd
import numpy as np
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding
)
from sklearn.metrics import f1_score, accuracy_score

# ── Hyper-parameters ──────────────────────────────────────────────────────────
MODEL_NAME  = "roberta-large"  # 如果显存或时间吃紧，可换成 "roberta-base"
NUM_CLASSES = 7
MAX_LEN     = 128              # RoBERTa 最大支持 512，通常情感分类 128 足够
EPOCHS      = 5                # 预训练模型只需几个 epoch 即可收敛
BATCH_SIZE  = 32               # 40GB 显存可轻松应对 roberta-large + bs=32
LR          = 2e-5             # 微调预训练模型的学习率通常很小 (1e-5 ~ 5e-5)
SEED        = 42
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(eval_pred):
    """自定义评估指标，核心关注 Macro-F1"""
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    
    macro_f1 = f1_score(labels, predictions, average="macro", zero_division=0)
    acc = accuracy_score(labels, predictions)
    
    return {"macro_f1": macro_f1, "accuracy": acc}

def main():
    print(f"Loading {MODEL_NAME} tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # 1. 加载数据
    train_df = pd.read_csv("data/train.csv")
    valid_df = pd.read_csv("data/valid.csv")
    test_df  = pd.read_csv("data/test_no_label.csv")

    # 转换为 Hugging Face Dataset 格式
    train_ds = Dataset.from_pandas(train_df[["text", "label"]])
    valid_ds = Dataset.from_pandas(valid_df[["text", "label"]])
    test_ds  = Dataset.from_pandas(test_df[["id", "text"]])

    # 2. Tokenize 处理函数
    def tokenize_function(examples):
        return tokenizer(
            examples["text"], 
            truncation=True, 
            max_length=MAX_LEN
        )

    print("Tokenizing datasets...")
    train_tokenized = train_ds.map(tokenize_function, batched=True)
    valid_tokenized = valid_ds.map(tokenize_function, batched=True)
    test_tokenized  = test_ds.map(tokenize_function, batched=True)

    # 动态 Padding
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # 3. 加载预训练模型
    print(f"Loading {MODEL_NAME} model for sequence classification...")
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, 
        num_labels=NUM_CLASSES
    )

    # 4. 配置训练参数
    training_args = TrainingArguments(
        output_dir="./results",
        eval_strategy="epoch",            # 每个 epoch 结束后验证
        save_strategy="epoch",
        save_total_limit=2,               # 只保留最后2个 checkpoint
        learning_rate=LR,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE * 2, # 推理时不需要梯度，可加大 batch size
        num_train_epochs=EPOCHS,
        weight_decay=0.01,
        fp16=True,                        # 开启混合精度，加速训练并节省显存
        load_best_model_at_end=True,      # 训练结束后自动加载最好的模型
        metric_for_best_model="macro_f1", # 根据 macro-f1 挑选最佳模型
        seed=SEED,
        logging_steps=50,
        report_to="none"                  # 禁用 wandb 等日志上报
    )

    # 5. 初始化 Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tokenized,
        eval_dataset=valid_tokenized,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # 6. 开始训练
    print("Starting training...")
    trainer.train()

    # 打印验证集上的最佳结果
    eval_results = trainer.evaluate()
    print(f"\nBest Validation Macro-F1: {eval_results['eval_macro_f1']:.4f}")

    # 7. 在测试集上进行预测
    print("\nGenerating predictions on test set...")
    predictions = trainer.predict(test_tokenized)
    preds = np.argmax(predictions.predictions, axis=-1)

    # 保存预测结果
    out = pd.DataFrame({"id": test_df["id"], "label": preds})
    out.to_csv("roberta_pred.csv", index=False)
    print("Saved roberta_pred.csv")

if __name__ == "__main__":
    main()
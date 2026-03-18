import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from datasets import Dataset
from typing import Self
from transformers import (
    AutoTokenizer,
    AutoModel,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)
from safetensors.torch import save_model
from peft import LoraConfig, TaskType, get_peft_model
from sklearn.metrics import f1_score, accuracy_score

# -------------------- 1. Focal Loss 实现 --------------------
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None, reduction='mean'):
        """
        gamma: 聚焦参数，减少易分类样本的权重
        alpha: 类别权重（tensor），若为None则使用均匀权重
        """
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, logits, targets):
        """
        logits: [batch_size, num_classes]
        targets: [batch_size]
        """
        log_probs = F.log_softmax(logits, dim=-1)          # log(softmax)
        probs = torch.exp(log_probs)                       # softmax概率
        # 提取每个样本正确类别的log_prob和prob
        log_pt = log_probs.gather(1, targets.view(-1, 1)).squeeze()
        pt = probs.gather(1, targets.view(-1, 1)).squeeze()

        focal_weight = (1 - pt) ** self.gamma
        if self.alpha is not None:
            alpha_t = self.alpha.gather(0, targets)
            focal_weight = alpha_t * focal_weight

        loss = -focal_weight * log_pt

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

# -------------------- 2. 自定义分类模型（嵌入模型 + 分类头） --------------------
class EmbeddingForClassification(nn.Module):
    def __init__(self, base_model, num_labels=7, embedding_dim=2560, focal_loss_gamma=2.0, class_weights=None):
        super().__init__()
        self.base_model = base_model
        self.num_labels = num_labels
        self.embedding_dim = embedding_dim
        self.classifier = nn.Linear(embedding_dim, num_labels)
        self.focal_loss = FocalLoss(gamma=focal_loss_gamma, alpha=class_weights)

    def forward(self, input_ids, attention_mask, labels=None):
        # 获取嵌入表示
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        embeddings = outputs.last_hidden_state[:, -1, :]   # [batch, embedding_dim]
        logits = self.classifier(embeddings)

        loss = None
        if labels is not None:
            loss = self.focal_loss(logits, labels)

        return {"loss": loss, "logits": logits}   # Trainer 需要这种格式
    
    def train(self) :
        self.base_model.train()
        self.classifier.train()
    
    def eval(self) :
        self.base_model.eval()
        self.classifier.eval()

    def load(path : str) -> Self :
        with open(f"{path}/head_cfg.json", "r") as f:
            config = json.load(f)
            num_labels = config["num_labels"]
            embedding_dim = config["embedding_dim"]
        model = EmbeddingForClassification(
            AutoModel.from_pretrained(
                path, 
                device_map="auto",
                dtype=torch.bfloat16),
            num_labels=num_labels,
            embedding_dim=embedding_dim
        )
        model.classifier.load_state_dict(torch.load(f"{path}/classifier_head.pth"))
        model.classifier.to(dtype=torch.bfloat16)
        return model

# -------------------- 3. 数据加载与预处理 --------------------
def load_and_filter_data(train_path, valid_path, label_range=(0,6)):
    train_df = pd.read_csv(train_path)
    valid_df = pd.read_csv(valid_path)

    # 只保留指定范围内的标签
    train_df = train_df[train_df['label'].between(*label_range)]
    valid_df = valid_df[valid_df['label'].between(*label_range)]
    train_df['label'] = train_df['label'].astype(int)
    valid_df['label'] = valid_df['label'].astype(int)

    print("训练集标签分布:\n", train_df['label'].value_counts().sort_index())
    print("验证集标签分布:\n", valid_df['label'].value_counts().sort_index())
    return train_df, valid_df

# -------------------- 4. 主训练流程 --------------------
def main():
    # 配置参数
    model_name = "/home/hjr/Documents/Resources/Models/Embed/qwen3-embed-4b"   # 替换为实际路径
    train_csv = "../data/train.csv"
    valid_csv = "../data/valid.csv"
    num_labels = 7
    max_length = 128
    batch_size = 16
    epochs = 3
    lr = 1e-4
    gamma = 2.0                         # Focal Loss gamma
    output_dir = "./embed-qwen3-4b"

    # 加载数据
    train_df, valid_df = load_and_filter_data(train_csv, valid_csv)

    # 转为 Hugging Face Dataset
    train_dataset = Dataset.from_pandas(train_df[['text', 'label']])
    valid_dataset = Dataset.from_pandas(valid_df[['text', 'label']])

    # 加载 tokenizer 并设置 pad_token
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token   # 嵌入模型也需要 pad_token

    # Tokenize（只截断，不填充）
    def tokenize_function(examples):
        # Qwen 嵌入模型推荐添加指令前缀，可根据需要调整
        # texts = ["为情感分类任务生成向量: " + text for text in examples["text"]]
        texts = examples["text"]   # 暂时不加前缀
        return tokenizer(texts, truncation=True, max_length=max_length)

    tokenized_train = train_dataset.map(tokenize_function, batched=True, remove_columns=['text'])
    tokenized_valid = valid_dataset.map(tokenize_function, batched=True, remove_columns=['text'])

    # 数据整理器（动态 padding）
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # 加载基础嵌入模型
    base_model = AutoModel.from_pretrained(
        model_name,
        device_map="auto",
        dtype=torch.bfloat16,
        # 如果显存不足，可开启 4-bit 量化（需 bitsandbytes）
        # load_in_4bit=True,
    )

    # 计算类别权重 alpha（基于训练集频率）
    labels_train = train_df['label'].values
    class_counts = np.bincount(labels_train, minlength=num_labels)
    total = len(labels_train)
    alpha = total / (num_labels * (class_counts + 1e-6))   # 逆频率
    alpha = torch.tensor(alpha, dtype=torch.float)

    # 构建自定义分类模型
    model = EmbeddingForClassification(
        base_model,
        num_labels=num_labels,
        embedding_dim=base_model.config.hidden_size,  # Qwen3-Embedding-4B 通常是 2560
        focal_loss_gamma=gamma,
        class_weights=alpha.to(base_model.device) if torch.cuda.is_available() else alpha
    )

    # 应用 LoRA（仅对 base_model 部分）
    lora_config = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,   # 嵌入模型使用 FEATURE_EXTRACTION
        r=8,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # Qwen 常见模块名
        lora_dropout=0.1,
        bias="none",
    )
    model.base_model = get_peft_model(model.base_model, lora_config)
    trainable_params = [n for n, p in model.named_parameters() if p.requires_grad]
    print(f"Trainable parameters: {len(trainable_params)}")
    print(trainable_params[:10])  # 打印前几个

    # 评价指标
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        preds = np.argmax(predictions, axis=1)
        macro_f1 = f1_score(labels, preds, average='macro', zero_division=0)
        accuracy = accuracy_score(labels, preds)
        return {"macro_f1": macro_f1, "accuracy": accuracy}

    # 训练参数
    training_args = TrainingArguments(
        output_dir=output_dir,

        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size * 4,

        learning_rate=lr,
        num_train_epochs=epochs,
        lr_scheduler_type='cosine',
        warmup_steps=100,
        weight_decay=0.1,

        logging_steps=25,
        eval_strategy="steps",
        eval_steps=200,
        save_strategy="steps",
        save_steps=200,
        save_total_limit=2,
        load_best_model_at_end=True,

        metric_for_best_model="macro_f1",
        bf16=True,
        remove_unused_columns=False,

        report_to=["tensorboard"],                # 关键：指定使用TensorBoard
        logging_dir="./logs",                     # TensorBoard日志目录
        run_name="embed-qwen3-4b" # 实验名称
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_valid,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # 启动训练
    trainer.train()

    # 保存最终模型（分类头和 LoRA 权重）
    output_dir = "./embed-qwen3-4b/merged"
    model.base_model = model.base_model.merge_and_unload()
    model.base_model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    torch.save(model.classifier.state_dict(), os.path.join(output_dir, "classifier_head.pth"))
    head_cfg = {
        "num_labels": model.num_labels,
        "embedding_dim": model.embedding_dim,
        "focal_loss_gamma": 2.0,  # 可根据需要保存
    }
    import json
    with open(os.path.join(output_dir, "head_cfg.json"), "w") as f:
        json.dump(head_cfg, f)
    

if __name__ == "__main__":
    main()
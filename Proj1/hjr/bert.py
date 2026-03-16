import os
import sys
import pandas as pd
import numpy as np
import torch
from torch.optim import AdamW
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_cosine_schedule_with_warmup
from sklearn.metrics import f1_score, accuracy_score
from tqdm import tqdm, trange
import argparse
import logging
import datetime
import csv

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

his_path = 'his.csv'

# -------------------- 1. Focal Loss 实现 --------------------
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None, reduction='mean'):
        """
        Focal Loss for multi-class classification.
        :param gamma: 聚焦参数，减少易分类样本的权重
        :param alpha: 类别权重（tensor），若为None则使用均匀权重
        :param reduction: 'mean' 或 'sum'
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
        # 提取每个样本对应正确类别的log_prob和prob
        log_pt = log_probs.gather(1, targets.view(-1, 1)).squeeze()
        pt = probs.gather(1, targets.view(-1, 1)).squeeze()

        # 计算focal loss: -alpha * (1-pt)^gamma * log(pt)
        focal_weight = (1 - pt) ** self.gamma
        if self.alpha is not None:
            # alpha需要是类别数长度的tensor，按targets索引
            alpha_t = self.alpha.gather(0, targets)
            focal_weight = alpha_t * focal_weight

        loss = -focal_weight * log_pt

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

# -------------------- 2. 数据集类 --------------------
class GoEmotionsDataset(Dataset):
    def __init__(self, csv_path, tokenizer, max_length=128):
        self.df = pd.read_csv(csv_path)
        # 确保标签为整数
        self.df['label'] = self.df['label'].astype(int)
        self.texts = self.df['text'].tolist()
        self.labels = self.df['label'].tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        # 去掉batch维度
        input_ids = encoding['input_ids'].squeeze(0)
        attention_mask = encoding['attention_mask'].squeeze(0)

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': torch.tensor(label, dtype=torch.long)
        }

# -------------------- 3. 训练函数 --------------------
def train_epoch(model, dataloader, loss_fn, optimizer, scheduler, device):
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []

    for batch in tqdm(dataloader, dynamic_ncols=True):
        # 将数据移至设备
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        # 前向传播
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits

        # 计算损失
        loss = loss_fn(logits, labels)
        total_loss += loss.item()

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        # 收集预测结果用于指标计算
        preds = torch.argmax(logits, dim=-1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        # progress_bar.set_postfix({'loss': f"{loss.item():.3f}"})

    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    return avg_loss, accuracy, macro_f1

def eval_epoch(model, dataloader, loss_fn, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(dataloader, dynamic_ncols=True):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            loss = loss_fn(logits, labels)
            total_loss += loss.item()

            preds = torch.argmax(logits, dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    return avg_loss, accuracy, macro_f1

# -------------------- 4. 主程序 --------------------
def main(args):
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # 准备数据集
    train_dataset = GoEmotionsDataset(args.train_csv, tokenizer, max_length=args.max_length)
    valid_dataset = GoEmotionsDataset(args.valid_csv, tokenizer, max_length=args.max_length)

    # 计算类别权重alpha（可选，用于Focal Loss）
    # 统计训练集中各类别的样本数，然后计算权重（如逆频率）
    labels_train = train_dataset.labels
    class_counts = np.bincount(labels_train, minlength=args.num_classes)
    total = len(labels_train)
    # 权重 = 总样本数 / (类别数 * 该类样本数)，避免除零
    alpha = total / (args.num_classes * (class_counts + 1e-6))
    alpha = torch.tensor(alpha, dtype=torch.float).to(device)
    print(f'Class alpha weights: {alpha}')

    print(f'batch size: {args.batch_size}')

    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    # 加载模型
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=args.num_classes
    ).to(device)

    # 定义损失函数（使用Focal Loss，传入alpha）
    loss_fn = FocalLoss(gamma=args.gamma, alpha=alpha, reduction='mean')

    # 优化器
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # 学习率调度器（线性warmup）
    total_steps = len(train_loader) * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )

    # 训练与验证
    best_macro_f1 = 0.0
    best_model_path = args.save_path

    sys.stdout.flush()

    for epoch in range(1, args.epochs + 1):
        print(f'Epoch {epoch}/{args.epochs}')
        train_loss, train_acc, train_f1 = train_epoch(model, train_loader, loss_fn, optimizer, scheduler, device)
        print(f'Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, Macro F1: {train_f1:.4f}')

        val_loss, val_acc, val_f1 = eval_epoch(model, valid_loader, loss_fn, device)
        print(f'Valid - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, Macro F1: {val_f1:.4f}')

        # 保存最佳模型（基于验证集Macro F1）
        if val_f1 > best_macro_f1:
            best_macro_f1 = val_f1
            torch.save(model.state_dict(), best_model_path)
            print(f'New best model saved with Macro F1: {best_macro_f1:.4f}')

    print(f'Training completed. Best Macro F1: {best_macro_f1:.4f}')

    _, val_acc, val_f1 = eval_epoch(model, valid_loader, loss_fn, device)

    with open(his_path, mode='a', newline='', encoding='utf-8') as f :
        writer = csv.writer(f)
        writer.writerow((datetime.datetime.now(), val_f1, val_acc))

# -------------------- 5. 参数解析 --------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_csv', type=str, required=True, help='Path to training CSV')
    parser.add_argument('--valid_csv', type=str, required=True, help='Path to validation CSV')
    parser.add_argument('--model_name', type=str, default='bert-base-uncased', help='Pretrained model name')
    parser.add_argument('--num_classes', type=int, default=7, help='Number of emotion classes')
    parser.add_argument('--max_length', type=int, default=128, help='Max token length')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--epochs', type=int, default=3, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=2e-5, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay')
    parser.add_argument('--warmup_ratio', type=float, default=0.1, help='Warmup proportion')
    parser.add_argument('--gamma', type=float, default=2.0, help='Focal Loss gamma')
    parser.add_argument('--save_path', type=str, default='best_model.pt', help='Path to save best model')
    args = parser.parse_args()

    main(args)
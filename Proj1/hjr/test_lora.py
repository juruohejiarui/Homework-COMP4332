import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# 自定义数据集类，用于批量编码文本
class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=128):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(0),  # 去除 batch 维度
            'attention_mask': encoding['attention_mask'].squeeze(0)
        }

def predict(model, dataloader, device):
    """批量预测函数，返回预测标签列表"""
    model.eval()
    predictions = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Predicting"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=-1).cpu().numpy()
            predictions.extend(preds)
    return predictions

def main():
    # 配置路径
    model_path = "lora-qwen3-4b/merged"          # 合并后模型的保存路径
    valid_path = "../data/valid.csv"             # 验证集路径
    test_path = "../data/test_no_label.csv"      # 测试集路径
    valid_out = "pred_valid_new.csv"                  # 验证集输出文件
    test_out = "pred_test_new.csv"                    # 测试集输出文件

    # 设备设置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 加载 tokenizer 和模型（合并后的完整模型）
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None
    ).to(device)

    # ---------- 验证集预测 ----------
    print("Processing validation set...")
    valid_df = pd.read_csv(valid_path)
    valid_texts = valid_df['text'].tolist()

    valid_dataset = TextDataset(valid_texts, tokenizer)
    valid_loader = DataLoader(valid_dataset, batch_size=128, shuffle=False)
    valid_preds = predict(model, valid_loader, device)

    # 将预测结果添加到原 DataFrame 并保存
    valid_df['label'] = valid_preds
    valid_df = valid_df.drop(columns=['text'])
    valid_df.to_csv(valid_out, index=False)
    print(f"Saved {valid_out}")

    # ---------- 测试集预测 ----------
    print("Processing test set...")
    test_df = pd.read_csv(test_path)
    test_texts = test_df['text'].tolist()

    test_dataset = TextDataset(test_texts, tokenizer)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
    test_preds = predict(model, test_loader, device)

    # 生成只包含 id 和 label 的输出文件
    test_output = pd.DataFrame({
        'id': test_df['id'],
        'label': test_preds
    })
    test_output.to_csv(test_out, index=False)
    print(f"Saved {test_out}")

if __name__ == "__main__":
    main()
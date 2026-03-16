from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
import torch
import pandas as pd
from sklearn.metrics import f1_score
import numpy as np

class BertEmotionClassifier:
    def __init__(self, num_labels=7):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertForSequenceClassification.from_pretrained(
            'bert-base-uncased', 
            num_labels=num_labels
        )
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
    
    def train(self, train_texts, train_labels, valid_texts, valid_labels, epochs=5):
        # 编码数据
        train_encodings = self.tokenizer(train_texts, truncation=True, padding=True, max_length=128)
        valid_encodings = self.tokenizer(valid_texts, truncation=True, padding=True, max_length=128)
        
        # 创建数据集
        class EmotionDataset(torch.utils.data.Dataset):
            def __init__(self, encodings, labels):
                self.encodings = encodings
                self.labels = labels
            def __getitem__(self, idx):
                item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
                item['labels'] = torch.tensor(self.labels[idx])
                return item
            def __len__(self):
                return len(self.labels)
        
        train_dataset = EmotionDataset(train_encodings, train_labels)
        valid_dataset = EmotionDataset(valid_encodings, valid_labels)
        
        # 训练参数
        training_args = TrainingArguments(
            output_dir='./results',
            num_train_epochs=epochs,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=64,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir='./logs',
            logging_steps=10,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            greater_is_better=True
        )
        
        # 自定义评估指标
        def compute_metrics(eval_pred):
            predictions, labels = eval_pred
            predictions = np.argmax(predictions, axis=1)
            return {
                'f1': f1_score(labels, predictions, average='macro'),
                'accuracy': (predictions == labels).mean()
            }
        
        # 训练
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=valid_dataset,
            compute_metrics=compute_metrics
        )
        
        trainer.train()
        return trainer
    
    def predict(self, texts):
        self.model.eval()
        predictions = []
        with torch.no_grad():
            for i in range(0, len(texts), 32):
                batch_texts = texts[i:i+32]
                inputs = self.tokenizer(batch_texts, return_tensors="pt", 
                                       truncation=True, padding=True, max_length=128)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                outputs = self.model(**inputs)
                batch_preds = torch.argmax(outputs.logits, dim=-1)
                predictions.extend(batch_preds.cpu().numpy())
        return predictions

# 使用示例
train_df = pd.read_csv('data/train.csv')
valid_df = pd.read_csv('data/valid.csv')

classifier = BertEmotionClassifier()
classifier.train(
    train_df['text'].tolist(), 
    train_df['label'].tolist(),
    valid_df['text'].tolist(), 
    valid_df['label'].tolist()
)

# 预测测试集
test_df = pd.read_csv('data/test_no_label.csv')
predictions = classifier.predict(test_df['text'].tolist())

# 保存结果
submission = pd.DataFrame({
    'id': test_df['id'],
    'label': predictions
})
submission.to_csv('bert_pred.csv', index=False)
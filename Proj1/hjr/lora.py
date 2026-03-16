# 环境准备 (需要安装较新的库)
# pip install torch transformers accelerate peft bitsandbytes datasets scikit-learn

import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    TrainingArguments, 
    Trainer, 
    DataCollatorWithPadding
)
from peft import LoraConfig, TaskType, get_peft_model
from datasets import Dataset
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score
import numpy as np

# 1. 加载数据和tokenizer (数据格式与之前相同)
train_df = pd.read_csv('../data/train.csv')
valid_df = pd.read_csv('../data/valid.csv')
# ... (数据预处理，将label列转为整数) ...

train_dataset = Dataset.from_pandas(train_df[['text', 'label']])
valid_dataset = Dataset.from_pandas(valid_df[['text', 'label']])

model_name = "mistralai/Mistral-7B-v0.1"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token  # Mistral 需要设置 pad_token

def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, max_length=128, padding="max_length")

tokenized_train = train_dataset.map(tokenize_function, batched=True)
tokenized_valid = valid_dataset.map(tokenize_function, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# 2. 加载基础模型并配置 LoRA
base_model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=7,  # GoEmotions 的类别数
    device_map="auto",  # 自动利用多GPU
    # load_in_4bit=True,  # 4-bit 量化，大幅降低显存
    torch_dtype=torch.bfloat16,
)

# LoRA 配置
lora_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    r=8,                # 低秩矩阵的维度
    lora_alpha=32,      # 缩放参数
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],  # 常见设置
    lora_dropout=0.1,
    bias="none",
)

model = get_peft_model(base_model, lora_config)
model.print_trainable_parameters()  # 你会看到可训练参数占比极低

# 3. 定义评价函数 (Macro F1)
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    macro_f1 = f1_score(labels, predictions, average='macro', zero_division=0)
    accuracy = accuracy_score(labels, predictions)
    return {"macro_f1": macro_f1, "accuracy": accuracy}

# 4. 设置训练参数
training_args = TrainingArguments(
    output_dir="./mistral-goemotions-lora",
    per_device_train_batch_size=8,   # 4-bit 量化下，8的batch在48G显存内很安全
    per_device_eval_batch_size=16,
    gradient_accumulation_steps=2,    # 模拟更大batch
    learning_rate=2e-4,               # LoRA 通常需要比全参数微调更高的LR
    num_train_epochs=5,
    logging_steps=50,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="macro_f1",
    fp16=True,                         # 混合精度训练
    remove_unused_columns=False,
)

# 5. 开始训练
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_valid,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()

# 6. 保存最终模型
model.save_pretrained("./final-mistral-lora")
tokenizer.save_pretrained("./final-mistral-lora")
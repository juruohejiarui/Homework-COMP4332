# 环境准备 (需要安装较新的库)
# pip install torch transformers accelerate peft bitsandbytes datasets scikit-learn

import torch
from transformers import (
    Qwen3_5Tokenizer,
    AutoTokenizer, 
    Qwen3_5ForSequenceClassification,
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
import torch.nn as nn
import tensorboard

# 1. 加载数据和tokenizer (数据格式与之前相同)
train_df = pd.read_csv('../data/train.csv')
valid_df = pd.read_csv('../data/valid.csv')

train_dataset = Dataset.from_pandas(train_df[['text', 'label']])
valid_dataset = Dataset.from_pandas(valid_df[['text', 'label']])

model_name = "/home/hjr/Documents/Resources/Models/Chat/qwen3.5-4b"
tokenizer = Qwen3_5Tokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token  # Mistral 需要设置 pad_token

def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, max_length=128)

tokenized_train = train_dataset.map(tokenize_function, batched=True, remove_columns=['text'])
tokenized_valid = valid_dataset.map(tokenize_function, batched=True, remove_columns=['text'])
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# 2. 加载基础模型并配置 LoRA
base_model = Qwen3_5ForSequenceClassification.from_pretrained(
    model_name,
    device_map="auto",  # 自动利用多GPU
    # load_in_4bit=True,  # 4-bit 量化，大幅降低显存
    dtype=torch.bfloat16,
    trust_remote_code=True
)

# print(base_model)

# replace (score)
base_model.score = nn.Linear(base_model.score.in_features, 7)
base_model.score.requires_grad_()
base_model.config.num_labels = 7

print(base_model)

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
    output_dir="./lora-qwen3-4b",
    per_device_train_batch_size=16,   # 4-bit 量化下，8的batch在48G显存内很安全
    per_device_eval_batch_size=64,
    learning_rate=1.5e-4,               # LoRA 通常需要比全参数微调更高的LR
    num_train_epochs=3,

    lr_scheduler_type='cosine',
    warmup_steps=100,
    weight_decay=0.1,

    logging_steps=25,

    eval_strategy="steps",
    eval_steps=150,

    save_strategy="no",
    # save_steps=450,
    # save_total_limit=2,

    # load_best_model_at_end=True,
    metric_for_best_model="macro_f1",

    bf16=True,                         # 混合精度训练

    remove_unused_columns=False,
    report_to=["tensorboard"],                # 关键：指定使用TensorBoard
    logging_dir="./logs",                     # TensorBoard日志目录
    run_name="lora-qwen3-4b" # 实验名称
)

# 5. 开始训练
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_valid,
    processing_class=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()

# 6. 保存最终模型
output_path = "./lora-qwen3-4b/merged"
merged_model = model.merge_and_unload()
merged_model.save_pretrained(output_path)
tokenizer.save_pretrained(output_path)
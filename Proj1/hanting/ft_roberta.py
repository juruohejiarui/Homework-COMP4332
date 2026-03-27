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
MODEL_NAME  = "roberta-large"  
NUM_CLASSES = 7
MAX_LEN     = 128             
EPOCHS      = 5                
BATCH_SIZE  = 32               
LR          = 2e-5             
SEED        = 42
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    
    macro_f1 = f1_score(labels, predictions, average="macro", zero_division=0)
    acc = accuracy_score(labels, predictions)
    
    return {"macro_f1": macro_f1, "accuracy": acc}

def main():
    print(f"Loading {MODEL_NAME} tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    train_df = pd.read_csv("data/train.csv")
    valid_df = pd.read_csv("data/valid.csv")
    test_df  = pd.read_csv("data/test_no_label.csv")
    train_ds = Dataset.from_pandas(train_df[["text", "label"]])
    valid_ds = Dataset.from_pandas(valid_df[["text", "label"]])
    test_ds  = Dataset.from_pandas(test_df[["id", "text"]])

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

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    print(f"Loading {MODEL_NAME} model for sequence classification...")
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, 
        num_labels=NUM_CLASSES
    )

    training_args = TrainingArguments(
        output_dir="./results",
        eval_strategy="epoch",           
        save_strategy="epoch",
        save_total_limit=2,              
        learning_rate=LR,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE*2,
        num_train_epochs=EPOCHS,
        weight_decay=0.01,
        fp16=True,                        
        load_best_model_at_end=True,     
        metric_for_best_model="macro_f1", 
        seed=SEED,
        logging_steps=50,
        report_to="none"                
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tokenized,
        eval_dataset=valid_tokenized,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    print("Starting training...")
    trainer.train()
    
    eval_results = trainer.evaluate()
    print(f"\nBest Validation Macro-F1: {eval_results['eval_macro_f1']:.4f}")

    print("\nGenerating predictions on test set...")
    predictions = trainer.predict(test_tokenized)
    preds = np.argmax(predictions.predictions, axis=-1)

    out = pd.DataFrame({"id": test_df["id"], "label": preds})
    out.to_csv("roberta_pred.csv", index=False)
    print("Saved roberta_pred.csv")

if __name__ == "__main__":
    main()

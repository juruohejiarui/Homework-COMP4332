import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import re
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import f1_score
from collections import Counter
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import time
import matplotlib.pyplot as plt
from collections import defaultdict
import json

# ── Hyper-parameters ──────────────────────────────────────────────────────────
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
NUM_CLASSES = 7
MAX_LEN     = 64
EPOCHS      = 4
BATCH_SIZE  = 64
LR          = 2e-5
DROPOUT     = 0.3
ATTE_DROPOUT= 0.1
SEED        = 42
WEIGHT_DECAY= 0
MAX_NORM    = 1.0
VALIDATION_STEPS = 100

torch.manual_seed(SEED)

# ── Dataset ───────────────────────────────────────────────────────────────────
class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len: int):
        self.data   = texts
        self.labels = labels  # may be None for test set
        self.tokenizer = tokenizer
        self.max_len =max_len
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        y = int(self.labels[idx]) if self.labels is not None else -1
        
        encoding=self.tokenizer(
            x,
            truncation=True,
            padding='max_length',
            max_length=self.max_len,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(), 
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(y, dtype=torch.long)
        }

# ── Training helpers ──────────────────────────────────────────────────────────
# def train_epoch(model, loader, optimizer, device):
#     model.train()
#     total_loss = 0.0
    
#     for batch in loader:
#         input_ids = batch['input_ids'].to(device)
#         attention_mask = batch['attention_mask'].to(device)
#         labels = batch['labels'].to(device)
        
        
#         output=model(
#             input_ids=input_ids,
#             attention_mask=attention_mask,
#             labels=labels
#         )
        
#         loss=output.loss
#         optimizer.zero_grad()
#         loss.backward()

#         torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm= MAX_NORM)
#         optimizer.step()
        
#         total_loss += loss.item()
#     return total_loss / len(loader)

def train_step(model, batch, optimizer, device):
    model.train()
    
    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    labels = batch['labels'].to(device)
        
    output=model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels
    )
    
    loss=output.loss
    optimizer.zero_grad()
    loss.backward()

    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm= MAX_NORM)
    optimizer.step()

    return loss.item()

def evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels']
            
            preds = model(input_ids=input_ids,attention_mask=attention_mask).logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
    return f1_score(all_labels, all_preds, average="macro")
                    
                    


def plot_training_curves(history, save_path='training_curves.png'):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss
    ax1.plot(history['steps'], history['train_loss'], 'b-', label='Train Loss', alpha=0.7)
    val_steps = [h['step'] for h in history['val_history']]
    val_losses = [h['loss'] for h in history['val_history']]
    ax1.scatter(val_steps, val_losses, c='red', s=50, label='Validation Loss', zorder=5)
    ax1.set_xlabel('Training Steps')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # F1
    ax2.plot(history['steps'], history['train_f1'], 'g-', label='Train F1', alpha=0.7)
    val_steps = [h['step'] for h in history['val_history']]
    val_f1s = [h['f1'] for h in history['val_history']]
    ax2.scatter(val_steps, val_f1s, c='orange', s=50, label='Validation F1', zorder=5)
    ax2.set_xlabel('Training Steps')
    ax2.set_ylabel('F1 Score')
    ax2.set_title('Training and Validation F1 Score')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=100)
    plt.show()
    print(f"Training curves saved to {save_path}")

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    train_df = pd.read_csv("/kaggle/input/datasets/meixuanli3/three-data/data/train.csv")
    valid_df = pd.read_csv("/kaggle/input/datasets/meixuanli3/three-data/data/valid.csv")
    test_df  = pd.read_csv("/kaggle/input/datasets/meixuanli3/three-data/data/test_no_label.csv")
    
    print("Building dataset …")
    tokenizer=DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    train_ds = TextDataset(train_df["text"], train_df["label"].values, tokenizer, MAX_LEN)
    valid_ds = TextDataset(valid_df["text"], valid_df["label"].values, tokenizer, MAX_LEN)
    test_ds  = TextDataset(test_df["text"],  None,                     tokenizer, MAX_LEN)
    
    print("Building dataloader...")
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    valid_loader = DataLoader(valid_ds, batch_size=512,        shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=512,        shuffle=False, num_workers=0)

    model=DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased",
        num_labels=NUM_CLASSES,
        dropout=DROPOUT,
        attention_dropout=ATTE_DROPOUT
    ).to(DEVICE)
    
    
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    
    best_f1, best_state = 0.0, None
    history = {
    'steps': [],
    'train_loss': [],
    'train_f1': [],
    'val_history': []
    }
    
    global_steps=0
    
    for epoch in range(1, EPOCHS + 1):
        print(f"\n{'='*30}")
        print(f"Epoch {epoch}/{EPOCHS}")
        print(f"{'='*30}")

        epoch_preds = []
        epoch_labels = []
        epoch_loss = 0.0
        epoch_steps = 0
        start_time=time.time()
        for batch in train_loader:
            loss= train_step(model,batch,optimizer,DEVICE)
            
            with torch.no_grad():
                model.eval()
                input_ids = batch['input_ids'].to(DEVICE)
                attention_mask = batch['attention_mask'].to(DEVICE)
                labels = batch['labels']
                
                preds = model(input_ids=input_ids,attention_mask=attention_mask).logits.argmax(dim=1).cpu().numpy()
                model.train()
                epoch_preds.extend(preds)
                epoch_labels.extend(labels.numpy())
                
            epoch_steps+=1
            epoch_loss+=loss
            global_steps+=1

            if global_steps % VALIDATION_STEPS==0:
                avg_train_loss = epoch_loss/epoch_steps
                
                if len(epoch_preds) > 0:
                    train_f1 = f1_score(epoch_labels, epoch_preds, average="macro")
                else:
                    train_f1 = 0.0

                val_f1 = evaluate(model, valid_loader, DEVICE)

                history['steps'].append(global_steps)
                history['train_loss'].append(avg_train_loss)
                history['train_f1'].append(train_f1)
                history['val_history'].append({
                    'step': global_steps,
                    'f1': val_f1,
                    'loss': avg_train_loss
                })
                
                print(f"Step {global_steps:5d} | "
                      f"Train Loss: {avg_train_loss:.4f} | "
                      f"Train F1: {train_f1:.4f} | "
                      f"Val F1: {val_f1:.4f}")
                
                if val_f1 > best_f1:
                    best_f1 = val_f1
                    best_state = {k: v.clone() for k, v in model.state_dict().items()}
                    print(f"\nBest Validation Macro-F1: {best_f1:.4f}")

                epoch_preds = []
                epoch_labels = []
                epoch_loss = 0.0
                epoch_steps = 0
        end_time=time.time()
        final_val_f1 = evaluate(model, valid_loader, DEVICE)
        print(f"\nEpoch {epoch} Complete | Val F1-Macro: {final_val_f1:.4f} | Time: {end_time-start_time}")
    # ── Generate test predictions ────────────────────────────────────────────
    model.load_state_dict(best_state)
    model.eval()
    all_preds = []
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            
            preds = model(input_ids=input_ids,attention_mask=attention_mask).logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)

    out = pd.DataFrame({"id": test_df["id"], "label": all_preds})
    out.to_csv("rnn_pred.csv", index=False)
    print("Saved rnn_pred.csv")

    with open("history.json", 'w') as f:
        json.dump(history, f, indent=2)
    print("Saved history.json")

    plot_training_curves(history, save_path='training_curves.png')
    print("Saved curves")

    
if __name__ == "__main__":
    main()
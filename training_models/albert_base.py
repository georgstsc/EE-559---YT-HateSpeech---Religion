# Parse command-line arguments
# parser = argparse.ArgumentParser(description="Train ALBERT model on hate speech dataset")
# parser.add_argument('--train_path', type=str, default="/home/schwabed/EE-559---YT-HateSpeech---Religion/data/train_balanced.csv",
#                     help="Path to training CSV file")
# parser.add_argument('--val_path', type=str, default="/home/schwabed/EE-559---YT-HateSpeech---Religion/data/val_balanced.csv",
#                     help="Path to validation CSV file")
# parser.add_argument('--model_path', type=str, default="/home/schwabed/EE-559---YT-HateSpeech---Religion/models/albert_base_retrained",
#                     help="Path to save trained ALBERT model")
# parser.add_argument('--batch_size', type=int, default=16,
#                     help="Batch size for training")
# parser.add_argument('--epochs', type=int, default=10,
#                     help="Number of training epochs")
# parser.add_argument('--max_length', type=int, default=256,
#                     help="Maximum sequence length for tokenization")
# parser.add_argument('--learning_rate', type=float, default=1e-5,
#                     help="Learning rate for optimizer")
# args = parser.parse_args()

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from sklearn.metrics import accuracy_score, f1_score
import pandas as pd
from tqdm import tqdm
import argparse
import os
import tempfile
import optuna

parser = argparse.ArgumentParser(description="Train ALBERT model on hate speech dataset")
parser.add_argument('--train_path', type=str, default="../data/train_balanced.csv")
parser.add_argument('--val_path', type=str, default="../data/val_balanced.csv")
parser.add_argument('--model_path', type=str, default="../models/albert_base_optuna")
parser.add_argument('--epochs', type=int, default=10)
args = parser.parse_args()

# Config
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🖥️ Device: {device}")

def check_write_permission(path):
    try:
        os.makedirs(path, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=path, delete=True) as tmp_file:
            tmp_file.write(b"Test write permission")
            tmp_file.flush()
        print(f"✅ Write permission confirmed for {path}")
    except (OSError, PermissionError) as e:
        raise PermissionError(f"Cannot write to {path}: {str(e)}")

check_write_permission(args.model_path)

train_df = pd.read_csv(args.train_path)
val_df = pd.read_csv(args.val_path)
if 'text' not in train_df.columns or 'label' not in train_df.columns:
    raise ValueError("CSV files must have 'text' and 'label' columns")

model_name = "albert-base-v2"

class CommentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.encodings = tokenizer(texts.tolist(), truncation=True, padding=True, max_length=max_length)
        self.labels = labels.tolist()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            'input_ids': torch.tensor(self.encodings['input_ids'][idx]),
            'attention_mask': torch.tensor(self.encodings['attention_mask'][idx]),
            'labels': torch.tensor(self.labels[idx])
        }

def train_and_evaluate(params):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    train_data = CommentDataset(train_df['text'], train_df['label'], tokenizer, params['max_length'])
    val_data = CommentDataset(val_df['text'], val_df['label'], tokenizer, params['max_length'])
    train_loader = DataLoader(train_data, batch_size=params['batch_size'], shuffle=True)
    val_loader = DataLoader(val_data, batch_size=params['batch_size'], shuffle=False)

    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    model.to(device)

    class_counts = train_df['label'].value_counts().sort_index().values
    class_weights = torch.tensor([1.0 / count for count in class_counts], dtype=torch.float).to(device)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

    optimizer = AdamW(model.parameters(), lr=params['learning_rate'])
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=params['learning_rate'],
                                                    steps_per_epoch=len(train_loader), epochs=args.epochs)

    best_f1 = 0
    for epoch in range(args.epochs):
        model.train()
        for batch in tqdm(train_loader, desc=f"🚀 Training Epoch {epoch+1}/{args.epochs}"):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = criterion(outputs.logits, batch['labels'])
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        model.eval()
        val_preds, val_labels = [], []
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="🔍 Validating"):
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)
                preds = torch.argmax(outputs.logits, dim=-1)
                val_preds.extend(preds.cpu().tolist())
                val_labels.extend(batch['labels'].cpu().tolist())

        val_f1 = f1_score(val_labels, val_preds)
        if val_f1 > best_f1:
            best_f1 = val_f1

    return best_f1

def objective(trial):
    params = {
        'learning_rate': trial.suggest_loguniform("learning_rate", 1e-6, 1e-4),
        'batch_size': trial.suggest_categorical("batch_size", [8, 16, 32]),
        'max_length': trial.suggest_int("max_length", 128, 512, step=64)
    }
    return train_and_evaluate(params)

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=10)
print("Best hyperparameters:", study.best_params)

# Retrain with best parameters and save the model
print("Retraining final model with best parameters...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
train_data = CommentDataset(train_df['text'], train_df['label'], tokenizer, study.best_params['max_length'])
train_loader = DataLoader(train_data, batch_size=study.best_params['batch_size'], shuffle=True)

model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
model.to(device)

class_counts = train_df['label'].value_counts().sort_index().values
class_weights = torch.tensor([1.0 / count for count in class_counts], dtype=torch.float).to(device)
criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

optimizer = AdamW(model.parameters(), lr=study.best_params['learning_rate'])
scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=study.best_params['learning_rate'],
                                                steps_per_epoch=len(train_loader), epochs=args.epochs)

for epoch in range(args.epochs):
    model.train()
    for batch in tqdm(train_loader, desc=f"🚀 Final Training Epoch {epoch+1}/{args.epochs}"):
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = criterion(outputs.logits, batch['labels'])
        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

model.save_pretrained(args.model_path)
tokenizer.save_pretrained(args.model_path)
print(f"✅ Final model saved to {args.model_path}")

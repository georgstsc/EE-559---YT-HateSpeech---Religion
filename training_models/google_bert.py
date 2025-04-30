# Parse command-line arguments
# parser = argparse.ArgumentParser(description="Train Google BERT model on hate speech dataset")
# parser.add_argument('--train_path', type=str, default="/home/schwabed/EE-559---YT-HateSpeech---Religion/data/train_balanced.csv",
#                     help="Path to training CSV file")
# parser.add_argument('--val_path', type=str, default="/home/schwabed/EE-559---YT-HateSpeech---Religion/data/val_balanced.csv",
#                     help="Path to validation CSV file")
# parser.add_argument('--model_path', type=str, default="/home/schwabed/EE-559---YT-HateSpeech---Religion/models/bert-compact-hate",
#                     help="Path to save trained Google BERT model")
# parser.add_argument('--batch_size', type=int, default=16,
#                     help="Batch size for training")
# parser.add_argument('--epochs', type=int, default=10,
#                     help="Number of training epochs")
# parser.add_argument('--max_length', type=int, default=256,
#                     help="Maximum sequence length for tokenization")
# parser.add_argument('--learning_rate', type=float, default=3e-5,
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

# ----------------- ARGUMENT PARSING -----------------
parser = argparse.ArgumentParser(description="Tune Google BERT Compact on hate speech")
parser.add_argument('--train_path', type=str, default="../data/train_balanced.csv")
parser.add_argument('--val_path', type=str, default="../data/val_balanced.csv")
parser.add_argument('--model_path', type=str, default="../models/bert_compact_optuna")
parser.add_argument('--epochs', type=int, default=10)
args = parser.parse_args()

# ----------------- CONFIG -----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🖥️ Device: {device}")

def check_write_permission(path):
    try:
        os.makedirs(path, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=path, delete=True) as tmp_file:
            tmp_file.write(b"Test")
            tmp_file.flush()
        print(f"✅ Write permission confirmed for {path}")
    except Exception as e:
        raise PermissionError(f"❌ Cannot write to {path}: {e}")

check_write_permission(args.model_path)

# ----------------- LOAD DATA -----------------
train_df = pd.read_csv(args.train_path)
val_df = pd.read_csv(args.val_path)
if 'text' not in train_df.columns or 'label' not in train_df.columns:
    raise ValueError("CSV must have 'text' and 'label' columns")

model_name = "google/bert_uncased_L-4_H-256_A-4"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# ----------------- DATASET -----------------
class CommentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.encodings = tokenizer(texts.tolist(), truncation=True, padding=True, max_length=max_length)
        self.labels = labels.tolist()
    def __len__(self): return len(self.labels)
    def __getitem__(self, idx):
        return {
            'input_ids': torch.tensor(self.encodings['input_ids'][idx]),
            'attention_mask': torch.tensor(self.encodings['attention_mask'][idx]),
            'labels': torch.tensor(self.labels[idx])
        }

# ----------------- TRAIN & EVAL FUNCTION -----------------
def train_and_eval(params):
    train_data = CommentDataset(train_df['text'], train_df['label'], tokenizer, params['max_length'])
    val_data = CommentDataset(val_df['text'], val_df['label'], tokenizer, params['max_length'])
    train_loader = DataLoader(train_data, batch_size=params['batch_size'], shuffle=True)
    val_loader = DataLoader(val_data, batch_size=params['batch_size'], shuffle=False)

    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2).to(device)

    class_counts = train_df['label'].value_counts().sort_index().values
    weights = torch.tensor([sum(class_counts) / (2.0 * c) for c in class_counts], dtype=torch.float).to(device)
    criterion = torch.nn.CrossEntropyLoss(weight=weights)

    optimizer = AdamW(model.parameters(), lr=params['learning_rate'])
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=params['learning_rate'],
                                                    steps_per_epoch=len(train_loader), epochs=args.epochs)

    best_f1 = 0
    for epoch in range(args.epochs):
        model.train()
        for batch in tqdm(train_loader, desc=f"🚀 Epoch {epoch+1}/{args.epochs}"):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = criterion(outputs.logits, batch['labels'])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        model.eval()
        preds, labels = [], []
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="🔍 Validating"):
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)
                pred = torch.argmax(outputs.logits, dim=-1)
                preds.extend(pred.cpu().tolist())
                labels.extend(batch['labels'].cpu().tolist())

        val_f1 = f1_score(labels, preds, zero_division=0)
        if val_f1 > best_f1:
            best_f1 = val_f1

    return best_f1

# ----------------- OPTUNA OBJECTIVE -----------------
def objective(trial):
    params = {
        "learning_rate": trial.suggest_loguniform("learning_rate", 1e-6, 1e-4),
        "batch_size": trial.suggest_categorical("batch_size", [8, 16, 32]),
        "max_length": trial.suggest_int("max_length", 128, 512, step=64),
    }
    return train_and_eval(params)

# ----------------- OPTUNA TUNING -----------------
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=10)
print("🏁 Best hyperparameters found:", study.best_params)

# ----------------- FINAL TRAINING -----------------
print("🎯 Retraining final model...")
best_params = study.best_params
train_data = CommentDataset(train_df['text'], train_df['label'], tokenizer, best_params['max_length'])
train_loader = DataLoader(train_data, batch_size=best_params['batch_size'], shuffle=True)

model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2).to(device)
weights = torch.tensor([sum(class_counts) / (2.0 * c) for c in class_counts], dtype=torch.float).to(device)
criterion = torch.nn.CrossEntropyLoss(weight=weights)

optimizer = AdamW(model.parameters(), lr=best_params['learning_rate'])
scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=best_params['learning_rate'],
                                                steps_per_epoch=len(train_loader), epochs=args.epochs)

for epoch in range(args.epochs):
    model.train()
    for batch in tqdm(train_loader, desc=f"🔥 Final Train Epoch {epoch+1}/{args.epochs}"):
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = criterion(outputs.logits, batch['labels'])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

os.makedirs(args.model_path, exist_ok=True)
model.save_pretrained(args.model_path)
tokenizer.save_pretrained(args.model_path)
print(f"✅ Final model saved to: {args.model_path}")

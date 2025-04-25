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

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Train DistilBERT model on hate speech dataset")
parser.add_argument('--train_path', type=str, default="/home/schwabed/EE-559---YT-HateSpeech---Religion/data/train_balanced.csv",
                    help="Path to training CSV file")
parser.add_argument('--val_path', type=str, default="/home/schwabed/EE-559---YT-HateSpeech---Religion/data/val_balanced.csv",
                    help="Path to validation CSV file")
parser.add_argument('--model_path', type=str, default="/home/schwabed/EE-559---YT-HateSpeech---Religion/models/distilbert",
                    help="Path to save trained DistilBERT model")
parser.add_argument('--batch_size', type=int, default=16,
                    help="Batch size for training")
parser.add_argument('--epochs', type=int, default=10,
                    help="Number of training epochs")
parser.add_argument('--max_length', type=int, default=256,
                    help="Maximum sequence length for tokenization")
parser.add_argument('--learning_rate', type=float, default=5e-5,
                    help="Learning rate for optimizer")
args = parser.parse_args()

# Config
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🖥️ Device: {device}")

# Check write permissions for model_path
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

# Load data
train_df = pd.read_csv(args.train_path)
val_df = pd.read_csv(args.val_path)
if 'text' not in train_df.columns or 'label' not in train_df.columns:
    raise ValueError("CSV files must have 'text' and 'label' columns")

# Tokenizer
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Dataset wrapper
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

# Prepare datasets & loaders
train_data = CommentDataset(train_df['text'], train_df['label'], tokenizer, args.max_length)
val_data = CommentDataset(val_df['text'], val_df['label'], tokenizer, args.max_length)
train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False)

# Model
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
model.to(device)

# Compute class weights
class_counts = train_df['label'].value_counts().sort_index().values
class_weights = torch.tensor([1.0 / count for count in class_counts], dtype=torch.float).to(device)
criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

# Optimizer and scheduler
optimizer = AdamW(model.parameters(), lr=args.learning_rate)
total_steps = len(train_loader) * args.epochs
scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.learning_rate, steps_per_epoch=len(train_loader), epochs=args.epochs)

# Training loop
best_f1 = 0.0
patience = 3
counter = 0
for epoch in range(args.epochs):
    model.train()
    train_loss = 0
    for batch in tqdm(train_loader, desc=f"🚀 Training Epoch {epoch+1}/{args.epochs}"):
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = criterion(outputs.logits, batch['labels'])
        train_loss += loss.item()
        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    avg_train_loss = train_loss / len(train_loader)
    print(f"✅ Epoch {epoch+1} - Average Training Loss: {avg_train_loss:.4f}")

    # Validation
    model.eval()
    val_preds, val_labels = [], []
    val_loss = 0
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validating"):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = criterion(outputs.logits, batch['labels'])
            val_loss += loss.item()
            preds = torch.argmax(outputs.logits, dim=-1)
            val_preds.extend(preds.cpu().tolist())
            val_labels.extend(batch['labels'].cpu().tolist())

    avg_val_loss = val_loss / len(val_loader)
    val_accuracy = accuracy_score(val_labels, val_preds)
    val_f1 = f1_score(val_labels, val_preds)
    print(f"✅ Epoch {epoch+1} - Average Validation Loss: {avg_val_loss:.4f}")
    print(f"Validation Accuracy: {val_accuracy:.4f}")
    print(f"Validation F1 Score: {val_f1:.4f}")

    # Early stopping
    if val_f1 > best_f1:
        best_f1 = val_f1
        counter = 0
        model.save_pretrained(args.model_path)
        tokenizer.save_pretrained(args.model_path)
        print(f"✅ New best F1 score: {best_f1:.4f}. Model saved to {args.model_path}")
    else:
        counter += 1
        if counter >= patience:
            print(f"✅ No improvement in F1 for {patience} epochs. Stopping early.")
            break

if counter < patience:
    print(f"✅ Training completed. Final model saved to {args.model_path}")
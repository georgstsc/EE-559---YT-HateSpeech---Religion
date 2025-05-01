import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
import pandas as pd
from tqdm import tqdm
import argparse
import os
import re
from datasets import load_dataset

# -----------------------
# 🔧 Command-line args
# -----------------------
parser = argparse.ArgumentParser(description="Evaluate model on test + ETHOS datasets")
parser.add_argument('--data_path', type=str, default="../data/test_balanced.csv")
parser.add_argument('--output_path', type=str, default="../output_tests/albert_base_ethos.csv")
parser.add_argument('--model_path', type=str, default="../models/albert_base")
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--max_length', type=int, default=256)
args = parser.parse_args()

# -----------------------
# ⚙️ Config
# -----------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🖥️ Device: {device}")

# -----------------------
# ✅ Load model + tokenizer
# -----------------------
tokenizer = AutoTokenizer.from_pretrained(args.model_path)
model = AutoModelForSequenceClassification.from_pretrained(args.model_path)
model.to(device)
model.eval()

# -----------------------
# 📁 Load Test Dataset
# -----------------------
test_df = pd.read_csv(args.data_path)
assert 'text' in test_df.columns and 'label' in test_df.columns, "Missing required columns"

# ✅ Dataset wrapper
class CommentDataset(Dataset):
    def __init__(self, texts, labels):
        self.encodings = tokenizer(texts.tolist(), truncation=True, padding=True, max_length=args.max_length)
        self.labels = labels.tolist()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            'input_ids': torch.tensor(self.encodings['input_ids'][idx]),
            'attention_mask': torch.tensor(self.encodings['attention_mask'][idx]),
            'labels': torch.tensor(self.labels[idx])
        }

# -----------------------
# 📊 Evaluate function
# -----------------------
def evaluate(model, dataloader, name="Dataset"):
    all_preds, all_labels = [], []
    total_loss = 0
    criterion = torch.nn.CrossEntropyLoss()

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Evaluating: {name}"):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = criterion(outputs.logits, batch["labels"])
            total_loss += loss.item()
            preds = torch.argmax(outputs.logits, dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(batch["labels"].cpu().tolist())

    avg_loss = total_loss / len(dataloader)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds)
    rec = recall_score(all_labels, all_preds)
    cm = confusion_matrix(all_labels, all_preds)

    print(f"\n📋 Results on {name}:")
    print(f"📉 Avg Loss: {avg_loss:.4f}")
    print(f"✅ Accuracy: {acc:.4f} | 🎯 Precision: {prec:.4f} | 🔁 Recall: {rec:.4f} | 🧠 F1 Score: {f1:.4f}")
    print("🧾 Confusion Matrix:")
    print(f"[[TN={cm[0][0]} FP={cm[0][1]}]")
    print(f" [FN={cm[1][0]} TP={cm[1][1]}]]")

# -----------------------
# 📌 Evaluate on TEST SET
# -----------------------
test_dataset = CommentDataset(test_df['text'], test_df['label'])
test_loader = DataLoader(test_dataset, batch_size=args.batch_size)
evaluate(model, test_loader, name="Original Test Set")

# Save predictions with tqdm
preds = []
with torch.no_grad():
    for batch in tqdm(test_loader, desc="Saving Predictions"):
        batch = {k: v.to(device) for k, v in batch.items()}
        logits = model(**batch).logits
        preds.extend(torch.argmax(logits, dim=-1).cpu().tolist())

pd.DataFrame({
    'text': test_df['text'],
    'true_label': test_df['label'],
    'predicted_label': preds
}).to_csv(args.output_path, index=False)

print(f"✅ Test predictions saved to {args.output_path}")

# -----------------------
# 🌍 ETHOS Generalization
# -----------------------
ethos = load_dataset("ethos", "binary", split="train").to_pandas()

# 🧠 Filter religion mentions
religion_keywords = [
    "muslim", "islam", "islamic", "jew", "jewish", "judaism",
    "christian", "christianity", "bible", "jesus", "god", "catholic", "pope",
    "hindu", "hinduism", "buddha", "buddhist", "atheist", "religion", "religious"
]
def mentions_religion(text):
    text = str(text).lower()
    return any(re.search(rf"\b{kw}\b", text) for kw in religion_keywords)

ethos["mentions_religion"] = ethos["text"].apply(mentions_religion)
ethos_religion = ethos[ethos["mentions_religion"]]
ethos_nonreligion = ethos[~ethos["mentions_religion"]]

# 🔍 Subset Dataset class
class EthosSubset(Dataset):
    def __init__(self, texts, labels):
        encodings = tokenizer(texts.tolist(), truncation=True, padding=True, max_length=args.max_length)
        self.input_ids = encodings["input_ids"]
        self.attention_mask = encodings["attention_mask"]
        self.labels = labels.tolist()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids": torch.tensor(self.input_ids[idx]),
            "attention_mask": torch.tensor(self.attention_mask[idx]),
            "labels": torch.tensor(self.labels[idx])
        }

# ✅ Loaders and Evaluation
print("\n🌍 ETHOS Generalization Tests")
ethos_loader_religion = DataLoader(EthosSubset(ethos_religion["text"], ethos_religion["label"]), batch_size=32)
ethos_loader_nonreligion = DataLoader(EthosSubset(ethos_nonreligion["text"], ethos_nonreligion["label"]), batch_size=32)

evaluate(model, ethos_loader_religion, name="ETHOS - Religion Subset")
evaluate(model, ethos_loader_nonreligion, name="ETHOS - No Keyword Subset (General Hate)")

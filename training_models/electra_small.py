import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score
import pandas as pd
from tqdm import tqdm

# ✅ Config
model_name = "google/electra-small-discriminator"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("🖥️ Device:", device)

# ✅ Load data
train_df = pd.read_csv("../data/train_balanced.csv")
val_df = pd.read_csv("../data/val_balanced.csv")
test_df = pd.read_csv("../data/test_balanced.csv")

# ✅ Tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)

# ✅ Dataset wrapper
class CommentDataset(Dataset):
    def __init__(self, texts, labels):
        self.encodings = tokenizer(texts.tolist(), truncation=True, padding=True, max_length=256)
        self.labels = labels.tolist()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            'input_ids': torch.tensor(self.encodings['input_ids'][idx]),
            'attention_mask': torch.tensor(self.encodings['attention_mask'][idx]),
            'labels': torch.tensor(self.labels[idx])
        }

# ✅ Prepare datasets & loaders
train_data = CommentDataset(train_df['text'], train_df['label'])
val_data = CommentDataset(val_df['text'], val_df['label'])
test_data = CommentDataset(test_df['text'], test_df['label'])

train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
val_loader = DataLoader(val_data, batch_size=64)
test_loader = DataLoader(test_data, batch_size=64)

# ✅ Model
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
model.to(device)

# ✅ Optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5)

# ✅ Training loop
epochs = 3
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for batch in tqdm(train_loader, desc=f"🚀 Epoch {epoch+1}/{epochs}"):
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        total_loss += loss.item()
    print(f"✅ Avg Training Loss (epoch {epoch+1}): {total_loss / len(train_loader):.4f}")

# ✅ Save model
save_path = "../models/electra-small-hate"
model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)
print(f"✅ Model saved to: {save_path}")

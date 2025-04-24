import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
import pandas as pd
from tqdm import tqdm
import argparse
import os

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Evaluate BERT-Tiny model on a test dataset")
parser.add_argument('--data_path', type=str, required=True,
                    help="Path to test CSV file (must have 'text' and 'label' columns)")
parser.add_argument('--model_path', type=str, required=True,
                    help="Path to the trained BERT-Tiny model directory")
parser.add_argument('--output_path', type=str, required=True,
                    help="Path to save predictions CSV")
parser.add_argument('--batch_size', type=int, default=16,
                    help="Batch size for evaluation")
parser.add_argument('--max_length', type=int, default=256,
                    help="Maximum sequence length for tokenization")
args = parser.parse_args()

# Config
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🖥️ Device: {device}")

# Verify paths
if not os.path.exists(args.data_path):
    raise FileNotFoundError(f"Test data file not found: {args.data_path}")
if not os.path.exists(args.model_path):
    raise FileNotFoundError(f"Model directory not found: {args.model_path}")

# Load data
test_df = pd.read_csv(args.data_path)
if 'text' not in test_df.columns or 'label' not in test_df.columns:
    raise ValueError("Test CSV must have 'text' and 'label' columns")

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(args.model_path)
model = AutoModelForSequenceClassification.from_pretrained(args.model_path)
model.to(device)
model.eval()

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

# Prepare dataset and loader
test_data = CommentDataset(test_df['text'], test_df['label'], tokenizer, args.max_length)
test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)

# Evaluation
test_preds, test_labels = [], []
test_loss = 0
criterion = torch.nn.CrossEntropyLoss()

with torch.no_grad():
    for batch in tqdm(test_loader, desc="Evaluating BERT-Tiny"):
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = criterion(outputs.logits, batch['labels'])
        test_loss += loss.item()
        preds = torch.argmax(outputs.logits, dim=-1)
        test_preds.extend(preds.cpu().tolist())
        test_labels.extend(batch['labels'].cpu().tolist())

# Compute metrics
avg_test_loss = test_loss / len(test_loader)
accuracy = accuracy_score(test_labels, test_preds)
f1 = f1_score(test_labels, test_preds)
precision = precision_score(test_labels, test_preds)
recall = recall_score(test_labels, test_preds)
conf_matrix = confusion_matrix(test_labels, test_preds)

# Print results
print(f"\n✅ Evaluation Results for BERT-Tiny on {os.path.basename(args.data_path)}:")
print(f"Average Test Loss: {avg_test_loss:.4f}")
print(f"Accuracy: {accuracy:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"Confusion Matrix:\n{conf_matrix}")

# Save predictions
predictions_df = pd.DataFrame({
    'text': test_df['text'],
    'label': test_labels,
    'prediction': test_preds
})
predictions_df.to_csv(args.output_path, index=False)
print(f"Predictions saved to {args.output_path}")
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
import pandas as pd
from tqdm import tqdm
import argparse
import os

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Evaluate Google BERT model on a test dataset")
parser.add_argument('--data_path', type=str, required=True,
                    help="Path to test CSV file (must have 'text' and 'label' columns)")
parser.add_argument('--output_path', type=str, required=True,
                    help="Path to save predictions CSV")
parser.add_argument('--model_path', type=str, default="/home/schwabed/EE-559---YT-HateSpeech---Religion/models/bert-compact-hate",
                    help="Path to trained Google BERT model")
parser.add_argument('--batch_size', type=int, default=64,
                    help="Batch size for evaluation")
parser.add_argument('--max_length', type=int, default=256,
                    help="Maximum sequence length for tokenization")
args = parser.parse_args()

# Config
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🖥️ Device: {device}")

# Validate input file
if not os.path.exists(args.data_path):
    raise FileNotFoundError(f"Test file not found at {args.data_path}")
test_df = pd.read_csv(args.data_path)
if 'text' not in test_df.columns or 'label' not in test_df.columns:
    raise ValueError("Test CSV must have 'text' and 'label' columns")
if not test_df['label'].isin([0, 1]).all():
    raise ValueError("Labels must be binary (0 or 1)")

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

# Load tokenizer and model
if not os.path.exists(args.model_path):
    raise FileNotFoundError(f"Model directory not found at {args.model_path}")
tokenizer = AutoTokenizer.from_pretrained(args.model_path)
model = AutoModelForSequenceClassification.from_pretrained(args.model_path)
model.to(device)
model.eval()

# Prepare test dataset and loader
test_data = CommentDataset(test_df['text'], test_df['label'], tokenizer, args.max_length)
test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)

# Evaluate
test_preds, test_labels = [], []
test_loss = 0
criterion = torch.nn.CrossEntropyLoss()

with torch.no_grad():
    for i, batch in enumerate(tqdm(test_loader, desc="Evaluating")):
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = criterion(outputs.logits, batch['labels'])
        test_loss += loss.item()
        preds = torch.argmax(outputs.logits, dim=-1)
        test_preds.extend(preds.cpu().tolist())
        test_labels.extend(batch['labels'].cpu().tolist())
        # Debug: Print logits for first batch
        if i == 0:
            print("\nSample logits (first 5 samples):")
            for j in range(min(5, len(outputs.logits))):
                logits = outputs.logits[j].cpu().tolist()
                print(f"Sample {j+1}: Logits = {logits}, Predicted = {preds[j].item()}, True = {batch['labels'][j].item()}")

# Compute metrics
avg_test_loss = test_loss / len(test_loader)
accuracy = accuracy_score(test_labels, test_preds)
f1 = f1_score(test_labels, test_preds, zero_division=0)
precision = precision_score(test_labels, test_preds, zero_division=0)
recall = recall_score(test_labels, test_preds, zero_division=0)
conf_matrix = confusion_matrix(test_labels, test_preds)

# Print results
print(f"\n✅ Evaluation Results on {os.path.basename(args.data_path)}:")
print(f"Average Test Loss: {avg_test_loss:.4f}")
print(f"Accuracy: {accuracy:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print("\nConfusion Matrix:")
print(f"[[TN={conf_matrix[0,0]} FP={conf_matrix[0,1]}]")
print(f" [FN={conf_matrix[1,0]} TP={conf_matrix[1,1]}]]")

# Save predictions
results_df = pd.DataFrame({
    'text': test_df['text'],
    'true_label': test_labels,
    'predicted_label': test_preds
})
results_df.to_csv(args.output_path, index=False)
print(f"Predictions saved to {args.output_path}")
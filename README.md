# 🕊️ Religious Hate Speech Detection with Transformers

This repository contains our EE-559 mini-project for detecting **religious hate speech** in user-generated comments. We trained and compared six lightweight transformer models, optimized them using Optuna, and deployed our best-performing model in a multilingual **Gradio web app**.

🔗 **Live Demo**: [huggingface.co/spaces/zazo2002/Hate_Speech_Detector_Deep_Learning](https://huggingface.co/spaces/zazo2002/Hate_Speech_Detector_Deep_Learning)

---

## 🧠 Project Overview

Our system performs binary classification:

- `0`: Not Hate  
- `1`: Religious Hate

We evaluated the following six transformer-based models:

- `bert-tiny`
- `distilbert-base-uncased`
- `roberta-base`
- `albert-base-v2`
- `electra-small-discriminator`
- `bert_uncased_L-4_H-256_A-4` (Google compact BERT)

Each model was fine-tuned on a balanced version of our dataset, then optimized using Optuna for hyperparameter tuning. DistilBERT achieved the best trade-off between recall and precision and was selected for deployment.

---

## 🌍 Dataset and Generalization

We constructed our dataset by filtering the Civil Comments dataset for religion-related content, then labeling it based on keyword presence and toxicity thresholds. The dataset was split into training (balanced via upsampling), validation, and test sets (unbalanced).

To evaluate generalization, we tested our models on the public **ETHOS dataset**, revealing significant drops in performance for implicit or keyword-free hate speech—highlighting the challenges of real-world deployment.

---

## 🗂️ Project Structure

```text
.
├── app.py                # Gradio application (text/audio/file interface)
├── requirements.txt      # Python dependencies
├── Dockerfile            # Optional Docker config
├── data/                 # Training, validation, test splits + ETHOS
│   ├── train_balanced.csv
│   ├── val_balanced.csv
│   ├── test_balanced.csv
│   └── external/ethos_dataset.csv
├── models/               # Fine-tuned transformer model checkpoints
├── outputs_tests/        # CSV predictions from test runs
├── training_models/      # Training scripts per model
├── testing_models/       # Evaluation scripts per model and dataset
└── README.md             # Project documentation

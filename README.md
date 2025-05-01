# 🕊️ Religious Hate Speech Detection with Transformers

This repository contains our EE-559 mini-project for detecting **religious hate speech** in user-generated comments. We fine-tune and compare several transformer-based models and provide a user-facing **Gradio web app** to test predictions interactively.

---

## 🧠 Project Overview

Our system is designed to classify text as:

- `0`: Not Hate  
- `1`: Religious Hate

We experiment with 7 transformer models:

- `bert-tiny`
- `distilbert-base-uncased`
- `bert-base-uncased`
- `roberta-base`
- `albert-base-v2`
- `electra-small-discriminator`
- `bert_uncased_L-4_H-256_A-4`

In addition to training and evaluation scripts, we provide an app with language detection, translation, and word-level importance highlighting.

---

## 🗂️ Repository Structure

 ├── app.py # Gradio web app 
 ├── training_models/ # Model training scripts 
 ├── testing_models/ # Evaluation scripts 
 ├── models/ # Saved fine-tuned models 
 ├── data/ # Balanced CSVs and external datasets 
 ├── outputs_tests/ # Output CSVs from evaluations 
 ├── requirements.txt # Dependencies  to finish 
 ├── test_predictions.csv # to delete
 ├── LICENSE 
 ├── README.md 

 ---

## 🚀 Running the Gradio App

You can launch the web interface locally with:

    python app.py

### Features

- 📝 Paste Text: Manually enter a comment
- 🎤 Voice Input: Speak a comment (transcribed using Whisper)
- 📄 File Upload: Upload `.txt` files for batch prediction

The app supports:

- 🌍 Auto language detection (English, French, German, Italian)
- 🌐 Automatic translation to English before analysis
- 🔍 Religious hate speech classification
- 💡 Word importance highlighting
- 📊 Confidence scores display

---

## 🏋️ Training the Models

To train a model, run the appropriate script inside the `training_models/` directory. Example:

    python training_models/train_distilbert.py

Requirements:

- Training data should be in `data/train_balanced.csv`
- Validation data should be in `data/val_balanced.csv`
- Test data should be in `data/test_balanced.csv`

Trained models will be saved inside the `models/` folder.

---

## 📊 Model Evaluation

Use the `testing_models/` scripts to evaluate performance. Example:

    python testing_models/evaluate_albert.py \
        --model_path models/albert_base \
        --data_path data/test_balanced.csv \
        --output_path outputs_tests/albert_test_predictions.csv

This will:

- Print Accuracy, F1, Precision, Recall
- Display the confusion matrix
- Save predictions to CSV
- Evaluate the model on:
  - ETHOS dataset filtered to religious comments
  - ETHOS dataset without religious keywords (general hate)

---

## 📦 Installation

Install all dependencies with:

    pip install -r requirements.txt

Key packages used:

- transformers
- torch
- gradio
- langdetect
- deep-translator
- datasets
- scikit-learn
- openai-whisper

---

## 👥 Authors

**EE-559 Deep Learning — Group 12**  
Spring 2025, EPFL

- Benjamin Bahurel  
- Georg Tilman Peter Schwabedal  
- Alessio Zazo

---

## 📄 License

MIT License

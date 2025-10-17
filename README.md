# YouTube Hate Speech — Religion (EE-559 Project) 🎯

This repository contains the code, data pipelines, experiments and results for the EE‑559 course project: automatic detection of hate speech targeting religion in YouTube comments. The goal is to build and evaluate reproducible machine‑learning and deep‑learning models, analyze their behaviour, and discuss ethical considerations for deployment.

Quick links
- Project entrypoint / demo: `app.py`
- Data folder: `data/`
- Training code: `training_models/`
- Testing / evaluation code: `testing_models/`
- Saved/produced artifacts: `models/`, `outputs_tests/`
- Repro environment: `requirements.txt`, `Dockerfile`
- License: `LICENSE`

Why this project? 🤔
- Online platforms host large volumes of user content. Automated detection of hateful and abusive language helps moderation and community safety.
- Religion is a recurrent target of hate; this project focuses on detecting religiously‑targeted hate speech and understanding model limitations.

What’s included 🔍
- Data processing scripts and instructions (see `data/`) — cleaning, tokenization and dataset splits.
- Training scripts and configs in `training_models/` for classical ML and deep-learning baselines.
- Model evaluation and test utilities in `testing_models/`.
- Example inference / demo server: `app.py`.
- Trained model artifacts and experiment outputs (if included) in `models/` and `outputs_tests/`.
- A Dockerfile for reproducible execution environments: `Dockerfile`.
- Python dependencies: `requirements.txt`.

Label taxonomy (example)
- `hate_religion` — explicit hateful/derogatory content targeting religion or religious groups
- `abusive_general` — abusive language not specifically religious
- `neutral/other` — non‑abusive, neutral content
(See `data/` for the exact labeling schema used in this repo.)

Modeling approaches used 🧠
- Preprocessing & feature engineering (TF‑IDF, n‑grams, tokenization)
- Classical ML baselines (Logistic Regression, SVM)
- Neural approaches (LSTM, CNN with pretrained embeddings)
- Transformer fine‑tuning (BERT / RoBERTa when applicable)
- Evaluation with stratified splits, cross‑validation and standard metrics (precision, recall, F1)

How to reproduce the main experiments (short)
1. Clone the repo:
   ```bash
   git clone https://github.com/georgstsc/EE-559---YT-HateSpeech---Religion.git
   cd EE-559---YT-HateSpeech---Religion
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   or build the Docker image:
   ```bash
   docker build -t yt-hatespeech .
   ```
3. Prepare the data:
   - Place/prepare raw comments and run the preprocessing scripts in `data/`.
4. Train a model:
   - Follow examples in `training_models/` (see training scripts / README inside that folder).
5. Run tests/evaluations:
   - Use `testing_models/` scripts to evaluate checkpoints and produce metrics placed in `outputs_tests/`.
6. (Optional) Start demo:
   - Run `python app.py` to start the simple inference/demo (see `app.py` for CLI options and endpoints).

Evaluation & results ✅
- Check the `outputs_tests/` folder for experiment outputs and the `models/` folder for saved checkpoints (if present).
- Notebooks or result summaries (if available) are referenced in the repo; inspect `training_models/` and `testing_models/` for exact commands used to reproduce reported metrics.

Ethical considerations & responsible use ⚖️
- Hate‑speech detection can cause harm if misapplied (false positives censoring legitimate critiques, false negatives missing harmful content).
- Models can reflect dataset and annotation biases; special care must be taken with minority dialects, sarcasm, quotation of abusive content, and context.
- This project documents the labeling guidelines and dataset provenance (see `data/`). Do not deploy models without human‑in‑the‑loop review, transparent thresholds and monitoring.

Acknowledgements 🙏
We worked on this project together; the following people contributed to the data collection, annotation, experiments, and write‑up:

- Benjamin Bahurel
- Georg Tilman Peter Schwabedal
- Alessio Zazo

Thanks also to the EE‑559 course staff and teaching assistants for guidance, and to EPFL for supporting the course and the project.

How you can help / contribute 🤝
- If you find issues or missing steps to reproduce experiments, please open an issue.
- Contributions (improvements, extended label taxonomy, additional experiments) are welcome—open an issue first to discuss.

Contact & citation
- Owner / contact: georgstsc (GitHub)
- Course: EE‑559 (YT Hate Speech — Religion)

License
- See `LICENSE` for licensing details.

---
If you'd like, I can also open a pull request that updates this README instead of pushing directly.
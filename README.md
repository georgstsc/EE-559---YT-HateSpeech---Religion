# EE-559---YT-HateSpeech---Religion


| Student's name | SCIPER |
| -------------- | ------ |
| Alessio Zazo | 328450 |
| Schwabedal Georg Tilman Peter | 328434 |
| Bejamin Bahurel| 326888 |

# Religious Hate Speech Detection with DeBERTa v3

## Introduction

For the project on hate speech identification in the context of the **EE-559 Deep Learning** course, we aim to create a neural network-based model specialized in identifying **religious discrimination**. Our model will be based on **DeBERTa v3**, a state-of-the-art transformer architecture, which will be fine-tuned using large-scale annotated data. The primary goal is to improve detection of subtle and overt forms of **religious hate speech** in online platforms, with an emphasis on comments from sources like YouTube.

We will compare our model's performance to the results presented in recent literature on religious hate speech detection and evaluate whether a general-purpose language model, when fine-tuned on identity-based toxicity data, can match or surpass more domain-specific models.

---

## Literature & Dataset Summary

We conducted a review of key papers and datasets in the domain of religious hate speech detection:

1. **[Aldreabi et al. (2022)](https://par.nsf.gov/servlets/purl/10499720)**  
   - Used Reddit data to identify Islamophobic content via **BERT** and topic modeling.  
   - Demonstrated that transformer models can surface hate speech in specific religious contexts.

2. **[Lee et al. (2023)](https://arxiv.org/html/2311.04916v4)**  
   - Proposed a **graph neural network (GNN)** using relational embeddings to detect Islamophobic hate.  
   - Trained on [**HateXplain**](https://github.com/hate-alert/HateXplain), with an emphasis on **explainability**.

3. **[Liu et al. (2024)](https://arxiv.org/html/2405.03794v1)**  
   - Collected and annotated **anti-Semitic tweets** using a custom annotation pipeline.  
   - Fine-tuned transformer models to improve detection of hate speech.

4. **[Zhao et al. (2023)](https://pmc.ncbi.nlm.nih.gov/articles/PMC10280248)**  
   - Introduced a **multilingual religious hate speech dataset**.  
   - Showed strong **cross-lingual performance** using mBERT and XLM-R.

5. **[Srivastava et al. (2025)](https://aclanthology.org/2025.chipsal-1.5.pdf)**  
   - Created **THAR**, a dataset for code-mixed Hindi-English religious hate speech.  
   - Developed **DweshVaani**, a multilingual transformer model that outperformed monolingual baselines.

6. **[ETHOS Dataset (Hugging Face)](https://huggingface.co/datasets/iamollas/ethos)**  
   - Public dataset with hate speech annotations across identity categories (religion, gender, ethnicity).  
   - [GitHub Repository](https://github.com/intelligence-csd-auth-gr/Ethos-Hate-Speech-Dataset)

7. **[Jigsaw Unintended Bias in Toxicity Classification (Kaggle)](https://www.kaggle.com/competitions/jigsaw-unintended-bias-in-toxicity-classification/data)**  
   - Large dataset of online comments annotated for toxicity and bias.  
   - Widely used in fairness and debiasing research in NLP.

---

## Why DeBERTa v3?

**DeBERTa v3** (Decoding-enhanced BERT with disentangled attention) improves on traditional BERT-based models through the following innovations:

- **Disentangled Attention**  
  Separates **content** and **position** embeddings, leading to better context handling and generalization.

- **Enhanced Mask Decoder**  
  Improves masked language modeling tasks and pretraining efficiency.

- **Larger Pretraining**  
  Trained with more data, longer sequences, and larger batch sizes for better downstream performance.

- **Superior Benchmarks**  
  Outperforms BERT, RoBERTa, and even GPT-2 on several NLP benchmarks, including GLUE and SuperGLUE.

These improvements make DeBERTa v3 a strong choice for hate speech detection, particularly when detecting nuanced and context-dependent content like **religious discrimination**.

---

## Methodology

We will fine-tune **DeBERTa v3** using the [**Civil Comments** dataset](https://huggingface.co/datasets/civil_comments) from Hugging Face.

### Why Civil Comments?

- Over **2 million** online comments  
- Annotated for **toxicity**, **identity references** (religion, race, gender, etc.)  
- Real-world data sourced from public forums and comment sections

### Training Plan

1. Load the pre-trained model `microsoft/deberta-v3-base`.
2. Preprocess Civil Comments:
   - Filter for identity-related mentions (especially religious terms).
   - Binary labeling: *toxic* vs *non-toxic*.
3. Fine-tune DeBERTa v3 using stratified training and validation sets.
4. Evaluate using metrics:
   - **F1-score**, **Precision**, **Recall**, **Accuracy**, **ROC-AUC**

### Evaluation Plan

We will compare our model’s performance with benchmarks from:

- Aldreabi et al. (Reddit + BERT)
- Lee et al. (GNN + HateXplain)
- Liu et al. (Anti-Semitic tweets)
- Srivastava et al. (THAR + DweshVaani)

Optionally, we will test **generalizability** on datasets like ETHOS or THAR, provided the labeling is compatible.

---

## Acknowledgments

This project is part of the **EE-559 Deep Learning** course at EPFL.  
We thank the authors of the datasets and papers reviewed for their contributions to the field.

---

## TODO

- [ ] Data preprocessing script  
- [ ] Model fine-tuning notebook  
- [ ] Evaluation and comparison tables  
- [ ] Final report & slides  

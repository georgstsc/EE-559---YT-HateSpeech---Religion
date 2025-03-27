# EE-559---YT-HateSpeech---Religion


| Student's name | SCIPER |
| -------------- | ------ |
| Alessio Zazo | 328450 |
| Schwabedal Georg Tilman Peter | 328434 |
| Bejamin Bahurel| 326888 |

## Introduction

For the project on hate speech identification in the context of the EE-559 Deep Learning course, we aim to create a neural network-based model specialized in identifying religious discrimination. We plan to build our model upon a pre-trained BERT architecture, as introduced in class, and fine-tune it using the datasets listed below. The model will then be used to assess the presence of religious discrimination in YouTube's comment section. The final part of the report will discuss the results.

### Summary of Literature Review

Researchers have explored religious hate speech detection by first constructing or leveraging specialized datasets. For example, Aldreabi et al. (2022) [1] collected Reddit comments related to Islam and used BERT to semi-automatically filter and label Islamophobic content. Similarly, Liu et al. (2024) [3] collected tweets with potential anti-Semitic language and annotated them using a custom labeling algorithm. Sharma et al. (2024) introduced the THAR dataset targeting religious hate in code-mixed Hindi-English content, while ETHOS [6] and Jigsaw [7] provide large-scale, multi-label datasets covering identity-based hate speech, including religion.

Most papers applied transformer-based models like BERT, RoBERTa, or mBERT to classify hate speech. Srivastava et al. (2025) [5] fine-tuned multilingual models to handle code-mixed religious hate, while Zhao et al. (2023) [4] evaluated cross-lingual performance using XLM-R to detect hate in multiple languages. Lee et al. (2023) [2] explored a graph neural network (GNN) approach, modeling semantic relationships between comments to improve performance and explainability.

In terms of methodology, several studies emphasized annotation quality and interpretability. HateXplain [2] includes crowd-labeled rationales for each annotation, allowing explainable evaluation. Others used topic modeling or keyword filtering during data collection to ensure relevance and balance across religious groups.

Overall, these works contribute to religious hate speech detection by building focused datasets, fine-tuning state-of-the-art models, and addressing multilingual, code-mixed, and explainable modeling challenges. Their combined efforts provide a foundation for detecting nuanced, identity-specific hate across languages and platforms.

---
### Literature & Dataset Review with References

1. [**Aldreabi et al., 2022 (NSF)**](https://par.nsf.gov/servlets/purl/10499720)  
   Collected Reddit comments related to Islam and used BERT-based filtering and topic modeling to identify Islamophobic content.

2. [**Lee et al., 2023 (arXiv)**](https://arxiv.org/html/2311.04916v4)  
   Proposed a graph neural network (GNN) model for detecting Islamophobic hate, using relational embeddings and explainable linkages. Used the [HateXplain dataset](https://github.com/hate-alert/HateXplain).

3. [**Liu et al., 2024 (arXiv)**](https://arxiv.org/html/2405.03794v1)  
   Collected and labeled anti-Semitic tweets using a custom annotation algorithm for fine-tuning transformer-based hate detection models.

4. [**Zhao et al., 2023 (PubMed)**](https://pmc.ncbi.nlm.nih.gov/articles/PMC10280248)  
   Built a multilingual religious hate speech dataset and evaluated cross-lingual transferability using mBERT and XLM-R.

5. [**Srivastava et al., 2025 (ACL Anthology)**](https://aclanthology.org/2025.chipsal-1.5.pdf)  
   Introduced the THAR dataset focused on code-mixed Hindi-English religious hate speech. Developed DweshVaani, a multilingual transformer model.

6. [**ETHOS Dataset (Hugging Face)**](https://huggingface.co/datasets/iamollas/ethos)  
   Contains hate speech across identity categories (e.g., religion, ethnicity, gender). GitHub repo: [Ethos-Hate-Speech-Dataset](https://github.com/intelligence-csd-auth-gr/Ethos-Hate-Speech-Dataset)

7. [**Jigsaw Unintended Bias in Toxicity Classification (Kaggle)**](https://www.kaggle.com/competitions/jigsaw-unintended-bias-in-toxicity-classification/data)  
   Large-scale dataset of online comments annotated for toxicity and identity-related bias, widely used in fairness research for NLP.


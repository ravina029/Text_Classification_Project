# 🎯 Text Classification Project (IMDb Reviews)

This repository provides a **modular NLP pipeline** for binary sentiment classification on the **IMDb movie reviews dataset**.  
The objective is to classify reviews as **positive** or **negative**, while exploring modern research practices in reproducible NLP.  

This project is designed to serve both as a **hands-on engineering framework** and as a **research-oriented case study**, bridging the gap between applied machine learning and academic-level NLP research.

---

## 📚 Motivation & Research Context

Text classification is a foundational task in NLP with applications in:  
- **Opinion Mining** (analyzing customer/product feedback)  
- **Content Moderation** (filtering toxic or harmful content)  
- **Information Retrieval** (ranking relevant results by sentiment/context)  

While large pre-trained language models (LMs) like **BERT** and **GPT** have set new benchmarks, there remain **open research challenges**:  
- How do **classical baselines** compare with transformers on resource efficiency?  
- What preprocessing choices significantly impact downstream performance?  
- How do issues of **bias, fairness, and interpretability** manifest in sentiment classification?  

This repository attempts to provide a **scalable, modular pipeline** where these research questions can be explored systematically.

---

## 📂 Folder Structure

```bash
text_classification_project/
│── data/                         # dataset (auto-downloaded by Hugging Face)
│
│── notebooks/                    # exploratory analysis notebooks
│   └── imdb_eda.ipynb
│
│── src/                          # core source code
│   ├── __init__.py
│   ├── preprocess.py             # data loading & tokenization
│   ├── train.py                  # model training
│   ├── evaluate.py               # evaluation & metrics
│   └── utils.py                  # helper functions (logging, saving models, etc.)
│
│── results/                      # trained models, logs, confusion matrix plot
│
│── configs/                      # hyperparameter configs
│   └── default.yaml
│
│── main.py                       # entry point (ties everything together)
│── requirements.txt              # dependencies
│── README.md                     # project documentation
│── .gitignore                    # to ignore unnecessary files
```

---

## 📊 Dataset Details (IMDb Reviews)

- **Source**: [IMDb dataset](https://ai.stanford.edu/~amaas/data/sentiment/)  
- **Size**: 50,000 labeled reviews (25k train, 25k test, balanced)  
- **Labels**: Binary (Positive = 1, Negative = 0)  
- **Preprocessing**:  
  - Removal of HTML tags  
  - Handling punctuation and contractions  
  - Tokenization (WordPiece for transformers)  
- **Ethical Considerations**:  
  - Language bias (primarily English, mostly U.S. cultural context)  
  - Limited representation of sarcasm/nuance  
  - Does not account for neutrality  

---

## ⚙️ Methodology

We experiment with a progression of models:

1. **Baseline (Classical ML)**  
   - Bag-of-Words + Logistic Regression  
   - TF-IDF + Linear SVM  

2. **Neural Models (RNN/CNN)**  
   - BiLSTM with pre-trained GloVe embeddings  
   - 1D CNN for sentence classification  

3. **Transformer Models (State-of-the-Art)**  
   - DistilBERT (efficient fine-tuning)  
   - BERT-base (benchmark model)  

---

## 🧪 Evaluation Protocol

- **Metrics**: Accuracy, Precision, Recall, F1-score, ROC-AUC  
- **Error Analysis**: Confusion matrices, per-class breakdown  
- **Reproducibility**: Fixed random seeds, config-driven runs  

Stretch research goals:  
- Interpretability via **attention visualization**  
- Fairness evaluation across review length & writing style  

---

## 🚀 Setup Instructions

### 1️⃣ Clone the repository
```bash
git clone https://github.com/ravina029/Text_Classification_Project/tree/main
cd text_classification_project
```

### 2️⃣ Create a virtual environment
```bash
conda create -p ./venv python=3.11 -y   # Linux/Mac
venv\Scripts\activate     # Windows
```

### 3️⃣ Install dependencies
```bash
pip install -r requirements.txt
```

---

## ▶️ Running the Pipeline

### Training
```bash
python main.py --epochs 3 --batch_size 16
```

### Evaluation
```bash
python main.py --evaluate
```

### Custom Configurations
Edit hyperparameters in `configs/default.yaml`.

---

## 📈 Research Contributions

- **Goal**: Reproduce and extend sentiment classification experiments with modular code.  
- **Technical Focus**: Comparative analysis of baselines vs. transformers.  
- **Stretch Goal**: Investigating interpretability, robustness, and efficiency trade-offs.  
- **Skill Gain**:  
  - Deep learning workflow design  
  - Hugging Face integration  
  - Research reproducibility practices  
  - Critical evaluation of models  

---

## 📜 License
This project is licensed under the MIT License.  

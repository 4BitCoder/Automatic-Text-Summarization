# 📘 Automatic Text Summarization – Team 4bitcoders  

![Python](https://img.shields.io/badge/Python-3.10-blue.svg)  
![Pytorch](https://img.shields.io/badge/PyTorch-ML-orange)  
![React](https://img.shields.io/badge/React-Frontend-61DBFB.svg)  
![License](https://img.shields.io/badge/License-MIT-green.svg)  
![Status](https://img.shields.io/badge/Status-In%20Progress-yellow.svg)  

---

## 🚀 Overview  
Automatic Text Summarization is the process of generating concise and coherent summaries from long texts such as research papers, news articles, or reports.  

Our goal as **Team 4bitcoders** is to:  
- Study summarization techniques (extractive, abstractive, hybrid).  
- Build baseline models (TextRank, Seq2Seq, Transformer).  
- Propose an **enhanced solution** combining research + innovation.  
- Deploy a working **web application** for real-world use.  

---

## 🗂️ Folder Structure  

```
[not finalized]
📦 text-summarizer-4bitcoders
├── 📁 backend           # FastAPI/Python backend
│   ├── models/          # ML models (extractive, abstractive)
│   ├── data/            # Preprocessed datasets
│   ├── utils/           # Helper functions
│   ├── main.py          # API entrypoint
│
├── 📁 frontend          # React app
│   ├── components/      # UI components
│   ├── pages/           # App pages
│   ├── App.js
│
├── 📁 research          # Papers, notes, literature review
│   ├── papers/          # Downloaded research PDFs
│   ├── notes.md         # Team notes & findings
│
├── 📁 notebooks         # Jupyter notebooks for prototyping
│   ├── extractive.ipynb
│   ├── abstractive.ipynb
│
├── 📁 evaluation        # ROUGE/BLEU scripts
│
├── README.md            # Project documentation
├── requirements.txt     # Dependencies
└── LICENSE
```

---

## 📚 Resources  

### 🔹 Research Papers
- [TextRank: Bringing Order into Texts (2004)](https://aclanthology.org/W04-3252.pdf)  
- [Abstractive Summarization with Seq2Seq (2015)](https://arxiv.org/abs/1509.00685)  
- [Attention Is All You Need (2017)](https://arxiv.org/abs/1706.03762)  
- [PEGASUS (Google, 2020)](https://arxiv.org/abs/1912.08777)  
- [BERTSUM (2019)](https://arxiv.org/abs/1908.08345)  

### 🔹 Datasets
- [CNN/DailyMail](https://github.com/abisee/cnn-dailymail)  
- [XSum](https://huggingface.co/datasets/xsum)  
- [ArXiv/PubMed](https://huggingface.co/datasets/ccdv/arxiv-summarization)  

### 🔹 Frameworks
- **Python (3.10+)**  
- **PyTorch / TensorFlow**  
- **Hugging Face Transformers**  
- **FastAPI / Flask**  
- **React (Frontend)**  

---

## 🗓 1-Week Timeline (Kickoff Sprint)

| Day | Member 1 | Member 2 | Member 3 | Member 4 |
|-----|----------|----------|----------|----------|
| 1 | Setup repo + research extractive papers | Research abstractive models | Dataset exploration | App boilerplate (React + FastAPI) |
| 2 | Implement TextRank baseline | Seq2Seq tutorial implementation | Preprocess CNN/DailyMail | Frontend upload box + API setup |
| 3 | Refine extractive pipeline | Train/test abstractive small model | Integrate dataset pipeline | API ↔ Frontend connection |
| 4 | Draft hybrid approach | Draft enhanced model idea | Evaluation metrics setup | UI integration |
| 5 | Build extractive module | Build abstractive module | Run ROUGE/BLEU | Connect backend + frontend |
| 6 | Integrate extractive + abstractive | Debug + test | Human eval + fine-tune | UI polish + testing |
| 7 | Write documentation | Write report draft | Dataset report | Demo + README update |

---

## ⚡ Getting Started  

### 🔹 Clone Repo
```bash
git clone https://github.com/4bitcoders/text-summarizer.git
cd text-summarizer
```

### 🔹 Backend Setup
```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload
```

### 🔹 Frontend Setup
```bash
cd frontend
npm install
npm start
```

---

## 🧪 Evaluation  
We will use the following metrics:  
- **ROUGE-1 / ROUGE-2 / ROUGE-L** – Recall-based overlap.  
- **BLEU Score** – Precision-based overlap.  
- **BERTScore** – Embedding similarity.  
- **Human Evaluation** – Readability, Informativeness, Coherence.  

---

## 👨‍💻 Team – 4bitcoders  
- **Member 1** – Research & Extractive Models  
- **Member 2** – Abstractive Models  
- **Member 3** – Data & Evaluation  
- **Member 4** – Web App & Deployment  

(🌀 Roles rotate weekly for equal exposure)  

---

## 🤝 Contribution Guidelines  
1. Create a new branch for your task.  
2. Commit changes with meaningful messages.  
3. Open a pull request → wait for team review.  
4. Merge after at least **1 approval**.  

---

## 📜 License  
This project is licensed under the **GNU General Public License v3.0 (GPL-3.0)** – see [LICENSE](./LICENSE) file for details.  

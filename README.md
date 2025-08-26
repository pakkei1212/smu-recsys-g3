# Collection-Aware Recommender System (CARE)

A course project for **CS608 (Recommender Systems)** at Singapore Management University, Group 3. This repository implements and evaluates a **Collection-Aware Recommender System (CARE)** that shifts from recommending substitutes (similar items) to recommending **complementary collections** (items that go together).

> **Goal:** Move beyond “buy a football → recommend another football” toward “buy a football → recommend jersey + shorts + cleats.”

---

## 🚀 Key Features
- **Collection-Aware Modeling** with FP-Growth, similarity, and co-occurrence analysis.
- **Data Cleaning & Feature Engineering** using custom scripts for large-scale Amazon data.
- **Dataset**: Amazon Reviews 2023 (Sports & Outdoors) – 10.3M reviews, 1.6M items.
- **Notebooks** for preprocessing, exploration, and model implementation.

---

## 📁 Project Structure
```
smu-recsys-g3/
├─ Code and Files for Collection Model/
│  └─ RecSys_Collection_Group3_Project_Final.ipynb   # CARE model notebook
├─ Codes for Data Processing/
│  ├─ Data Processing - Access 9M rows_filtered to 1.5M rows/
│  │  ├─ ingest_amazon_data.py
│  │  ├─ merge_parquet_chunks.py
│  │  ├─ pipeline_config.yaml
│  │  ├─ docker-compose.yml, Dockerfile
│  ├─ Data Processing - Collections Maker/
│  │  └─ collections_maker.ipynb
│  └─ Data Processing - Top 5 Categories Only/
│     └─ EDA & Generate Top 5 Categories Dataset.ipynb
├─ Baseline_Models.ipynb
└─ README.md
```

---

## 🧰 Requirements
- Python 3.10+
- Core libraries: `numpy`, `pandas`, `scikit-learn`, `tqdm`
- For Spark merging: `pyspark`
- Jupyter Notebook / JupyterLab

Install:
```bash
pip install -r requirements.txt
```

---

## 🗂️ Data (Not exists in this github)
Create a local `data/` directory and place the **Amazon Sports & Outdoors** raw files there:
```
smu-recsys-g3/
└─ data/
   ├─ Sports_and_Outdoors.jsonl.gz
   └─ Meta_Sports_and_Outdoors.jsonl.gz
```
Configuration:
```
pipeline_config.yaml   # toggles pipeline stages
```
Outputs:
```
data/processed/
   ├─ Temporary_Chunks_*
   ├─ final_joined.parquet
   └─ logs/
```

---

## 🏗️ Data Processing
**Location:** `Codes for Data Processing/Data Processing - Access 9M rows_filtered to 1.5M rows/`

### 1) Run ingestion pipeline (local, pandas-based)
```bash
python "Codes for Data Processing/Data Processing - Access 9M rows_filtered to 1.5M rows/ingest_amazon_data.py"
```
- Reads raw files and writes outputs to `data/processed/`.
- Controlled via `pipeline_config.yaml`.

### 2) Merge parquet chunks with PySpark (optional)
```bash
python "Codes for Data Processing/Data Processing - Access 9M rows_filtered to 1.5M rows/merge_parquet_chunks.py"   --temp-dir data/processed/Temporary_Chunks_Sports_and_Outdoors   --output   data/processed/9m_reviews_merged.parquet   --partitions 10   --memory 4g
```

### 3) Collections Maker notebook
```
Codes for Data Processing/Data Processing - Collections Maker/collections_maker.ipynb
```
- Prototype for creating and testing collection-based recommendation logic.
- Useful for validating FP-Growth outputs and co-occurrence signals.

### 4) Top 5 Categories EDA notebook
```
Codes for Data Processing/Data Processing - Top 5 Categories Only/EDA & Generate Top 5 Categories Dataset.ipynb
```
- Explores and generates a dataset focusing on the **top 5 categories**.
- Includes cleaning, preprocessing, and feature engineering specific to these subsets.

---

## 📚 Baseline Models
Open `Baseline_Models.ipynb` for experiments with MF, WMF, BPR, EASE.
- Provides comparative metrics (Recall@K, NDCG@K) against CARE.

---

## 🧩 CARE Collection Model
Implemented in:
```
Code and Files for Collection Model/RecSys_Collection_Group3_Project_Final.ipynb
```
Run interactively:
```bash
jupyter notebook "Code and Files for Collection Model/RecSys_Collection_Group3_Project_Final.ipynb"
```
Or headless:
```bash
jupyter nbconvert --to notebook --inplace --execute   "Code and Files for Collection Model/RecSys_Collection_Group3_Project_Final.ipynb"
```

---

## 🙌 Contributors
- **Group 3:** Chia Wai Mun, Lee How Chih, Calvin Li, Aditya Vijay, Yip Pak Kei

---

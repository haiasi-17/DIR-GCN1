# Dir-GCN Fraud Detection Demo

This repo contains:

- **Baseline Dir-GCN** and **Enhanced Dir-GCN (Gated)** models  
- Experiments on three fraud datasets:
  - **Synthetic Fraud** (`fraud-syn`)
  - **Online Payments** (`online-payments`)
  - **Elliptic Bitcoin** (`elliptic`)
- A **Streamlit dashboard** (`app.py`) to:
  - Visualize **time/memory/runtime (Problem 1)**
  - Compare **accuracy / F1 / precision / recall (Problem 2)**
  - Show **redundancy, caching, LCS masking (Problem 3)**
  - Run **live inference** on new Online Payments transactions  
    (upload CSV or manual single-transaction input)

---

## 1. Prerequisites

### Recommended

- **OS:** Linux / macOS / WSL
- **Python:** **3.10** (important)
- **conda**

> ⚠️ Torch 1.12.1 does **not** support Python 3.13+.  
> Use Python **3.10** to avoid “No matching distribution found for torch==1.12.1”.

---

## 2. Clone the Repository

```bash
git clone <your-repo-url>.git
cd gnn
```

---

## 3. Create and Activate Conda Environment

```bash
conda create -n dirgnn python=3.10 -y
conda activate dirgnn
```
```

---

## 4. Install Dependencies

### Option A — use provided requirements (recommended)

Contents of `requirements.txt`:

```txt
streamlit==1.52.2
pandas
openpyxl

torch==1.12.1
torch-geometric==2.1.0

-f https://data.pyg.org/whl/torch-1.12.1+cpu.html
torch-scatter==2.0.9

-f https://data.pyg.org/whl/torch-1.12.1+cpu.html
torch-sparse==0.6.16

pytorch-lightning>=2.2
```

Install:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

or with **uv**:

```bash
uv pip install -r requirements.txt
```

> These URLs install compatible **PyG wheels for CPU**.

---

### Option B — newer PyTorch/PyG stack (optional)

You can upgrade Torch and PyG if you wish, but ensure version compatibility:
https://github.com/pyg-team/pytorch_geometric#installation

---

## 5. Dataset Setup

### 5.1 Synthetic Fraud (`fraud-syn`)
No downloads needed — generated automatically.

Output appears under:

```text
data/syn-fraud/processed/fraud_data_v4.pt
```

---

### 5.2 Online Payments (`online-payments`)

1. Download PaySim / Online Payments Fraud dataset from Kaggle
2. Create folders:

```bash
mkdir -p data/online_payments/raw
```

3. Save file as:

```text
data/online_payments/raw/online_payments.csv
```

---

### 5.3 Elliptic Bitcoin (`elliptic`)

Download:

- `elliptic_txs_features.csv`
- `elliptic_txs_classes.csv`
- `elliptic_txs_edgelist.csv`

Create:

```bash
mkdir -p data/elliptic/raw
```

Place files there.

Processed file will be created:

```text
data/elliptic/processed/elliptic.pt
```

---

## 6. Training Models (optional)

Skip if you already have a `results/` directory with checkpoints.

Run:

```bash
python -m src.compare_models --dataset fraud-syn
python -m src.compare_models --dataset online-payments
python -m src.compare_models --dataset elliptic
```

Results structure:

```text
results/
  <dataset>/
    <exp_id>/
      dir-gcn/
      dir-gcn-gated/
      predictions/
      runtime/
      plots/
      problem3_metrics/
```

---

## 7. Run the Streamlit App

```bash
streamlit run app.py
```

Open the printed **local URL** in your browser.

---

## 8. Using the Dashboard

### Sidebar controls

- Select **Dataset**
- Select **Configuration (C1–C4)**
- Select **Model (Baseline vs Enhanced)**

---

### Section 1 – Problem 1
- training time
- testing time
- memory usage
- inference runtime

---

### Section 2 – Problem 2
- accuracy
- precision
- recall
- F1-score
- ROC / PR curves
- confusion matrix

---

### Section 3 – Problem 3
- redundant / recurring transactions
- cache hit ratio
- saved message aggregations
- LCS masking metrics

---

### Section 4 – Predictions & Live Inference

#### Tab 1 — existing predictions
Shows:
- `y_true`
- `y_pred`
- class probabilities

#### Tab 2 — live inference

- **Online Payments only**

Supports:
- **Upload CSV**
- **Manual transaction entry**

Outputs:
- `y_pred`
- `prob_fraud`

Synthetic Fraud & Elliptic currently:
- show consistent UI
- do not expose manual/CSV entry

---

## 9. Common Errors & Fixes

### `torch==1.12.1 not found`
Use:

```bash
python=3.10
```

### `ModuleNotFoundError: pytorch_lightning`
Install:

```bash
pip install pytorch-lightning
```

---

## 10. TL;DR Quickstart

```bash
git clone <repo>
cd gnn

conda create -n dirgnn python=3.10 -y
conda activate dirgnn

pip install -r requirements.txt

streamlit run app.py
```

Done 🚀

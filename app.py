import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

import torch
from torch_geometric.data import Data
from torch_geometric.utils import degree
import torch.nn.functional as F

from src.model import GNN

# ============================================================
# Page config
# ============================================================
st.set_page_config(
    page_title="Dir-GCN Fraud Detection Demo",
    layout="wide",
)

st.title("Dir-GCN vs Enhanced Dir-GCN – Fraud Detection Results")

# ============================================================
# Mappings
# ============================================================

DATASETS = {
    "Synthetic Fraud": "fraud-syn",
    "Online Payments": "online-payments",
    "Elliptic Bitcoin": "elliptic",
}

CONFIGS = ["C1", "C2", "C3", "C4"]

MODELS = {
    "Baseline Dir-GCN": "dir-gcn",
    "Enhanced Dir-GCN (Gated)": "dir-gcn-gated",
}

# node feature dims per dataset (from dataset code)
NUM_FEATURES = {
    "fraud-syn": 14,
    "online-payments": 12,
    "elliptic": 166,  # elliptic_txs_features has 166 feature columns
}

NUM_CLASSES = {
    "fraud-syn": 2,
    "online-payments": 2,
    "elliptic": 2,
}


def get_exp_id(config: str) -> str:
    return {
        "C1": "hdim32_L3_drop0p5_lr0p001",
        "C2": "hdim32_L3_drop0p6_lr0p0005",
        "C3": "hdim64_L3_drop0p5_lr0p001",
        "C4": "hdim64_L3_drop0p6_lr0p0005",
    }[config]


def parse_exp_id(exp_id: str):
    """
    Parse strings like 'hdim32_L3_drop0p5_lr0p001'
    → hidden_dim, num_layers, dropout, lr
    """
    parts = exp_id.split("_")
    hdim = int(parts[0].replace("hdim", ""))
    num_layers = int(parts[1].replace("L", ""))
    drop_str = parts[2].replace("drop", "")
    lr_str = parts[3].replace("lr", "")
    dropout = float(drop_str.replace("p", "."))
    lr = float(lr_str.replace("p", "."))
    return hdim, num_layers, dropout, lr


# ============================================================
# Generic helpers (JSON, summaries, etc.)
# ============================================================


def load_json(path: Path):
    if not path.exists():
        return None
    with open(path, "r") as f:
        return json.load(f)


def load_summary(base_dir: Path, conv_type: str):
    """
    summary.json produced by compare_models.py under:
        results/<dataset>/<exp_id>/<conv_type>/summary.json
    """
    path = base_dir / conv_type / "summary.json"
    return load_json(path)


def load_runtime_info(base_dir: Path, conv_type: str):
    """
    Runtime JSON produced by create_diagnostics_plots under:
        results/<dataset>/<exp_id>/runtime/<conv_type>_best_run_inference_runtime.json
    """
    path = base_dir / "runtime" / f"{conv_type}_best_run_inference_runtime.json"
    return load_json(path)


def load_predictions_df(base_dir: Path, conv_type: str):
    """
    Prediction CSV produced by create_diagnostics_plots under:
        results/<dataset>/<exp_id>/predictions/<conv_type>_best_run_predictions.csv
    """
    path = base_dir / "predictions" / f"{conv_type}_best_run_predictions.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


def load_problem3_metrics(base_dir: Path, conv_type: str):
    """
    Problem 3 metrics for redundancy / caching:

    CSV:   results/.../problem3_metrics/<conv_type>_best_run_problem3_metrics.csv
    JSON:  results/.../problem3_metrics/<conv_type>_best_run_problem3_summary.json
    """
    csv_path = (
        base_dir / "problem3_metrics" / f"{conv_type}_best_run_problem3_metrics.csv"
    )
    json_path = (
        base_dir / "problem3_metrics" / f"{conv_type}_best_run_problem3_summary.json"
    )

    df = pd.read_csv(csv_path) if csv_path.exists() else None
    summary = load_json(json_path)
    return df, summary


def df_download_button(df: pd.DataFrame, filename: str, label: str):
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label=label,
        data=csv_bytes,
        file_name=filename,
        mime="text/csv",
    )


def load_overall_plot(dataset_name: str, filename: str) -> Path | None:
    """
    Plots summarizing all configs live under:
      results/<dataset>/overall/<filename>
    """
    path = Path("results") / dataset_name / "overall" / filename
    return path if path.exists() else None


def compute_eval_metrics_from_df(df: pd.DataFrame):
    """
    Compute accuracy, precision, recall, and F1 (macro) from a prediction DataFrame.
    Expects columns: 'y_true' and 'y_pred'.
    """
    if "y_true" not in df.columns or "y_pred" not in df.columns:
        return None

    mask = ~pd.isna(df["y_true"])
    if mask.sum() == 0:
        return None

    y_true = df.loc[mask, "y_true"].astype(int).values
    y_pred = df.loc[mask, "y_pred"].astype(int).values

    acc = (y_true == y_pred).mean()

    classes = np.unique(y_true)
    precisions, recalls, f1s = [], [], []

    for c in classes:
        tp = np.logical_and(y_pred == c, y_true == c).sum()
        fp = np.logical_and(y_pred == c, y_true != c).sum()
        fn = np.logical_and(y_pred != c, y_true == c).sum()

        prec_c = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec_c = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        if prec_c + rec_c > 0:
            f1_c = 2 * prec_c * rec_c / (prec_c + rec_c)
        else:
            f1_c = 0.0

        precisions.append(prec_c)
        recalls.append(rec_c)
        f1s.append(f1_c)

    precision_macro = float(np.mean(precisions))
    recall_macro = float(np.mean(recalls))
    f1_macro = float(np.mean(f1s))

    return {
        "accuracy": float(acc),
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "f1_macro": f1_macro,
    }


# ============================================================
# Model loading for live inference
# ============================================================


@st.cache_resource
def load_trained_model(dataset_name: str, conv_type: str, exp_id: str):
    """
    Load a trained GNN from the Lightning checkpoint and return (model, device).

    - Finds a .ckpt under:
        results/<dataset>/<exp_id>/<conv_type>/.../*.ckpt
    - Inspects the checkpoint to infer whether Jumping Knowledge (JK)
      was used and in which mode ("cat" vs "max").
    - Builds a GNN with matching architecture and loads the weights.
    """
    if dataset_name not in NUM_FEATURES:
        raise RuntimeError(f"No num_features configured for dataset '{dataset_name}'.")

    num_features = NUM_FEATURES[dataset_name]
    num_classes = NUM_CLASSES[dataset_name]
    hidden_dim, num_layers, dropout, lr = parse_exp_id(exp_id)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---------- locate checkpoint ----------
    ckpt_root = Path("results") / dataset_name / exp_id / conv_type
    if not ckpt_root.exists():
        raise FileNotFoundError(f"No checkpoint directory at {ckpt_root}")

    ckpts = sorted(ckpt_root.glob("**/*.ckpt"))
    if not ckpts:
        raise FileNotFoundError(f"No .ckpt files found under {ckpt_root}")
    ckpt_path = ckpts[-1]  # last one = best/latest run

    ckpt = torch.load(ckpt_path, map_location=device)
    raw_state = ckpt.get("state_dict", ckpt)

    # Strip leading "model." if present (Lightning wrapper)
    plain_state = {}
    for k, v in raw_state.items():
        if k.startswith("model."):
            plain_state[k[len("model.") :]] = v
        else:
            plain_state[k] = v

    # ---------- infer Jumping Knowledge mode from shapes ----------
    # Last conv layer weight: convs.{L-1}.lin_src_to_dst.weight  -> [out_dim, in_dim]
    last_conv_key = f"convs.{num_layers - 1}.lin_src_to_dst.weight"
    if last_conv_key not in plain_state:
        raise RuntimeError(
            f"Could not find '{last_conv_key}' in checkpoint state_dict."
        )

    last_w = plain_state[last_conv_key]
    last_out_dim = last_w.shape[0]  # out_features of last conv

    # If last_out_dim == hidden_dim -> JK was used; otherwise out_dim == num_classes
    jk_mode = None
    if last_out_dim == hidden_dim:
        # Some kind of JumpingKnowledge was used. Decide between "cat" and "max"
        lin_w = plain_state.get("lin.weight", None)
        if lin_w is not None:
            in_features = lin_w.shape[1]
            if in_features == hidden_dim * num_layers:
                jk_mode = "cat"
            else:
                # could be "max" / "lstm" etc.; we treat all non-cat as "max"
                jk_mode = "max"
        else:
            # Conservative fallback: still assume JK but "max"
            jk_mode = "max"
    else:
        jk_mode = None  # no JK, last conv already outputs num_classes

    # ---------- build model with matching architecture ----------
    gnn = GNN(
        num_features=num_features,
        num_classes=num_classes,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
        conv_type=conv_type,
        jumping_knowledge=jk_mode,  # <- key fix
        normalize=False,
        alpha=0.5,
        learn_alpha=False,
        lcs_threshold=0.0,
        enable_lcs_masking=(conv_type == "dir-gcn-gated"),
    ).to(device)

    # ---------- load weights ----------
    missing, unexpected = gnn.load_state_dict(plain_state, strict=False)

    # Optional: you can uncomment this to debug if needed
    # print("Missing keys:", missing)
    # print("Unexpected keys:", unexpected)

    gnn.eval()
    return gnn, device


def gnn_predict_probabilities(model: GNN, device, data: Data):
    """
    Run a forward pass on the given graph and return (y_pred, prob_fraud) per node.
    """
    data = data.to(device)
    with torch.no_grad():
        log_probs = model(data.x, data.edge_index, batch_nodes=None)
        probs = torch.exp(log_probs)

    prob_fraud = probs[:, 1]  # class 1 = fraud
    y_pred = prob_fraud > 0.7

    return y_pred.cpu().numpy(), prob_fraud.cpu().numpy()


# ============================================================
# Graph builder for Online Payments (for NEW CSV/manual input)
# ============================================================


def build_online_payments_graph_from_df(df: pd.DataFrame) -> Data:
    """
    Build a PyG Data object from a *new* Online Payments CSV,
    using the same feature logic as OnlinePaymentsDataset.process().
    """
    required_cols = {
        "step",
        "type",
        "amount",
        "nameOrig",
        "oldbalanceOrg",
        "newbalanceOrig",
        "nameDest",
        "oldbalanceDest",
        "newbalanceDest",
        "isFraud",
        "isFlaggedFraud",
    }
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in uploaded CSV: {missing}")

    # Build node index
    all_accounts = pd.Index(pd.concat([df["nameOrig"], df["nameDest"]]).unique())
    acc_to_idx = {acc: i for i, acc in enumerate(all_accounts)}
    num_nodes = len(all_accounts)

    src_idx = df["nameOrig"].map(acc_to_idx).to_numpy()
    dst_idx = df["nameDest"].map(acc_to_idx).to_numpy()

    edge_index = torch.tensor(np.vstack([src_idx, dst_idx]), dtype=torch.long)

    # Edge features
    amount = torch.tensor(df["amount"].to_numpy(), dtype=torch.float32)
    step = torch.tensor(df["step"].to_numpy(), dtype=torch.float32)
    is_flagged = torch.tensor(df["isFlaggedFraud"].to_numpy(), dtype=torch.float32)

    log_amount = torch.log1p(amount).unsqueeze(-1)
    step_norm = (step / (step.max() + 1e-6)).unsqueeze(-1)

    type_cat = df["type"].astype("category")
    type_codes = torch.tensor(type_cat.cat.codes.to_numpy(), dtype=torch.long)
    num_types = len(type_cat.cat.categories)
    type_one_hot = F.one_hot(type_codes, num_classes=num_types).float()

    is_flagged_feat = is_flagged.unsqueeze(-1)

    edge_attr = torch.cat(
        [log_amount, step_norm, type_one_hot, is_flagged_feat], dim=-1
    )

    # Node features
    src = edge_index[0]
    dst = edge_index[1]

    out_deg = degree(src, num_nodes=num_nodes, dtype=torch.float32)
    in_deg = degree(dst, num_nodes=num_nodes, dtype=torch.float32)

    total_out_amt = torch.zeros(num_nodes, dtype=torch.float32)
    total_in_amt = torch.zeros(num_nodes, dtype=torch.float32)
    total_out_amt.index_add_(0, src, amount)
    total_in_amt.index_add_(0, dst, amount)

    avg_out_amt = total_out_amt / (out_deg + 1e-6)
    avg_in_amt = total_in_amt / (in_deg + 1e-6)

    is_fraud_edge = torch.tensor(df["isFraud"].to_numpy(), dtype=torch.float32)

    fraud_out_count = torch.zeros(num_nodes, dtype=torch.float32)
    fraud_in_count = torch.zeros(num_nodes, dtype=torch.float32)
    fraud_out_count.index_add_(0, src, is_fraud_edge)
    fraud_in_count.index_add_(0, dst, is_fraud_edge)

    fraud_out_ratio = fraud_out_count / (out_deg + 1e-6)
    fraud_in_ratio = fraud_in_count / (in_deg + 1e-6)

    noise_feat = torch.randn(num_nodes, 4)

    x = torch.cat(
        [
            out_deg.unsqueeze(-1),
            in_deg.unsqueeze(-1),
            total_out_amt.unsqueeze(-1),
            total_in_amt.unsqueeze(-1),
            avg_out_amt.unsqueeze(-1),
            avg_in_amt.unsqueeze(-1),
            fraud_out_ratio.unsqueeze(-1),
            fraud_in_ratio.unsqueeze(-1),
            noise_feat,
        ],
        dim=-1,
    )

    fraud_out_flag = fraud_out_count > 0.0
    y = fraud_out_flag.long()

    data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=y,
        num_nodes=num_nodes,
    )

    # Keep for mapping back
    data._account_index = all_accounts
    return data


def run_online_payments_inference_from_df(
    df: pd.DataFrame, dataset_name: str, conv_type: str, exp_id: str
) -> pd.DataFrame:
    """
    Use the trained model to predict fraud *per account*, then broadcast
    origin-account predictions back to each transaction.
    """
    model, device = load_trained_model(dataset_name, conv_type, exp_id)
    data = build_online_payments_graph_from_df(df)

    y_pred_nodes, prob_fraud_nodes = gnn_predict_probabilities(model, device, data)

    account_names = data._account_index
    pred_map = {
        acc: (int(y_pred_nodes[i]), float(prob_fraud_nodes[i]))
        for i, acc in enumerate(account_names)
    }

    y_pred_rows = []
    prob_rows = []
    for name in df["nameOrig"]:
        p, pr = pred_map[name]
        y_pred_rows.append(p)
        prob_rows.append(pr)

    out = df.copy()
    out["y_pred"] = y_pred_rows
    out["prob_fraud"] = prob_rows
    return out


def run_online_payments_inference_from_single(
    amount: float,
    step: int,
    tx_type: str,
    is_flagged: bool,
    dataset_name: str,
    conv_type: str,
    exp_id: str,
) -> pd.DataFrame:
    df = pd.DataFrame(
        [
            {
                "step": step,
                "type": tx_type,
                "amount": amount,
                "isFlaggedFraud": int(is_flagged),
                "nameOrig": "ACC_ORIG",
                "oldbalanceOrg": 0.0,
                "newbalanceOrig": 0.0,
                "nameDest": "ACC_DEST",
                "oldbalanceDest": 0.0,
                "newbalanceDest": 0.0,
                "isFraud": 0,
            }
        ]
    )
    return run_online_payments_inference_from_df(
        df, dataset_name=dataset_name, conv_type=conv_type, exp_id=exp_id
    )


# ============================================================
# Sidebar controls
# ============================================================

st.sidebar.header("Configuration")

dataset_label = st.sidebar.selectbox("Select dataset", list(DATASETS.keys()))
config = st.sidebar.selectbox("Select configuration", CONFIGS)
model_label = st.sidebar.selectbox("Select model", list(MODELS.keys()))

dataset_name = DATASETS[dataset_label]
conv_type = MODELS[model_label]
exp_id = get_exp_id(config)

base_dir = Path("results") / dataset_name / exp_id

# ============================================================
# Load summaries + runtime once (used in multiple sections)
# ============================================================

baseline_summary = load_summary(base_dir, "dir-gcn")
enh_summary = load_summary(base_dir, "dir-gcn-gated")

runtime_baseline = load_runtime_info(base_dir, "dir-gcn")
runtime_enh = load_runtime_info(base_dir, "dir-gcn-gated")

# ============================================================
# Current selection info
# ============================================================

st.markdown("### Current Selection")
st.write(f"- **Dataset**: `{dataset_name}`")
st.write(f"- **Configuration**: `{config}` → `{exp_id}`")
st.write(f"- **Model**: `{model_label}` (`{conv_type}`)")
st.write(f"- **Base directory**: `{base_dir}`")

st.caption(
    "Note: Values displayed in the tables are formatted (rounded) for readability. "
    "The exported CSV files keep full numerical precision, which is why you may see "
    "slightly longer decimal values there (e.g., 0.00007 vs 0.000078)."
)

st.markdown("---")

# ============================================================
# SECTION 1 – Problem 1 vs Solution 1 (Time, Memory, Runtime)
# ============================================================

st.header("Problem 1 vs Solution 1 – Time, Memory, Runtime")

st.write(
    "Problem 1 focuses on the **computational cost** of the models. "
    "Here we compare **training/testing time, total runtime, memory usage**, "
    "and **average inference time per forward pass** between the Baseline and "
    "Enhanced Dir-GCN."
)

if baseline_summary is None or enh_summary is None:
    st.warning(
        "One or both `summary.json` files are missing. "
        "Make sure you have run `compare_models.py` for this dataset & config."
    )
else:
    time_mem_metrics = ["train_time", "test_time", "total_time", "mem_mb"]
    rows_tm = []
    for m in time_mem_metrics:
        b_mean, b_std = baseline_summary[m]
        e_mean, e_std = enh_summary[m]
        rows_tm.append(
            {
                "Metric": m,
                "Baseline mean": b_mean,
                "Baseline std": b_std,
                "Enhanced mean": e_mean,
                "Enhanced std": e_std,
            }
        )
    df_time_mem = pd.DataFrame(rows_tm)

    st.subheader("Training / Testing Time and Memory")
    st.dataframe(df_time_mem, use_container_width=True)

    df_download_button(
        df_time_mem,
        filename=f"time_memory_metrics_{dataset_name}_{config}.csv",
        label="Download Time & Memory Metrics (CSV)",
    )

    st.subheader("Inference Runtime (Average Forward Pass)")

    if runtime_baseline is None or runtime_enh is None:
        st.info(
            "Runtime information not found. Make sure the diagnostics step "
            "(`create_diagnostics_plots`) ran successfully."
        )
    else:
        t_base = runtime_baseline.get("avg_inference_time_seconds", None)
        t_enh = runtime_enh.get("avg_inference_time_seconds", None)

        cols_rt = st.columns(3)
        with cols_rt[0]:
            st.metric("Baseline inference time (s)", f"{t_base:.6f}")
        with cols_rt[1]:
            st.metric("Enhanced inference time (s)", f"{t_enh:.6f}")
        with cols_rt[2]:
            speedup = (
                t_base / t_enh if (t_base is not None and t_enh is not None) else 0.0
            )
            st.metric("Enhanced speedup vs Baseline (×)", f"{speedup:.3f}")

    st.subheader("Time & Memory vs Config (All Configurations)")

    overall_mem = load_overall_plot(dataset_name, "memory_by_config.png")
    overall_total_time = load_overall_plot(dataset_name, "total_time_by_config.png")

    cols_overall_tm = st.columns(2)
    with cols_overall_tm[0]:
        if overall_mem is not None:
            st.image(str(overall_mem), use_container_width=True)
            st.caption("Memory usage (MB) vs Config – Baseline vs Enhanced")
        else:
            st.info("No `memory_by_config.png` found in the overall directory.")

    with cols_overall_tm[1]:
        if overall_total_time is not None:
            st.image(str(overall_total_time), use_container_width=True)
            st.caption("Total runtime (s) vs Config – Baseline vs Enhanced")
        else:
            st.info("No `total_time_by_config.png` found in the overall directory.")

st.markdown("---")

# ============================================================
# SECTION 2 – Problem 2 vs Solution 2 (Accuracy, F1, Precision, Recall)
# ============================================================

st.header("Problem 2 vs Solution 2 – Classification Performance")

st.write(
    "Problem 2 focuses on **predictive performance**. "
    "Here we compare **validation and test metrics** — Accuracy, F1-score, "
    "Precision, and Recall — between the Baseline Dir-GCN and the Enhanced "
    "Dir-GCN (Gated)."
)

if baseline_summary is None or enh_summary is None:
    st.warning(
        "Classification summaries are unavailable because `summary.json` files "
        "are missing."
    )
else:
    metrics_for_problem2 = ["val_f1", "test_acc", "test_f1", "test_prec", "test_rec"]

    rows = []
    for m in metrics_for_problem2:
        b_mean, b_std = baseline_summary[m]
        e_mean, e_std = enh_summary[m]
        rows.append(
            {
                "Metric": m,
                "Baseline mean": b_mean,
                "Baseline std": b_std,
                "Enhanced mean": e_mean,
                "Enhanced std": e_std,
            }
        )

    df_perf = pd.DataFrame(rows)
    st.subheader("Classification Metrics (Baseline vs Enhanced)")
    st.dataframe(df_perf, use_container_width=True)
    df_download_button(
        df_perf,
        filename=f"classification_metrics_{dataset_name}_{config}.csv",
        label="Download Classification Metrics (CSV)",
    )

    st.subheader("Performance vs Config (All Configurations)")

    overall_val_f1 = load_overall_plot(dataset_name, "val_f1_by_config.png")
    overall_test_acc = load_overall_plot(dataset_name, "test_acc_by_config.png")
    overall_test_f1 = load_overall_plot(dataset_name, "test_f1_by_config.png")
    overall_test_prec = load_overall_plot(dataset_name, "test_prec_by_config.png")
    overall_test_rec = load_overall_plot(dataset_name, "test_rec_by_config.png")

    cols_row1 = st.columns(2)
    with cols_row1[0]:
        if overall_val_f1 is not None:
            st.image(str(overall_val_f1), use_container_width=True)
            st.caption("Best Validation F1 vs Config – Baseline vs Enhanced")
        else:
            st.info("No `val_f1_by_config.png` found.")

    with cols_row1[1]:
        if overall_test_acc is not None:
            st.image(str(overall_test_acc), use_container_width=True)
            st.caption("Test Accuracy vs Config – Baseline vs Enhanced")
        else:
            st.info("No `test_acc_by_config.png` found.")

    cols_row2 = st.columns(3)
    with cols_row2[0]:
        if overall_test_f1 is not None:
            st.image(str(overall_test_f1), use_container_width=True)
            st.caption("Test F1-score vs Config – Baseline vs Enhanced")
        else:
            st.info("No `test_f1_by_config.png` found.")

    with cols_row2[1]:
        if overall_test_prec is not None:
            st.image(str(overall_test_prec), use_container_width=True)
            st.caption("Test Precision vs Config – Baseline vs Enhanced")
        else:
            st.info("No `test_prec_by_config.png` found.")

    with cols_row2[2]:
        if overall_test_rec is not None:
            st.image(str(overall_test_rec), use_container_width=True)
            st.caption("Test Recall vs Config – Baseline vs Enhanced")
        else:
            st.info("No `test_rec_by_config.png` found.")

    st.markdown("#### Diagnostic Curves & Confusion Matrix (Selected Model)")

    cols = st.columns(2)
    with cols[0]:
        st.markdown("##### ROC Curve")
        roc_path = (
            base_dir / "plots" / conv_type / f"{conv_type}_best_run_roc_curve.png"
        )
        if roc_path.exists():
            st.image(str(roc_path), use_container_width=True)
        else:
            st.info("No ROC curve generated for this model.")

    with cols[1]:
        st.markdown("##### Precision–Recall Curve")
        pr_path = base_dir / "plots" / conv_type / f"{conv_type}_best_run_pr_curve.png"
        if pr_path.exists():
            st.image(str(pr_path), use_container_width=True)
        else:
            st.info("No Precision–Recall curve generated for this model.")

    cm_path = (
        base_dir / "plots" / conv_type / f"{conv_type}_best_run_confusion_matrix.png"
    )
    st.markdown("##### Confusion Matrix")
    if cm_path.exists():
        st.image(str(cm_path), use_container_width=True)
    else:
        st.info("No confusion matrix generated for this model.")

st.markdown("---")

# ============================================================
# SECTION 3 – Problem 3 vs Solution 3 (Redundancy, Caching, LCS)
# ============================================================

st.header("Problem 3 vs Solution 3 – Redundancy, Caching, LCS Masking")

st.write(
    "Problem 3 studies **structural redundancy** in the transaction graph "
    "(recurring edges / repeated transactions) and how the Enhanced model "
    "reuses past computations.\n\n"
    "- **Cache Hit Ratio**: how often duplicate (recurring) edges allow reuse\n"
    "  of previous computations instead of recomputing messages.\n"
    "- **Number of Aggregations Executed**: naive vs actual aggregation counts.\n"
    "- **Saved Ratio**: proportion of aggregations avoided by structural hashing\n"
    "  and LCS masking.\n\n"
    "These metrics are computed during the forward passes of the Enhanced "
    "Dir-GCN and summarized per layer and globally."
)

p3_df_enh, p3_summary_enh = load_problem3_metrics(base_dir, "dir-gcn-gated")

if p3_df_enh is None or p3_summary_enh is None:
    st.info(
        "Problem 3 metrics for the Enhanced model are not available. "
        "Ensure that `collect_problem3_metrics` and diagnostics ran successfully."
    )
else:
    st.subheader("Global Metrics (Enhanced Dir-GCN)")

    total_naive = p3_summary_enh.get("total_naive_aggregations", 0.0)
    total_actual = p3_summary_enh.get("total_actual_aggregations", 0.0)
    total_saved = p3_summary_enh.get("total_saved_aggregations", 0.0)
    saved_ratio = p3_summary_enh.get("total_saved_ratio", 0.0)
    total_edges = p3_summary_enh.get("total_edges", 0)
    total_rec = p3_summary_enh.get("total_recurring_edges", 0)
    cache_hit_ratio = p3_summary_enh.get("global_cache_hit_ratio", 0.0)

    cols_p3 = st.columns(3)
    with cols_p3[0]:
        st.metric("Total naive aggregations", f"{total_naive:.0f}")
        st.metric("Total actual aggregations", f"{total_actual:.0f}")
    with cols_p3[1]:
        st.metric("Aggregations saved", f"{total_saved:.0f}")
        st.metric("Saved ratio", f"{saved_ratio:.3f}")
    with cols_p3[2]:
        st.metric("Total edges", f"{total_edges}")
        st.metric("Recurring edges", f"{total_rec}")
        st.metric("Cache hit ratio", f"{cache_hit_ratio:.3f}")

    st.subheader("Per-layer Metrics (Enhanced Dir-GCN)")
    st.dataframe(p3_df_enh, use_container_width=True)

    df_download_button(
        p3_df_enh,
        filename=f"problem3_layer_metrics_{dataset_name}_{config}_enhanced.csv",
        label="Download Problem 3 Per-layer Metrics (CSV)",
    )

st.markdown("---")

# ============================================================
# SECTION 4 – Predictions & Live Inference
# ============================================================

st.header("Section 4 – Predictions & Live Inference")

tabs = st.tabs(
    [
        "Existing Predictions (from training)",
        "Live Inference (new data)",
    ]
)

# ---------- Tab 1: existing prediction CSV ----------
with tabs[0]:
    st.write(
        "This tab shows the **saved prediction table** produced during diagnostics "
        "for the selected dataset / configuration / model."
    )

    pred_df = load_predictions_df(base_dir, conv_type)

    if pred_df is None:
        st.info(
            f"No prediction file found at "
            f"`results/{dataset_name}/{exp_id}/predictions/{conv_type}_best_run_predictions.csv`."
        )
    else:
        st.subheader(f"Prediction Table – {model_label}")
        st.dataframe(pred_df, use_container_width=True)

        df_download_button(
            pred_df,
            filename=f"predictions_{dataset_name}_{config}_{conv_type}.csv",
            label="Download Prediction Table (CSV)",
        )

        metrics = compute_eval_metrics_from_df(pred_df)
        if metrics is not None:
            st.subheader("Evaluation Metrics from Prediction Table")
            cols_eval = st.columns(4)
            with cols_eval[0]:
                st.metric("Accuracy", f"{metrics['accuracy']:.4f}")
            with cols_eval[1]:
                st.metric("Precision (macro)", f"{metrics['precision_macro']:.4f}")
            with cols_eval[2]:
                st.metric("Recall (macro)", f"{metrics['recall_macro']:.4f}")
            with cols_eval[3]:
                st.metric("F1-score (macro)", f"{metrics['f1_macro']:.4f}")
        else:
            st.info(
                "No usable ground-truth labels (`y_true`) found in the prediction CSV, "
                "so evaluation metrics cannot be computed here."
            )

# ---------- Tab 2: live inference ----------
with tabs[1]:
    st.write(
        "Use this tab to demonstrate **model predictions on new data**.\n\n"
        "- For **Online Payments**, you can upload a CSV in the original PaySim format "
        "or manually enter a single transaction. The app rebuilds the graph, runs the "
        "trained GNN, and shows the fraud probability for the origin account.\n"
        "- For **Synthetic Fraud** and **Elliptic**, live CSV/manual inference is not "
        "exposed here (these datasets are mainly for offline benchmarking). "
    )

    if dataset_name != "online-payments":
        st.info(
            "Live CSV / manual input is currently implemented **only for the "
            "Online Payments dataset**. Please switch the dataset in the sidebar "
            "to try it."
        )
    else:
        try:
            # Try to load the model once here, so any checkpoint error shows clearly
            _model, _device = load_trained_model(dataset_name, conv_type, exp_id)
        except Exception as e:
            st.error(f"Could not load trained model for live inference: {e}")
        else:
            sub_tabs = st.tabs(["Upload CSV", "Manual Input (single transaction)"])

            # ----- Upload CSV -----
            with sub_tabs[0]:
                st.subheader("Upload Online Payments CSV")
                uploaded_file = st.file_uploader(
                    "Upload a CSV in the Online Payments (PaySim) format",
                    type=["csv"],
                )
                if uploaded_file is not None:
                    try:
                        df_upload = pd.read_csv(uploaded_file)
                        st.write("Preview of uploaded data:")
                        st.dataframe(df_upload.head(), use_container_width=True)

                        preds_df = run_online_payments_inference_from_df(
                            df_upload,
                            dataset_name=dataset_name,
                            conv_type=conv_type,
                            exp_id=exp_id,
                        )

                        st.subheader("Predictions for Uploaded CSV")
                        st.dataframe(preds_df, use_container_width=True)

                        df_download_button(
                            preds_df,
                            filename=f"live_predictions_upload_{dataset_name}_{config}_{conv_type}.csv",
                            label="Download Live Predictions (CSV)",
                        )
                    except Exception as e:
                        st.error(f"Error during live inference on uploaded CSV: {e}")

            # ----- Manual Input -----
            with sub_tabs[1]:
                st.subheader("Manual Transaction Input")

                c1, c2 = st.columns(2)
                with c1:
                    amount = st.number_input(
                        "Amount",
                        min_value=0.0,
                        value=181.0,
                        step=1.0,
                    )
                    step_val = st.number_input(
                        "Step (time index)",
                        min_value=0,
                        value=1,
                        step=1,
                    )
                with c2:
                    tx_type = st.selectbox(
                        "Transaction Type",
                        ["TRANSFER", "CASH_OUT", "DEBIT", "PAYMENT", "CASH_IN"],
                    )
                    is_flagged = st.checkbox("isFlaggedFraud", value=False)

                if st.button("Predict"):
                    try:
                        preds_df_single = run_online_payments_inference_from_single(
                            amount=amount,
                            step=step_val,
                            tx_type=tx_type,
                            is_flagged=is_flagged,
                            dataset_name=dataset_name,
                            conv_type=conv_type,
                            exp_id=exp_id,
                        )

                        # --- Clear, explicit prediction summary ---
                        row = preds_df_single.iloc[0]
                        is_fraud = bool(row["y_pred"])
                        prob = float(row["prob_fraud"])

                        label_text = "FRAUD" if is_fraud else "NOT FRAUD"
                        label_color = "🔴" if is_fraud else "🟢"

                        st.markdown("#### Prediction Summary")
                        cols_summary = st.columns(2)
                        with cols_summary[0]:
                            st.metric("Predicted Class", f"{label_color} {label_text}")
                        with cols_summary[1]:
                            st.metric("Fraud Probability", f"{prob:.4f}")

                        st.caption(
                            "Prediction is made at the **account level** for the "
                            "origin account (`nameOrig`) in this mini-graph. "
                            "`y_pred = 1` means the account is predicted FRAUD."
                        )

                        st.subheader("Full Prediction Row (Key Fields Only)")

                        cols_to_show = [
                            "step",
                            "type",
                            "amount",
                            "isFlaggedFraud",
                            "y_pred",
                            "prob_fraud",
                        ]
                        cols_to_show = [
                            c for c in cols_to_show if c in preds_df_single.columns
                        ]

                        st.dataframe(
                            preds_df_single[cols_to_show], use_container_width=True
                        )

                    except Exception as e:
                        st.error(f"Error during manual-input inference: {e}")

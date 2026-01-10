import os
import json
import glob
import copy

import torch
import torch.nn.functional as F

from typing import Dict, Tuple

# ---------- Import your existing code ----------
from src.datasets.data_loading import get_dataset
from src.model import get_model
from src.utils.arguments import args as base_args  # reuse your defaults


# ==============================
# 1. Helpers (same logic as before)
# ==============================


def make_experiment_id(
    hidden_dim: int, num_layers: int, dropout: float, lr: float
) -> str:
    """
    Same logic as in compare_models.py:
    hdim32_L3_drop0p5_lr0p001
    """

    def f(x):
        s = str(x)
        return s.replace(".", "p")

    return f"hdim{hidden_dim}_L{num_layers}_drop{f(dropout)}_lr{f(lr)}"


CONFIGS: Dict[str, Dict] = {
    "C1": dict(hidden_dim=32, num_layers=3, dropout=0.5, lr=0.001),
    "C2": dict(hidden_dim=32, num_layers=3, dropout=0.6, lr=0.0005),
    "C3": dict(hidden_dim=64, num_layers=3, dropout=0.5, lr=0.001),
    "C4": dict(hidden_dim=64, num_layers=3, dropout=0.6, lr=0.0005),
}

DATASETS = {
    "fraud-syn": "Synthetic Fraud (fraud-syn)",
    "online-payments": "Online Payments",
    "elliptic": "Elliptic Bitcoin",
}

MODELS = {
    "dir-gcn": "Baseline Dir-GCN",
    "dir-gcn-gated": "Enhanced Dir-GCN (Gated)",
}


def build_results_root(dataset: str, cfg: Dict) -> Tuple[str, str]:
    """Return (results_root, exp_id) for a given dataset/config."""
    exp_id = make_experiment_id(
        hidden_dim=cfg["hidden_dim"],
        num_layers=cfg["num_layers"],
        dropout=cfg["dropout"],
        lr=cfg["lr"],
    )
    results_root = os.path.join("results", dataset, exp_id)
    return results_root, exp_id


def find_best_run(model_results_dir: str) -> int:
    """
    Pick the run index with highest best_val_f1 from run_*/metrics_run.json.
    Returns 1-based run index.
    """
    best_val = -1.0
    best_run = None

    for run_dir in sorted(glob.glob(os.path.join(model_results_dir, "run_*"))):
        metrics_path = os.path.join(run_dir, "metrics_run.json")
        if not os.path.exists(metrics_path):
            continue
        with open(metrics_path, "r") as f:
            m = json.load(f)
        val = m.get("best_val_f1", 0.0)
        if val > best_val:
            best_val = val
            run_name = os.path.basename(run_dir)
            try:
                idx = int(run_name.split("_")[1])
            except Exception:
                idx = 1
            best_run = idx

    if best_run is None:
        raise RuntimeError(f"No metrics_run.json found in {model_results_dir}")

    return best_run


def find_best_checkpoint(run_root: str) -> str:
    """
    Inside run_root, there is a random UUID folder containing .ckpt.
    Return the path to the first/only .ckpt.
    """
    uuid_dirs = [d for d in glob.glob(os.path.join(run_root, "*")) if os.path.isdir(d)]
    if not uuid_dirs:
        raise RuntimeError(f"No checkpoint folder found inside {run_root}")

    ckpt_files = glob.glob(os.path.join(uuid_dirs[0], "*.ckpt"))
    if not ckpt_files:
        raise RuntimeError(f"No .ckpt file found in {uuid_dirs[0]}")

    return ckpt_files[0]


def load_gnn_from_checkpoint(ckpt_path: str, model) -> None:
    """
    Load only the GNN weights from a Lightning checkpoint into the plain GNN model.
    The Lightning checkpoint stores keys like 'model.convs.0.lin_src_to_dst.weight'.
    We strip the 'model.' prefix and load into the bare GNN.
    """
    print(f"[INFO] Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state_dict = ckpt["state_dict"]

    gnn_state = {}
    for k, v in state_dict.items():
        if k.startswith("model."):
            new_k = k[len("model.") :]
            gnn_state[new_k] = v

    missing, unexpected = model.load_state_dict(gnn_state, strict=False)
    if missing:
        print("[Warning] Missing keys when loading GNN:", missing)
    if unexpected:
        print("[Warning] Unexpected keys when loading GNN:", unexpected)


def export_predictions_for_combo(dataset_name: str, cfg_label: str, conv_type: str):
    """
    For one dataset + config + model:
      - load dataset & model
      - find best checkpoint
      - run inference on all nodes
      - save predictions JSON to results/.../predictions/<conv_type>_node_predictions.json
    """
    cfg = CONFIGS[cfg_label]

    # Build results path
    results_root, exp_id = build_results_root(dataset_name, cfg)
    model_results_dir = os.path.join(results_root, conv_type)

    if not os.path.isdir(model_results_dir):
        print(f"[SKIP] No directory: {model_results_dir}")
        return

    best_run_idx = find_best_run(model_results_dir)
    run_root = os.path.join(model_results_dir, f"run_{best_run_idx}")
    ckpt_path = find_best_checkpoint(run_root)

    # Load dataset (same preprocessing as training)
    local_args = copy.deepcopy(base_args)
    local_args.dataset = dataset_name
    dataset, _ = get_dataset(
        name=dataset_name,
        root_dir=local_args.dataset_directory,
        undirected=local_args.undirected,
        self_loops=local_args.self_loops,
        transpose=local_args.transpose,
    )
    data = dataset._data

    # Build GNN model
    local_args.num_features = data.num_features
    local_args.num_classes = dataset.num_classes
    local_args.hidden_dim = cfg["hidden_dim"]
    local_args.num_layers = cfg["num_layers"]
    local_args.dropout = cfg["dropout"]
    local_args.lr = cfg["lr"]
    local_args.conv_type = conv_type

    gnn_model = get_model(local_args)
    gnn_model.eval()

    # Load weights
    load_gnn_from_checkpoint(ckpt_path, gnn_model)

    # Inference
    print(
        f"[INFO] Running inference for dataset={dataset_name}, cfg={cfg_label}, model={conv_type}"
    )
    with torch.no_grad():
        out = gnn_model(data.x, data.edge_index)  # shape [N, C]
        probs = F.softmax(out, dim=-1)  # probabilities
        preds = probs.argmax(dim=-1)  # predicted class
        y_true = getattr(data, "y", None)

    # Build prediction dict
    pred_dict = {}
    num_nodes = data.num_nodes
    for i in range(num_nodes):
        record = {
            "y_pred": int(preds[i].item()),
            "p0": float(probs[i, 0].item()),
            "p1": float(probs[i, 1].item()),
        }
        if y_true is not None:
            record["y_true"] = int(y_true[i].item())
        else:
            record["y_true"] = -1  # unlabeled
        pred_dict[str(i)] = record

    # Save JSON
    pred_dir = os.path.join(results_root, "predictions")
    os.makedirs(pred_dir, exist_ok=True)
    out_path = os.path.join(pred_dir, f"{conv_type}_node_predictions.json")
    with open(out_path, "w") as f:
        json.dump(pred_dict, f)

    print(f"[DONE] Saved predictions to: {out_path}")


def main():
    # Choose which combinations to export
    datasets_to_do = ["fraud-syn", "online-payments", "elliptic"]
    configs_to_do = ["C1", "C2", "C3", "C4"]
    models_to_do = ["dir-gcn", "dir-gcn-gated"]

    for ds in datasets_to_do:
        for cfg in configs_to_do:
            for m in models_to_do:
                try:
                    export_predictions_for_combo(ds, cfg, m)
                except Exception as e:
                    print(f"[ERROR] Failed for {ds}, {cfg}, {m}: {e}")


if __name__ == "__main__":
    main()

import os
import uuid
import time
import json
from copy import deepcopy

import numpy as np
import psutil

import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger

from src.datasets.data_loading import get_dataset, get_dataset_split
from src.datasets.dataset import FullBatchGraphDataset
from src.model import (
    get_model,
    LightingFullBatchModelWrapper,
    LightingMiniBatchDirGatedWrapper,
    export_node_predictions,
    measure_inference_time,
    collect_problem3_metrics,
    problem3_metrics_to_dataframe,
)
from src.utils.arguments import args  # parsed from CLI


# ============================================================================
# Helpers
# ============================================================================


def make_experiment_id(a) -> str:
    """
    Build a short ID from the current hyperparameters so each setting
    gets its own folder.

    Example: hidden_dim=32, num_layers=3, dropout=0.5, lr=0.001
    -> 'hdim32_L3_drop0p5_lr0p001'
    """

    def f(x):
        s = str(x)
        return s.replace(".", "p")

    return f"hdim{a.hidden_dim}_L{a.num_layers}_drop{f(a.dropout)}_lr{f(a.lr)}"


def save_hyperparams(local_args, results_root: str):
    """Save the hyperparameters for a single config under results_root."""
    hyperparams = {
        "dataset": local_args.dataset,
        "hidden_dim": local_args.hidden_dim,
        "num_layers": local_args.num_layers,
        "dropout": local_args.dropout,
        "lr": local_args.lr,
        "weight_decay": local_args.weight_decay,
        "num_epochs": local_args.num_epochs,
        "num_runs": local_args.num_runs,
        "alpha": local_args.alpha,
        "learn_alpha": local_args.learn_alpha,
        "undirected": local_args.undirected,
        "self_loops": local_args.self_loops,
        "transpose": local_args.transpose,
        "jk": local_args.jk,
        "normalize": local_args.normalize,
        "enable_lcs_masking": getattr(local_args, "enable_lcs_masking", False),
        "lcs_threshold": getattr(local_args, "lcs_threshold", 0.0),
        "mini_batch_training": getattr(local_args, "mini_batch_training", False),
        "mini_batch_size": getattr(local_args, "mini_batch_size", 1024),
    }
    os.makedirs(results_root, exist_ok=True)
    with open(os.path.join(results_root, "hyperparams.json"), "w") as f:
        json.dump(hyperparams, f, indent=2)
    print(f"Hyperparameters saved to: {os.path.join(results_root, 'hyperparams.json')}")


# ============================================================================
# Diagnostics: confusion / ROC / PR + structural / LCS / MP stats / Problem 3
# ============================================================================


def create_diagnostics_plots(
    conv_type: str,
    label: str,
    best_run_idx: int,
    best_ckpt_path: str,
    data,
    local_args,
    results_root: str,
):
    """
    For the *best* run of a given model, load the best checkpoint and
    generate:

      - Confusion matrix
      - ROC curve           (if binary)
      - Precision–Recall    (if binary)

    Also, for each convolution layer (especially the enhanced DIR-GCN),
    we:
      - trigger one forward pass on the full graph
      - read per-layer diagnostics:
          * recurring_transaction_stats
          * lcs_masking_stats
          * mp_operation_stats
        (and legacy names structural_redundancy / lcs_redundancy_stats)
      - save them into JSON so they can be used in Chapter 4 tables.

    NEW (Problem 3 / Solution 3 additions):
      - collect and export Problem 3 metrics across layers as CSV + JSON summary
      - export node-level prediction table as CSV
      - measure inference runtime (average over several runs) and save to JSON

    Saved under:
        results_root/plots/<conv_type>/
        results_root/struct_lcs_stats/<conv_type>_best_run_struct_lcs_stats.json
        results_root/problem3_metrics/<conv_type>_best_run_problem3_metrics.csv
        results_root/problem3_metrics/<conv_type>_best_run_problem3_summary.json
        results_root/predictions/<conv_type>_best_run_predictions.csv
        results_root/runtime/<conv_type>_best_run_inference_runtime.json
    """
    import matplotlib.pyplot as plt
    from sklearn.metrics import (
        confusion_matrix,
        roc_curve,
        auc,
        precision_recall_curve,
    )

    # Rebuild the same data split for this run index
    train_mask, val_mask, test_mask = get_dataset_split(
        local_args.dataset, data, local_args.dataset_directory, best_run_idx
    )

    plots_dir = os.path.join(results_root, "plots", conv_type)
    os.makedirs(plots_dir, exist_ok=True)

    # 1) Rebuild the underlying GNN with the same hyper-params
    model_args = deepcopy(local_args)
    model_args.conv_type = conv_type
    model_args.num_features = data.num_features

    # If num_classes is not set on args, infer from labels
    num_classes = int(getattr(model_args, "num_classes", int(data.y.max().item() + 1)))
    model_args.num_classes = num_classes

    model = get_model(model_args)

    # 2) Load checkpoint and extract only the "model." weights
    ckpt = torch.load(best_ckpt_path, map_location="cpu")
    state_dict = ckpt["state_dict"]

    cleaned_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("model."):
            new_key = k.replace("model.", "", 1)
            cleaned_state_dict[new_key] = v

    model.load_state_dict(cleaned_state_dict, strict=False)
    model.eval()
    device = torch.device("cpu")
    model.to(device)

    x = data.x.to(device)
    edge_index = data.edge_index.to(device)
    y = data.y.long().view(-1).to(device)

    # ---------------- Single full-graph forward for diagnostics --------------
    with torch.no_grad():
        logits = model(x, edge_index, batch_nodes=None)  # [N, C] (log-softmax)
        probs = torch.exp(logits)
        y_true = y[test_mask].cpu().numpy()
        y_pred = probs.argmax(dim=1)[test_mask].cpu().numpy()

    num_classes = probs.shape[1]

    # ---------------- Confusion matrix ----------------
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")

    ax.set_title(f"{label} – Confusion Matrix (best run {best_run_idx + 1})")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    fig.colorbar(im, ax=ax)

    tick_marks = np.arange(num_classes)
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels([str(i) for i in range(num_classes)])
    ax.set_yticklabels([str(i) for i in range(num_classes)])

    for i in range(num_classes):
        for j in range(num_classes):
            ax.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center",
                color="black",
                fontsize=8,
            )

    fig.tight_layout()
    cm_path = os.path.join(plots_dir, f"{conv_type}_best_run_confusion_matrix.png")
    plt.savefig(cm_path, dpi=200)
    plt.close(fig)
    print(f"Saved confusion matrix to: {cm_path}")

    # ------------- ROC & PR curves (binary only) -------------
    if num_classes == 2:
        # Assume class 1 is the "fraud"/positive class
        y_score = probs[test_mask, 1].cpu().numpy()

        # ROC
        fpr, tpr, _ = roc_curve(y_true, y_score, pos_label=1)
        roc_auc = auc(fpr, tpr)

        fig, ax = plt.subplots(figsize=(5, 4))
        ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
        ax.plot([0, 1], [0, 1], "k--", label="Random")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(f"{label} – ROC Curve (best run {best_run_idx + 1})")
        ax.legend(loc="lower right")
        fig.tight_layout()
        roc_path = os.path.join(plots_dir, f"{conv_type}_best_run_roc_curve.png")
        plt.savefig(roc_path, dpi=200)
        plt.close(fig)
        print(f"Saved ROC curve to: {roc_path}")

        # Precision–Recall
        prec, rec, _ = precision_recall_curve(y_true, y_score, pos_label=1)
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.plot(rec, prec)
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title(f"{label} – Precision–Recall (best run {best_run_idx + 1})")
        fig.tight_layout()
        pr_path = os.path.join(plots_dir, f"{conv_type}_best_run_pr_curve.png")
        plt.savefig(pr_path, dpi=200)
        plt.close(fig)
        print(f"Saved Precision–Recall curve to: {pr_path}")
    else:
        print(f"Skipping ROC/PR curves for {label} (num_classes={num_classes} > 2).")

    # ------------------------------------------------------------------
    # Structural / LCS / MP-operation stats for each layer (per-forward)
    # ------------------------------------------------------------------
    diagnostics_by_layer = {}
    if hasattr(model, "convs"):
        for layer_idx, conv in enumerate(model.convs):
            layer_key = f"layer_{layer_idx}"
            layer_stats = {}

            # New names from enhanced DIR-GCN implementation
            if hasattr(conv, "recurring_transaction_stats"):
                stats = getattr(conv, "recurring_transaction_stats")
                if stats:
                    layer_stats["recurring_transaction_stats"] = stats

            if hasattr(conv, "lcs_masking_stats"):
                stats = getattr(conv, "lcs_masking_stats")
                if stats:
                    layer_stats["lcs_masking_stats"] = stats

            if hasattr(conv, "mp_operation_stats"):
                stats = getattr(conv, "mp_operation_stats")
                if stats:
                    layer_stats["mp_operation_stats"] = stats

            # Legacy names (for backward compatibility)
            if hasattr(conv, "structural_redundancy"):
                stats = getattr(conv, "structural_redundancy")
                if stats:
                    layer_stats["structural_redundancy"] = stats

            if hasattr(conv, "lcs_redundancy_stats"):
                stats = getattr(conv, "lcs_redundancy_stats")
                if stats:
                    layer_stats["lcs_redundancy_stats"] = stats

            if layer_stats:
                diagnostics_by_layer[layer_key] = layer_stats

    if diagnostics_by_layer:
        stats_dir = os.path.join(results_root, "struct_lcs_stats")
        os.makedirs(stats_dir, exist_ok=True)
        stats_path = os.path.join(
            stats_dir, f"{conv_type}_best_run_struct_lcs_stats.json"
        )
        with open(stats_path, "w") as f:
            json.dump(diagnostics_by_layer, f, indent=2)
        print(f"Saved structural/LCS/MP diagnostics to: {stats_path}")
    else:
        print(f"No structural/LCS/MP diagnostics found for {label}.")

    # ------------------------------------------------------------------
    # NEW: Problem 3 metrics (aggregations + redundancy) across layers
    # ------------------------------------------------------------------
    p3_metrics = collect_problem3_metrics(model)
    p3_df = problem3_metrics_to_dataframe(p3_metrics)

    p3_dir = os.path.join(results_root, "problem3_metrics")
    os.makedirs(p3_dir, exist_ok=True)

    if not p3_df.empty:
        p3_csv_path = os.path.join(p3_dir, f"{conv_type}_best_run_problem3_metrics.csv")
        p3_df.to_csv(p3_csv_path, index=False)

        p3_summary_path = os.path.join(
            p3_dir, f"{conv_type}_best_run_problem3_summary.json"
        )
        with open(p3_summary_path, "w") as f:
            json.dump(p3_metrics.get("summary", {}), f, indent=2)

        print(f"Saved Problem 3 metrics to: {p3_csv_path}")
    else:
        print(f"No Problem 3 metrics collected for {label}.")

    # ------------------------------------------------------------------
    # NEW: Full prediction table (all nodes) -> CSV
    # ------------------------------------------------------------------
    preds_dir = os.path.join(results_root, "predictions")
    os.makedirs(preds_dir, exist_ok=True)
    preds_path = os.path.join(preds_dir, f"{conv_type}_best_run_predictions.csv")

    # Optional: provide class_names if desired; here we keep generic names.
    export_node_predictions(model, data, preds_path, class_names=None)
    print(f"Saved node-level prediction table to: {preds_path}")

    # ------------------------------------------------------------------
    # NEW: Inference runtime (Solution 3: Inference speed ×)
    # ------------------------------------------------------------------
    runtime_dir = os.path.join(results_root, "runtime")
    os.makedirs(runtime_dir, exist_ok=True)

    # Use a deepcopy so we don't further perturb diagnostic counters on 'model'.
    avg_inf_time = measure_inference_time(deepcopy(model), data, runs=10)
    runtime_path = os.path.join(
        runtime_dir, f"{conv_type}_best_run_inference_runtime.json"
    )
    with open(runtime_path, "w") as f:
        json.dump(
            {
                "conv_type": conv_type,
                "label": label,
                "best_run_index": int(best_run_idx),
                "avg_inference_time_seconds": float(avg_inf_time),
            },
            f,
            indent=2,
        )
    print(f"Saved inference runtime stats to: {runtime_path}")


# ============================================================================
# Training of a single model type (baseline or enhanced) – full-batch vs mini
# ============================================================================


def run_single_model(base_args, conv_type: str, label: str, results_root: str):
    """
    Train & evaluate a single model type (baseline or enhanced)
    using the SAME hyperparameters from base_args, except conv_type.

    - Baseline (dir-gcn): full-batch training (LightingFullBatchModelWrapper)
    - Enhanced (dir-gcn-gated):
        * full-batch if mini_batch_training is False
        * mini-batch if mini_batch_training is True

    Validation / Test:
        - full-graph evaluation; train/val/test masks control which nodes
          contribute to each metric.

    All outputs are saved neatly under:
        results_root/<conv_type>/run_<k>/  (checkpoints, per-run metrics, CSV logs)
        results_root/<conv_type>/summary.json

    NOTE:
      Inference runtime + Problem 3 metrics + prediction CSV are produced
      in create_diagnostics_plots(...) for the best run.
    """
    local_args = deepcopy(base_args)
    local_args.conv_type = conv_type

    # ---- NEW: auto-configure enhanced DIR-GCN for efficiency ----
    if conv_type == "dir-gcn-gated":
        # Turn on mini-batch training unless the user explicitly disabled it
        if not hasattr(local_args, "mini_batch_training"):
            local_args.mini_batch_training = True
        elif local_args.mini_batch_training is False:
            # respect explicit False from CLI
            pass
        else:
            local_args.mini_batch_training = True

        # Default mini-batch size if not provided
        if (
            not hasattr(local_args, "mini_batch_size")
            or local_args.mini_batch_size <= 0
        ):
            local_args.mini_batch_size = 1024

        # Make sure LCS masking is active with a reasonable threshold
        if (
            not hasattr(local_args, "enable_lcs_masking")
            or not local_args.enable_lcs_masking
        ):
            local_args.enable_lcs_masking = True
        if not hasattr(local_args, "lcs_threshold") or local_args.lcs_threshold <= 0.0:
            local_args.lcs_threshold = 0.5
    # --------------------------------------------------------------

    model_results_dir = os.path.join(results_root, conv_type)
    os.makedirs(model_results_dir, exist_ok=True)

    print(f"\n================ {label} (conv_type={conv_type}) ================")

    torch.manual_seed(0)

    # ---- load dataset once ----
    dataset, evaluator = get_dataset(
        name=local_args.dataset,
        root_dir=local_args.dataset_directory,
        undirected=local_args.undirected,
        self_loops=local_args.self_loops,
        transpose=local_args.transpose,
    )
    data = dataset._data

    # full-batch loader (baseline + enhanced share the same loader;
    # enhanced mini-batch wrapper samples seed nodes internally)
    data_loader = DataLoader(
        FullBatchGraphDataset(data),
        batch_size=1,
        collate_fn=lambda batch: batch[0],
    )

    # metrics storage
    val_scores = []  # best val_f1
    test_accs = []
    test_f1s = []
    test_precs = []
    test_recs = []

    train_times = []
    test_times = []
    total_times = []
    mem_usages_mb = []
    best_ckpt_paths = []  # path to best checkpoint per run

    for num_run in range(local_args.num_runs):
        print(
            f"\n---------- {label} | Run {num_run + 1} / {local_args.num_runs} ----------"
        )

        # same splits for this dataset/run index
        train_mask, val_mask, test_mask = get_dataset_split(
            local_args.dataset, data, local_args.dataset_directory, num_run
        )

        # ---- build model ----
        local_args.num_features = data.num_features
        local_args.num_classes = dataset.num_classes

        model = get_model(local_args)

        # Decide whether to use full-batch or mini-batch wrapper
        use_mini_batch = conv_type == "dir-gcn-gated" and getattr(
            local_args, "mini_batch_training", False
        )

        if use_mini_batch:
            print(
                f"[INFO] Using mini-batch training for enhanced model with "
                f"mini_batch_size={getattr(local_args, 'mini_batch_size', 1024)}"
            )
            lit_model = LightingMiniBatchDirGatedWrapper(
                model=model,
                lr=local_args.lr,
                weight_decay=local_args.weight_decay,
                evaluator=evaluator,
                train_mask=train_mask,
                val_mask=val_mask,
                test_mask=test_mask,
                mini_batch_size=getattr(local_args, "mini_batch_size", 1024),
            )
        else:
            lit_model = LightingFullBatchModelWrapper(
                model=model,
                lr=local_args.lr,
                weight_decay=local_args.weight_decay,
                evaluator=evaluator,
                train_mask=train_mask,
                val_mask=val_mask,
                test_mask=test_mask,
            )

        # folder for this run’s checkpoints + logs
        run_root = os.path.join(model_results_dir, f"run_{num_run + 1}")
        os.makedirs(run_root, exist_ok=True)

        # callbacks: early stopping + checkpoint on val_f1
        early_stopping_callback = EarlyStopping(
            monitor="val_f1",
            mode="max",
            patience=local_args.patience,
            check_on_train_epoch_end=True,
        )

        ckpt_dir = os.path.join(run_root, str(uuid.uuid4()))
        os.makedirs(ckpt_dir, exist_ok=True)

        model_checkpoint_callback = ModelCheckpoint(
            monitor="val_f1",
            mode="max",
            dirpath=ckpt_dir,
        )

        # CSV logger = "snapshot during training phase"
        csv_logger = CSVLogger(
            save_dir=run_root,
            name="logs",
        )

        trainer = pl.Trainer(
            log_every_n_steps=1,
            max_epochs=local_args.num_epochs,
            callbacks=[early_stopping_callback, model_checkpoint_callback],
            profiler="simple" if local_args.profiler else None,
            accelerator="cpu",
            devices=1,
            enable_model_summary=False,
            logger=csv_logger,
        )

        # ---------- train ----------
        t0_train = time.perf_counter()
        trainer.fit(
            model=lit_model,
            train_dataloaders=data_loader,
        )
        t1_train = time.perf_counter()
        train_time = t1_train - t0_train

        # best validation F1
        best_val_f1 = model_checkpoint_callback.best_model_score.item()
        best_ckpt_path = model_checkpoint_callback.best_model_path
        best_ckpt_paths.append(best_ckpt_path)

        # ---------- test ----------
        t0_test = time.perf_counter()
        test_results = trainer.test(ckpt_path="best", dataloaders=data_loader)[0]
        t1_test = time.perf_counter()
        test_time = t1_test - t0_test
        total_time = train_time + test_time

        # memory usage (snapshot at end of run)
        process = psutil.Process(os.getpid())
        mem_bytes = process.memory_info().rss
        mem_mb = mem_bytes / (1024**2)

        # metrics
        test_acc = float(test_results.get("test_acc", 0.0))
        test_f1 = float(test_results.get("test_f1", 0.0))
        test_prec = float(test_results.get("test_prec", 0.0))
        test_rec = float(test_results.get("test_rec", 0.0))

        val_scores.append(best_val_f1)
        test_accs.append(test_acc)
        test_f1s.append(test_f1)
        test_precs.append(test_prec)
        test_recs.append(test_rec)
        train_times.append(train_time)
        test_times.append(test_time)
        total_times.append(total_time)
        mem_usages_mb.append(mem_mb)

        print(f"Best Val F1-score : {best_val_f1:.4f}")
        print(f"Test Accuracy     : {test_acc:.4f}")
        print(f"Test F1-score     : {test_f1:.4f}")
        print(f"Test Precision    : {test_prec:.4f}")
        print(f"Test Recall       : {test_rec:.4f}")
        print(f"Train time (s)    : {train_time:.2f}")
        print(f"Test time  (s)    : {test_time:.2f}")
        print(f"Total time (s)    : {total_time:.2f}")
        print(f"Memory usage (MB) : {mem_mb:.2f}")

        run_metrics = {
            "run_index": num_run + 1,
            "best_val_f1": best_val_f1,
            "test_acc": test_acc,
            "test_f1": test_f1,
            "test_prec": test_prec,
            "test_rec": test_rec,
            "train_time": train_time,
            "test_time": test_time,
            "total_time": total_time,
            "mem_mb": mem_mb,
        }
        with open(os.path.join(run_root, "metrics_run.json"), "w") as f:
            json.dump(run_metrics, f, indent=2)

    # ------ aggregate ------
    def summarize(values):
        v = np.array(values, dtype=float)
        return v.mean(), v.std()

    results = {
        "label": label,
        "val_f1": summarize(val_scores),
        "test_acc": summarize(test_accs),
        "test_f1": summarize(test_f1s),
        "test_prec": summarize(test_precs),
        "test_rec": summarize(test_recs),
        "train_time": summarize(train_times),
        "test_time": summarize(test_times),
        "total_time": summarize(total_times),
        "mem_mb": summarize(mem_usages_mb),
        "num_runs": int(local_args.num_runs),
    }

    print(f"\n===== {label} FINAL SUMMARY over {local_args.num_runs} runs =====")
    for key in [
        "val_f1",
        "test_acc",
        "test_f1",
        "test_prec",
        "test_rec",
        "train_time",
        "test_time",
        "total_time",
        "mem_mb",
    ]:
        mean, std = results[key]
        print(f"{key:15s}: {mean:.4f} ± {std:.4f}")

    # summary.json
    summary_path = os.path.join(model_results_dir, "summary.json")
    serializable = {
        k: (float(v[0]), float(v[1])) if isinstance(v, tuple) else v
        for k, v in results.items()
        if k != "label"
    }
    serializable["label"] = label
    with open(summary_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"Saved {label} summary to: {summary_path}")

    # ----- diagnostics & redundancy stats on best run -----
    best_run_idx = int(np.argmax(np.array(val_scores)))
    best_ckpt_path = best_ckpt_paths[best_run_idx]

    print(
        f"\n[Diagnostics] {label}: using best run index {best_run_idx + 1} "
        f"with best_val_f1={val_scores[best_run_idx]:.4f}"
    )

    create_diagnostics_plots(
        conv_type=conv_type,
        label=label,
        best_run_idx=best_run_idx,
        best_ckpt_path=best_ckpt_path,
        data=data,
        local_args=local_args,
        results_root=results_root,
    )

    return results


# ============================================================================
# Overall plots vs Config (C1–C4)
# ============================================================================


def plot_overall_by_config(
    dataset_name: str,
    config_ids,
    baseline_by_cfg,
    enhanced_by_cfg,
    out_root: str,
):
    import matplotlib.pyplot as plt

    os.makedirs(out_root, exist_ok=True)

    metric_specs = [
        (
            "mem_mb",
            "Memory Usage vs Config",
            "Memory Usage (MB)",
            "memory_by_config.png",
        ),
        (
            "total_time",
            "Total Time vs Config",
            "Total Time (s)",
            "total_time_by_config.png",
        ),
        ("val_f1", "Best Validation F1 vs Config", "Score", "val_f1_by_config.png"),
        ("test_acc", "Test Accuracy vs Config", "Score", "test_acc_by_config.png"),
        ("test_f1", "Test F1-score vs Config", "Score", "test_f1_by_config.png"),
        ("test_prec", "Test Precision vs Config", "Score", "test_prec_by_config.png"),
        ("test_rec", "Test Recall vs Config", "Score", "test_rec_by_config.png"),
    ]

    x = np.arange(len(config_ids))
    width = 0.35

    for key, title, ylabel, fname in metric_specs:
        baseline_vals = [baseline_by_cfg[cid][key][0] for cid in config_ids]
        enhanced_vals = [enhanced_by_cfg[cid][key][0] for cid in config_ids]

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(x - width / 2, baseline_vals, width, label="Baseline Dir-GCN")
        ax.bar(x + width / 2, enhanced_vals, width, label="Enhanced Dir-GCN (Gated)")

        ax.set_title(f"{title} ({dataset_name})")
        ax.set_xlabel("Config")
        ax.set_ylabel(ylabel)
        ax.set_xticks(x)
        ax.set_xticklabels(config_ids)
        if key not in ("mem_mb", "total_time"):
            ax.set_ylim(0.0, 1.0)
        ax.legend()

        fig.tight_layout()
        out_path = os.path.join(out_root, fname)
        plt.savefig(out_path, dpi=200)
        plt.close(fig)
        print(f"Saved overall {key} plot to: {out_path}")


# ============================================================================
# Main – run all configs (C1–C4) in one go
# ============================================================================


if __name__ == "__main__":
    # All configs we want to sweep over (C1–C4)
    CONFIGS = [
        ("C1", dict(hidden_dim=32, num_layers=3, dropout=0.5, lr=0.001)),
        ("C2", dict(hidden_dim=32, num_layers=3, dropout=0.6, lr=0.0005)),
        ("C3", dict(hidden_dim=64, num_layers=3, dropout=0.5, lr=0.001)),
        ("C4", dict(hidden_dim=64, num_layers=3, dropout=0.6, lr=0.0005)),
    ]

    dataset_root = os.path.join("results", args.dataset)
    os.makedirs(dataset_root, exist_ok=True)

    baseline_by_cfg = {}
    enhanced_by_cfg = {}

    for cfg_id, cfg_params in CONFIGS:
        print("\n" + "=" * 80)
        print(f"Running config {cfg_id}: {cfg_params}")
        print("=" * 80)

        # Clone CLI args and override with config-specific hyperparams
        local_args = deepcopy(args)
        local_args.hidden_dim = cfg_params["hidden_dim"]
        local_args.num_layers = cfg_params["num_layers"]
        local_args.dropout = cfg_params["dropout"]
        local_args.lr = cfg_params["lr"]

        exp_id = make_experiment_id(local_args)
        results_root_cfg = os.path.join(dataset_root, exp_id)
        print(f"Experiment ID ({cfg_id}): {exp_id}")

        # Save the “global” config once (shared things like dataset, num_runs, etc.)
        save_hyperparams(local_args, results_root_cfg)

        # Baseline (full-batch)
        baseline_results = run_single_model(
            base_args=local_args,
            conv_type="dir-gcn",
            label=f"Baseline Dir-GCN ({cfg_id})",
            results_root=results_root_cfg,
        )

        # Enhanced (will auto-enable mini-batch + LCS)
        enhanced_results = run_single_model(
            base_args=local_args,
            conv_type="dir-gcn-gated",
            label=f"Enhanced Dir-GCN (Gated, {cfg_id})",
            results_root=results_root_cfg,
        )

        baseline_by_cfg[cfg_id] = baseline_results
        enhanced_by_cfg[cfg_id] = enhanced_results

    # After ALL configs are done, create overall plots vs config
    overall_root = os.path.join(dataset_root, "overall")
    config_ids = [cid for cid, _ in CONFIGS]
    plot_overall_by_config(
        dataset_name=args.dataset,
        config_ids=config_ids,
        baseline_by_cfg=baseline_by_cfg,
        enhanced_by_cfg=enhanced_by_cfg,
        out_root=overall_root,
    )

    print("\nAll configs finished. Overall plots saved under:", overall_root)

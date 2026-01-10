import os
import uuid
import time

import numpy as np
import psutil

import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint

from src.utils.utils import use_best_hyperparams
from src.datasets.data_loading import get_dataset, get_dataset_split
from src.datasets.dataset import FullBatchGraphDataset
from src.model import get_model, LightingFullBatchModelWrapper
from src.utils.arguments import args


def run(args):
    torch.manual_seed(0)

    # Get dataset and dataloader
    dataset, evaluator = get_dataset(
        name=args.dataset,
        root_dir=args.dataset_directory,
        undirected=args.undirected,
        self_loops=args.self_loops,
        transpose=args.transpose,
    )
    data = dataset._data
    data_loader = DataLoader(
        FullBatchGraphDataset(data), batch_size=1, collate_fn=lambda batch: batch[0]
    )

    # To store metrics across runs
    val_scores = []  # this will store best validation F1 (because we monitor val_f1)
    test_accs = []
    test_f1s = []
    test_precs = []
    test_recs = []

    train_times = []
    test_times = []
    total_times = []
    mem_usages_mb = []

    for num_run in range(args.num_runs):
        print(f"\n========== Run {num_run + 1} / {args.num_runs} ==========")

        # Get train/val/test splits for the current run
        train_mask, val_mask, test_mask = get_dataset_split(
            args.dataset, data, args.dataset_directory, num_run
        )

        # Get model
        args.num_features, args.num_classes = data.num_features, dataset.num_classes
        model = get_model(args)
        lit_model = LightingFullBatchModelWrapper(
            model=model,
            lr=args.lr,
            weight_decay=args.weight_decay,
            evaluator=evaluator,
            train_mask=train_mask,
            val_mask=val_mask,
            test_mask=test_mask,
        )

        # Setup PyTorch Lightning Callbacks
        # Use F1-score as primary metric for early stopping and checkpointing
        early_stopping_callback = EarlyStopping(
            monitor="val_f1", mode="max", patience=args.patience
        )

        if not os.path.exists(f"{args.checkpoint_directory}/"):
            os.mkdir(f"{args.checkpoint_directory}/")

        ckpt_dir = f"{args.checkpoint_directory}/{str(uuid.uuid4())}/"
        model_checkpoint_callback = ModelCheckpoint(
            monitor="val_f1",
            mode="max",
            dirpath=ckpt_dir,
        )

        # Setup PyTorch Lightning Trainer
        trainer = pl.Trainer(
            log_every_n_steps=1,
            max_epochs=args.num_epochs,
            callbacks=[
                early_stopping_callback,
                model_checkpoint_callback,
            ],
            profiler="simple" if args.profiler else None,
            accelerator="cpu",
            devices=1,
            enable_model_summary=False,  # <- prevents model architecture table every time
        )

        # Measure training time
        t0_train = time.perf_counter()
        trainer.fit(model=lit_model, train_dataloaders=data_loader)
        t1_train = time.perf_counter()
        train_time = t1_train - t0_train

        # Compute validation and test metrics
        # best_model_score now corresponds to best val_f1 (since we monitor "val_f1")
        best_val_score = model_checkpoint_callback.best_model_score.item()

        # Measure test time
        t0_test = time.perf_counter()
        test_results = trainer.test(ckpt_path="best", dataloaders=data_loader)[
            0
        ]  # dict with test_acc, test_f1, etc.
        t1_test = time.perf_counter()
        test_time = t1_test - t0_test

        total_time = train_time + test_time

        # Memory usage (approximate peak after training+test)
        process = psutil.Process(os.getpid())
        mem_bytes = process.memory_info().rss
        mem_mb = mem_bytes / (1024**2)

        # Collect metrics
        val_scores.append(best_val_score)

        test_acc = float(test_results.get("test_acc", 0.0))
        test_f1 = float(test_results.get("test_f1", 0.0))
        test_prec = float(test_results.get("test_prec", 0.0))
        test_rec = float(test_results.get("test_rec", 0.0))

        test_accs.append(test_acc)
        test_f1s.append(test_f1)
        test_precs.append(test_prec)
        test_recs.append(test_rec)

        train_times.append(train_time)
        test_times.append(test_time)
        total_times.append(total_time)
        mem_usages_mb.append(mem_mb)

        # Per-run printout
        print(f"\n--- Run {num_run + 1} metrics ---")
        print(f"Best Val F1-score : {best_val_score:.4f}")
        print(f"Test Accuracy     : {test_acc:.4f}")
        print(f"Test F1-score     : {test_f1:.4f}")
        print(f"Test Precision    : {test_prec:.4f}")
        print(f"Test Recall       : {test_rec:.4f}")
        print(f"Train time (s)    : {train_time:.2f}")
        print(f"Test time  (s)    : {test_time:.2f}")
        print(f"Total time (s)    : {total_time:.2f}")
        print(f"Memory usage (MB) : {mem_mb:.2f}")

    # ===== Final aggregated summary over all runs =====
    def summarize(name, values):
        values = np.array(values, dtype=float)
        return f"{name}: {values.mean():.4f} ± {values.std():.4f}"

    print("\n================= FINAL SUMMARY =================")
    print(f"Number of runs: {args.num_runs}")

    # Primary metric
    print(summarize("Val F1-score (best)", val_scores))
    print(summarize("Test Accuracy       ", test_accs))
    print(summarize("Test F1-score       ", test_f1s))
    print(summarize("Test Precision      ", test_precs))
    print(summarize("Test Recall         ", test_recs))

    # Runtime + memory
    print(summarize("Train time (s)      ", train_times))
    print(summarize("Test time (s)       ", test_times))
    print(summarize("Total time (s)      ", total_times))
    print(summarize("Memory usage (MB)   ", mem_usages_mb))


if __name__ == "__main__":
    run(args)

import os
from typing import Tuple

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data, InMemoryDataset


def compute_edge_homophily(edge_index: torch.Tensor, y: torch.Tensor) -> float:
    """
    Fraction of edges whose endpoints share the same label (ignores label -1 if present).
    """
    src, dst = edge_index
    mask = (y[src] != -1) & (y[dst] != -1)
    src = src[mask]
    dst = dst[mask]
    same = (y[src] == y[dst]).float()
    return same.mean().item() if same.numel() > 0 else 0.0


def build_temporal_edges(num_nodes: int, window: int = 5) -> torch.Tensor:
    """
    Build a simple directed temporal graph:
      For each transaction i, connect to the next 'window' transactions in time:
        i -> i+1, i -> i+2, ..., i -> i+window
    Assumes the rows are already sorted by Time.
    """
    src_list = []
    dst_list = []

    for offset in range(1, window + 1):
        # i goes from 0 to N - offset - 1
        src = np.arange(0, num_nodes - offset, dtype=np.int64)
        dst = np.arange(offset, num_nodes, dtype=np.int64)
        src_list.append(src)
        dst_list.append(dst)

    src_all = np.concatenate(src_list, axis=0)
    dst_all = np.concatenate(dst_list, axis=0)

    edge_index = torch.tensor([src_all, dst_all], dtype=torch.long)
    return edge_index


class ULBCCFDataset(InMemoryDataset):
    """
    ULB Credit Card Fraud Detection Dataset as a transaction graph.

    Expected raw file:
        {root}/raw/creditcard.csv

    Processing steps:
      - Sort transactions by Time
      - Node features: standardized [V1..V28, Amount]
      - Labels: Class (0 = normal, 1 = fraud)
      - Directed edges: temporal edges i -> i+1..i+window
      - Splits: temporal 60/20/20 on Time (like Elliptic)
    """

    def __init__(self, root: str, window: int = 5, transform=None, pre_transform=None):
        self.window = window
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ["creditcard.csv"]

    @property
    def processed_file_names(self):
        return ["ulb_credit.pt"]

    def download(self):
        # Nothing to download automatically. User must place creditcard.csv in raw/.
        pass

    def process(self):
        raw_path = os.path.join(self.raw_dir, "creditcard.csv")
        if not os.path.exists(raw_path):
            raise FileNotFoundError(
                f"Expected raw file at {raw_path}.\n"
                f"Download the ULB Credit Card Fraud Detection dataset "
                f"and place 'creditcard.csv' there."
            )

        df = pd.read_csv(raw_path)

        # Columns: Time, V1..V28, Amount, Class
        # Sort by Time to respect temporal order
        df = df.sort_values("Time").reset_index(drop=True)

        # Node features: V1..V28 + Amount
        feature_cols = [f"V{i}" for i in range(1, 29)] + ["Amount"]
        x_vals = df[feature_cols].values.astype(np.float32)

        # Standardize features
        x_mean = x_vals.mean(axis=0, keepdims=True)
        x_std = x_vals.std(axis=0, keepdims=True) + 1e-6
        x_vals = (x_vals - x_mean) / x_std

        x = torch.tensor(x_vals, dtype=torch.float32)

        # Labels
        y = torch.tensor(df["Class"].values.astype(np.int64), dtype=torch.long)

        # Time as tensor
        time_vals = torch.tensor(
            df["Time"].values.astype(np.float32), dtype=torch.float32
        )

        num_nodes = x.size(0)

        # Build directed temporal edges
        edge_index = build_temporal_edges(num_nodes=num_nodes, window=self.window)

        # Optional simple edge features: time difference and amount of dst
        src, dst = edge_index
        time_diff = (time_vals[dst] - time_vals[src]).unsqueeze(-1)  # [E,1]
        amount = torch.tensor(
            df["Amount"].values.astype(np.float32), dtype=torch.float32
        )
        amount_dst = amount[dst].unsqueeze(-1)  # [E,1]
        edge_attr = torch.cat([time_diff, amount_dst], dim=-1)  # [E,2]

        # Temporal splits 60/20/20 (on all nodes)
        uniq_times = torch.sort(time_vals.unique())[0]
        t1 = uniq_times[int(0.6 * len(uniq_times))]
        t2 = uniq_times[int(0.8 * len(uniq_times))]

        train_mask = time_vals <= t1
        val_mask = (time_vals > t1) & (time_vals <= t2)
        test_mask = time_vals > t2

        data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=y,
            time=time_vals,
            num_nodes=num_nodes,
            train_mask=train_mask,
            val_mask=val_mask,
            test_mask=test_mask,
        )

        # Print a quick summary
        num_fraud = int((y == 1).sum().item())
        num_norm = int((y == 0).sum().item())
        total_labeled = num_fraud + num_norm
        fraud_ratio = num_fraud / max(total_labeled, 1)
        h = compute_edge_homophily(edge_index, y)

        print(
            f"[ULBCCFDataset] Nodes={num_nodes}, Edges={edge_index.size(1)}, "
            f"Fraud ratio={fraud_ratio:.4f}, Edge homophily={h:.4f}"
        )

        data_list = [data]
        data, slices = self.collate(data_list)
        os.makedirs(self.processed_dir, exist_ok=True)
        torch.save((data, slices), self.processed_paths[0])


def get_ulb_credit_dataset(path: str) -> ULBCCFDataset:
    """
    Helper for data_loading.py.
    """
    dataset_root = os.path.join(path, "ulb_credit")
    return ULBCCFDataset(root=dataset_root)

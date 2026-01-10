import os
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.utils import degree
import torch.nn.functional as F


class OnlinePaymentsDataset(InMemoryDataset):
    """
    PaySim / Online Payments Fraud dataset as a node-classification graph.

    Nodes   = accounts (nameOrig / nameDest)
    Edges   = transactions (nameOrig -> nameDest)
    Labels  = 1 if an account has at least one fraudulent outgoing transaction
              0 otherwise

    Node features (per account, aggregated):
      - out_deg, in_deg
      - total_out_amount, total_in_amount
      - avg_out_amount, avg_in_amount
      - fraud_out_ratio, fraud_in_ratio
      - 4 Gaussian noise dimensions (to make feature space richer)

    Edge features (per transaction):
      - log(1 + amount)
      - normalized time step
      - one-hot transaction type
      - isFlaggedFraud (0/1)

    We also **downsample the majority (non-fraud) class** at the
    transaction level to avoid extreme class imbalance.
    """

    def __init__(
        self,
        root: str,
        max_rows: Optional[int] = 300_000,
        max_normal_per_fraud: Optional[int] = 5,
        transform=None,
        pre_transform=None,
    ):
        """
        Args
        ----
        max_rows: optional hard cap on total rows after rebalancing.
        max_normal_per_fraud: for every 1 fraud transaction, keep at most
                              `max_normal_per_fraud` normal transactions.
                              If None, no class-based downsampling is done.
        """
        self.max_rows = max_rows
        self.max_normal_per_fraud = max_normal_per_fraud
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    # ------------------------------------------------------------------ #
    # PyG bookkeeping
    # ------------------------------------------------------------------ #
    @property
    def raw_file_names(self):
        # Put the CSV here and rename it to this filename
        return ["online_payments.csv"]

    @property
    def processed_file_names(self):
        # NOTE: kept simple; if you change max_rows/max_normal_per_fraud,
        # delete this file to regenerate.
        return ["online_payments_graph.pt"]

    def download(self):
        # we don't auto-download; user must place the CSV manually
        raise RuntimeError(
            "Please download the PaySim / Online Payments CSV and place it at "
            f"{os.path.join(self.raw_dir, self.raw_file_names[0])}."
        )

    # ------------------------------------------------------------------ #
    # Processing
    # ------------------------------------------------------------------ #
    def process(self):
        raw_path = os.path.join(self.raw_dir, self.raw_file_names[0])
        if not os.path.exists(raw_path):
            raise FileNotFoundError(
                f"Expected raw file at {raw_path}.\n"
                "Download the dataset from Kaggle (PaySim / Online Payments) "
                "and save it as 'online_payments.csv' in that folder."
            )

        print("Processing Online Payments Fraud dataset...")
        df = pd.read_csv(raw_path)

        # -------------------------------------------------------------- #
        # 1) Class-based downsampling (keep all fraud, limit non-fraud)
        # -------------------------------------------------------------- #
        if self.max_normal_per_fraud is not None:
            fraud_df = df[df["isFraud"] == 1]
            normal_df = df[df["isFraud"] == 0]

            n_fraud = len(fraud_df)
            if n_fraud == 0:
                raise ValueError("Dataset contains no fraudulent transactions.")

            n_normal_keep = min(len(normal_df), n_fraud * self.max_normal_per_fraud)

            # Sample majority class deterministically
            normal_df = normal_df.sample(n=n_normal_keep, random_state=42)

            df = (
                pd.concat([fraud_df, normal_df])
                .sample(frac=1.0, random_state=42)
                .reset_index(drop=True)
            )

            print(
                f"  After class rebalancing: {len(df)} rows "
                f"(fraud={len(fraud_df)}, normal={n_normal_keep}, "
                f"ratio ≈ {len(fraud_df) / len(df):.4f})"
            )

        # -------------------------------------------------------------- #
        # 2) Optional global row cap
        # -------------------------------------------------------------- #
        if self.max_rows is not None and len(df) > self.max_rows:
            df = df.sample(n=self.max_rows, random_state=42).reset_index(drop=True)
            print(f"  Downsampled to {self.max_rows} rows (post-balance).")

        # Ensure important columns exist
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
            raise ValueError(f"Missing required columns: {missing}")

        # ------------------------------------------------------------------ #
        # Build node index: accounts from nameOrig ∪ nameDest
        # ------------------------------------------------------------------ #
        all_accounts = pd.Index(pd.concat([df["nameOrig"], df["nameDest"]]).unique())
        acc_to_idx = {acc: i for i, acc in enumerate(all_accounts)}
        num_nodes = len(all_accounts)
        num_edges = len(df)

        src_idx = df["nameOrig"].map(acc_to_idx).to_numpy()
        dst_idx = df["nameDest"].map(acc_to_idx).to_numpy()

        edge_index = torch.tensor(
            np.vstack([src_idx, dst_idx]), dtype=torch.long
        )  # [2, E]

        # ------------------------------------------------------------------ #
        # Edge features
        # ------------------------------------------------------------------ #
        amount = torch.tensor(df["amount"].to_numpy(), dtype=torch.float32)
        step = torch.tensor(df["step"].to_numpy(), dtype=torch.float32)
        is_flagged = torch.tensor(df["isFlaggedFraud"].to_numpy(), dtype=torch.float32)

        # log amount to reduce skew
        log_amount = torch.log1p(amount).unsqueeze(-1)  # [E,1]

        # normalize step to [0,1]
        step_norm = (step / (step.max() + 1e-6)).unsqueeze(-1)  # [E,1]

        # transaction type one-hot
        type_cat = df["type"].astype("category")
        type_codes = torch.tensor(type_cat.cat.codes.to_numpy(), dtype=torch.long)
        num_types = len(type_cat.cat.categories)
        type_one_hot = F.one_hot(type_codes, num_classes=num_types).float()  # [E,T]

        is_flagged_feat = is_flagged.unsqueeze(-1)  # [E,1]

        edge_attr = torch.cat(
            [log_amount, step_norm, type_one_hot, is_flagged_feat], dim=-1
        )  # [E, 2+T+1]

        # ------------------------------------------------------------------ #
        # Node features via aggregation
        # ------------------------------------------------------------------ #
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
        )  # [N, 8 + 4 = 12]

        # ------------------------------------------------------------------ #
        # Node labels: account is fraud if any outgoing tx is fraud
        # ------------------------------------------------------------------ #
        fraud_out_flag = fraud_out_count > 0.0
        y = fraud_out_flag.long()  # [N]

        fraud_ratio_nodes = float(y.float().mean().item())
        fraud_ratio_edges = float(is_fraud_edge.mean().item())

        data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=y,
            num_nodes=num_nodes,
        )

        print("=== Online Payments Fraud Graph Info ===")
        print(f"Num nodes           : {data.num_nodes}")
        print(f"Num edges           : {data.num_edges}")
        print(f"Num node feats      : {data.num_features}")
        print(f"Num edge feats      : {edge_attr.size(-1)}")
        print(f"Fraud ratio (edges) : {fraud_ratio_edges:.4f}")
        print(f"Fraud ratio (nodes) : {fraud_ratio_nodes:.4f}")

        data_list = [data]
        data, slices = self.collate(data_list)
        os.makedirs(self.processed_dir, exist_ok=True)
        torch.save((data, slices), self.processed_paths[0])


def get_online_payments_dataset(
    path: str,
    max_rows: Optional[int] = 300_000,
    max_normal_per_fraud: Optional[int] = 5,
):
    dataset_root = os.path.join(path, "online_payments")
    return OnlinePaymentsDataset(
        root=dataset_root,
        max_rows=max_rows,
        max_normal_per_fraud=max_normal_per_fraud,
    )

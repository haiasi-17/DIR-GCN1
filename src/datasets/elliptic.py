import os
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data


class EllipticDataset:
    """
    Lightweight wrapper so we can use dataset._data like others.

    - Expects raw CSVs in: {root}/raw/
        * elliptic_txs_features.csv
        * elliptic_txs_classes.csv
        * elliptic_txs_edgelist.csv

    - Creates: {root}/processed/elliptic.pt
    """

    def __init__(self, root: str):
        self.root = root
        self.raw_dir = os.path.join(root, "raw")
        self.processed_dir = os.path.join(root, "processed")
        os.makedirs(self.processed_dir, exist_ok=True)

        proc_path = os.path.join(self.processed_dir, "elliptic.pt")
        if not os.path.exists(proc_path):
            print("Processing Elliptic dataset...")
            data = self._process()
            torch.save(data, proc_path)
            print("Done!")
        else:
            data = torch.load(proc_path)

        self._data = data

    @property
    def num_classes(self) -> int:
        # binary: 0 = licit, 1 = illicit
        return 2

    # ------------------------------------------------------------------
    # Internal processing from raw CSV
    # ------------------------------------------------------------------
    def _process(self) -> Data:
        feats_path = os.path.join(self.raw_dir, "elliptic_txs_features.csv")
        classes_path = os.path.join(self.raw_dir, "elliptic_txs_classes.csv")
        edges_path = os.path.join(self.raw_dir, "elliptic_txs_edgelist.csv")

        # ---------- Features ----------
        # Official format: col0 = txId, col1 = time_step, col2.. = features
        feats_df = pd.read_csv(feats_path, header=None)
        tx_ids = feats_df.iloc[:, 0].values
        time_steps = feats_df.iloc[:, 1].values
        x_vals = feats_df.iloc[:, 2:].values.astype(np.float32)

        # ---------- Classes ----------
        # columns: txId, class
        # class values: 'unknown', '1' (licit), '2' (illicit)
        cls_df = pd.read_csv(classes_path)
        cls_map = cls_df.set_index("txId")["class"].to_dict()

        labels = []
        for tx in tx_ids:
            c = cls_map.get(tx, "unknown")
            if c == "unknown":
                labels.append(-1)
            elif str(c) == "1":
                labels.append(0)  # licit
            elif str(c) == "2":
                labels.append(1)  # illicit
            else:
                labels.append(-1)
        y = torch.tensor(labels, dtype=torch.long)

        # ---------- Edges ----------
        edges_df = pd.read_csv(edges_path)
        # columns: txId1, txId2 (directed)
        src_ids = edges_df.iloc[:, 0].values
        dst_ids = edges_df.iloc[:, 1].values

        # Map txId -> node index
        id_to_idx = {tx_id: i for i, tx_id in enumerate(tx_ids)}

        src_list = []
        dst_list = []
        for s, d in zip(src_ids, dst_ids):
            if s in id_to_idx and d in id_to_idx:
                src_list.append(id_to_idx[s])
                dst_list.append(id_to_idx[d])

        edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)

        x = torch.tensor(x_vals, dtype=torch.float32)
        time_tensor = torch.tensor(time_steps, dtype=torch.long)

        num_nodes = x.size(0)

        # ---------- Temporal train/val/test split on labeled nodes ----------
        labeled_mask = y != -1
        labeled_times = time_tensor[labeled_mask]

        uniq_times = torch.sort(labeled_times.unique())[0]
        # 60% / 20% / 20% temporal split
        t1 = uniq_times[int(0.6 * len(uniq_times))]
        t2 = uniq_times[int(0.8 * len(uniq_times))]

        train_mask = (time_tensor <= t1) & (y != -1)
        val_mask = (time_tensor > t1) & (time_tensor <= t2) & (y != -1)
        test_mask = (time_tensor > t2) & (y != -1)

        data = Data(
            x=x,
            edge_index=edge_index,
            y=y,
            time=time_tensor,
            num_nodes=num_nodes,
            train_mask=train_mask,
            val_mask=val_mask,
            test_mask=test_mask,
        )

        return data

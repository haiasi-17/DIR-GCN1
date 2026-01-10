import os
from typing import Tuple

import numpy as np
import torch
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.utils import degree


# ---------------------------------------------------------
# Homophily tools
# ---------------------------------------------------------
def compute_edge_homophily(edge_index: torch.Tensor, y: torch.Tensor) -> float:
    """
    Fraction of edges whose endpoints share the same label.
    """
    src, dst = edge_index
    same = (y[src] == y[dst]).float()
    return same.mean().item()


def enforce_low_homophily(
    edge_index: torch.Tensor,
    y: torch.Tensor,
    target_h: float = 0.25,
    max_iters: int = 7,
    flip_fraction: float = 0.06,
) -> torch.Tensor:
    """
    Iteratively flip labels of nodes that have many same-label neighbors
    until edge homophily <= target_h or max_iters is reached.
    """
    src, dst = edge_index
    y = y.clone()

    for _ in range(max_iters):
        h = compute_edge_homophily(edge_index, y)
        if h <= target_h:
            break

        # count how many same-label neighbors each node has
        same_edges = y[src] == y[dst]
        counts = torch.zeros_like(y, dtype=torch.float32)

        # for each same-label edge (u, v), increment counts[u] and counts[v]
        counts.index_add_(
            0, src[same_edges], torch.ones_like(src[same_edges], dtype=torch.float32)
        )
        counts.index_add_(
            0, dst[same_edges], torch.ones_like(dst[same_edges], dtype=torch.float32)
        )

        # flip labels of top-k nodes with highest same-label counts
        k = int(flip_fraction * y.numel())
        if k <= 0:
            break

        _, idx = counts.view(-1).topk(k)
        y[idx] = 1 - y[idx]  # flip 0↔1

    return y


# ---------------------------------------------------------
# Hard synthetic fraud graph
# ---------------------------------------------------------
def generate_synthetic_fraud_graph(
    num_accounts: int = 5000,
    base_fraud_ratio: float = 0.1,
    avg_out_degree: int = 8,
    heterophily_level: float = 0.3,
    num_repetitive_patterns: int = 400,
    initial_label_noise: float = 0.2,
    seed: int = 0,
) -> Tuple[Data, float]:
    """
    Create a harder directed transaction graph for fraud detection.

    Design:
      - Step 1: create latent risk profiles and a preliminary label y_pre.
      - Step 2: generate transactions and features using y_pre and profiles.
      - Step 3: build node features x by aggregation.
      - Step 4: re-label nodes as "fraud" from x via a noisy, non-linear risk score.
               -> final y is NOT trivially recoverable from any single feature.
      - Step 5: enforce low homophily on the final labels.
    """
    rng = np.random.default_rng(seed)

    # -----------------------------
    # 1) Latent risk profiles + preliminary labels
    # -----------------------------
    # 3 risk profiles; but we don't use them as final labels
    profiles = rng.choice(3, size=num_accounts, p=[0.5, 0.3, 0.2])
    fraud_probs = np.array([0.05, 0.18, 0.40])
    y_pre = rng.binomial(1, fraud_probs[profiles]).astype(np.int64)

    # Adjust approx overall fraud rate toward base_fraud_ratio
    current_ratio = y_pre.mean()
    if current_ratio > 0:
        scale = base_fraud_ratio / current_ratio
        keep_fraud = rng.random(num_accounts) < scale
        y_pre = np.where(y_pre == 1, keep_fraud.astype(np.int64), 0)

    # Add some initial label noise (only used to guide graph generation)
    noise_mask = rng.random(num_accounts) < initial_label_noise
    y_pre[noise_mask] = 1 - y_pre[noise_mask]

    # We'll use y_pre (0/1) only for edge generation, not as final label
    # (y_pre_t kept only if you want to debug)
    y_pre_t = torch.from_numpy(y_pre)

    # -----------------------------
    # 2) Directed edges with controlled heterophily
    # -----------------------------
    num_edges = num_accounts * avg_out_degree
    src_list = []
    dst_list = []
    edge_amounts = []
    edge_times = []
    edge_channels = []
    edge_repetitive_flag = []

    def sample_dst(u_label: int) -> int:
        """
        Sample a destination given the preliminary label u_label.

        heterophily_level ~ P(same-label edge)
        """
        if rng.random() < heterophily_level:
            # same label
            candidates = np.where(y_pre == u_label)[0]
        else:
            # different label
            candidates = np.where(y_pre != u_label)[0]
        if len(candidates) == 0:
            candidates = np.arange(num_accounts)
        return int(rng.choice(candidates))

    for _ in range(num_edges):
        u = int(rng.integers(0, num_accounts))
        u_label = int(y_pre[u])
        v = sample_dst(u_label)

        src_list.append(u)
        dst_list.append(v)

        prof_u = profiles[u]
        prof_v = profiles[v]
        risk_level = max(prof_u, prof_v)

        # Amount distributions – heavily overlapping
        base_means = np.array([3.8, 4.0, 4.2])
        base_stds = np.array([0.9, 0.9, 1.0])
        mu = base_means[risk_level]
        sigma = base_stds[risk_level]

        # slight uplift if either endpoint is prelim. fraud
        if y_pre[u] == 1 or y_pre[v] == 1:
            mu += 0.2

        amt = float(np.exp(rng.normal(mu, sigma)))

        # Time bucket (fraud slightly more often at night, but noisy)
        t_bucket = int(rng.integers(0, 24))
        if rng.random() < 0.3 and (y_pre[u] == 1 or y_pre[v] == 1):
            t_bucket = int(rng.choice([0, 1, 2, 3, 4, 5, 22, 23]))

        # Channel
        channel = int(rng.integers(0, 3))

        edge_amounts.append(amt)
        edge_times.append(t_bucket)
        edge_channels.append(channel)
        edge_repetitive_flag.append(0.0)

    # -----------------------------
    # 3) Repetitive motifs (not purely fraud)
    # -----------------------------
    centers = rng.choice(
        num_accounts, size=min(num_repetitive_patterns, num_accounts), replace=False
    )
    for c in centers:
        neighs = rng.choice(num_accounts, size=2, replace=False)
        a, b = int(neighs[0]), int(neighs[1])

        base_amt = float(np.exp(rng.normal(4.5, 0.4)))
        base_time = int(rng.integers(0, 24))
        channel = int(rng.integers(0, 3))

        motif_edges = [(c, a), (a, b), (b, c)]

        for u, v in motif_edges:
            src_list.append(u)
            dst_list.append(v)
            edge_amounts.append(base_amt)
            edge_times.append(base_time)
            edge_channels.append(channel)
            edge_repetitive_flag.append(1.0)

    edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)

    # -----------------------------
    # 4) Edge features
    # -----------------------------
    amounts = torch.tensor(edge_amounts, dtype=torch.float32).unsqueeze(-1)
    times = torch.tensor(edge_times, dtype=torch.float32).unsqueeze(-1)

    channels = torch.tensor(edge_channels, dtype=torch.long)
    channel_one_hot = torch.nn.functional.one_hot(channels, num_classes=3).float()

    repetitive = torch.tensor(edge_repetitive_flag, dtype=torch.float32).unsqueeze(-1)

    times_norm = times / 23.0

    edge_attr = torch.cat(
        [
            amounts,  # [E,1]
            times_norm,  # [E,1]
            channel_one_hot,  # [E,3]
            repetitive,  # [E,1]
        ],
        dim=-1,
    )  # [E,6]

    # -----------------------------
    # 5) Node features via aggregation
    # -----------------------------
    num_nodes = num_accounts
    src = edge_index[0]
    dst = edge_index[1]

    in_deg = degree(dst, num_nodes=num_nodes, dtype=torch.float32).unsqueeze(-1)
    out_deg = degree(src, num_nodes=num_nodes, dtype=torch.float32).unsqueeze(-1)

    edge_amounts_t = amounts.squeeze(-1)
    total_in_amt = torch.zeros(num_nodes, dtype=torch.float32)
    total_out_amt = torch.zeros(num_nodes, dtype=torch.float32)
    total_in_amt.index_add_(0, dst, edge_amounts_t)
    total_out_amt.index_add_(0, src, edge_amounts_t)
    total_in_amt = total_in_amt.unsqueeze(-1)
    total_out_amt = total_out_amt.unsqueeze(-1)

    avg_in_amt = total_in_amt / (in_deg + 1e-6)
    avg_out_amt = total_out_amt / (out_deg + 1e-6)

    rep = repetitive.squeeze(-1)
    rep_in = torch.zeros(num_nodes, dtype=torch.float32)
    rep_out = torch.zeros(num_nodes, dtype=torch.float32)
    rep_in.index_add_(0, dst, rep)
    rep_out.index_add_(0, src, rep)
    frac_rep_in = (rep_in / (in_deg.squeeze(-1) + 1e-6)).unsqueeze(-1)
    frac_rep_out = (rep_out / (out_deg.squeeze(-1) + 1e-6)).unsqueeze(-1)

    night_mask = (times.squeeze(-1) < 6) | (times.squeeze(-1) >= 22)
    night = night_mask.to(torch.float32)
    night_in = torch.zeros(num_nodes, dtype=torch.float32)
    night_out = torch.zeros(num_nodes, dtype=torch.float32)
    night_in.index_add_(0, dst, night)
    night_out.index_add_(0, src, night)
    frac_night_in = (night_in / (in_deg.squeeze(-1) + 1e-6)).unsqueeze(-1)
    frac_night_out = (night_out / (out_deg.squeeze(-1) + 1e-6)).unsqueeze(-1)

    # extra noise features
    noise_feat = torch.randn(num_nodes, 4)

    x = torch.cat(
        [
            in_deg,
            out_deg,
            total_in_amt,
            total_out_amt,
            avg_in_amt,
            avg_out_amt,
            frac_rep_in,
            frac_rep_out,
            frac_night_in,
            frac_night_out,
            noise_feat,
        ],
        dim=-1,
    )  # [N, 14]

    # -----------------------------
    # 6) Final labels from x via noisy, non-linear risk score
    #    + post-process to enforce low homophily
    # -----------------------------
    with torch.no_grad():
        # standardize features
        x_mean = x.mean(dim=0, keepdim=True)
        x_std = x.std(dim=0, keepdim=True) + 1e-6
        x_norm = (x - x_mean) / x_std

        d = x_norm.shape[1]
        # fixed random weights (seeded via numpy -> deterministic given seed)
        w_np = rng.normal(0.0, 0.6, size=d).astype(np.float32)
        w = torch.from_numpy(w_np)

        # non-linear combination: risk depends more on repetitive + night behavior
        # indices: [0]in_deg [1]out_deg [2]tot_in [3]tot_out [4]avg_in [5]avg_out
        #         [6]frac_rep_in [7]frac_rep_out [8]frac_night_in [9]frac_night_out [10..13]noise
        weights_boost = torch.zeros_like(w)
        for idx in [6, 7, 8, 9]:
            weights_boost[idx] = 1.0

        w_eff = w + weights_boost
        risk_score = (x_norm * w_eff).sum(dim=1)

        # extra noise so it's not trivially separable
        risk_score = risk_score + 0.8 * torch.randn_like(risk_score)

        # threshold so ~10% are fraud
        thresh = torch.quantile(risk_score, 0.9)
        y_final = (risk_score > thresh).long()

    # enforce low homophily on the final labels
    y = enforce_low_homophily(
        edge_index=edge_index,
        y=y_final,
        target_h=0.3,  # adjust if you want slightly higher/lower homophily
        max_iters=5,
        flip_fraction=0.05,  # ~5% of nodes flipped per iteration
    )

    data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=y,
        num_nodes=num_nodes,
    )

    h = compute_edge_homophily(edge_index, y)
    return data, h


# ---------------------------------------------------------
# Dataset wrapper
# ---------------------------------------------------------
class SyntheticFraudDataset(InMemoryDataset):
    """
    Synthetic fraud dataset with harder classification task.
    """

    def __init__(
        self,
        root: str,
        num_accounts: int = 5000,
        base_fraud_ratio: float = 0.1,
        transform=None,
        pre_transform=None,
    ):
        self._num_accounts = num_accounts
        self._base_fraud_ratio = base_fraud_ratio
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        # v4 -> new filename so PyG knows to regenerate
        return ["fraud_data_v4.pt"]

    def download(self):
        pass

    def process(self):
        data, h = generate_synthetic_fraud_graph(
            num_accounts=self._num_accounts,
            base_fraud_ratio=self._base_fraud_ratio,
        )
        fraud_ratio = float(data.y.float().mean().item())
        print(
            f"[SyntheticFraudDataset] Edge homophily = {h:.4f}, "
            f"fraud_ratio = {fraud_ratio:.4f}"
        )

        data_list = [data]
        data, slices = self.collate(data_list)
        os.makedirs(self.processed_dir, exist_ok=True)
        torch.save((data, slices), self.processed_paths[0])


def get_synthetic_fraud_dataset(path: str) -> SyntheticFraudDataset:
    dataset_root = os.path.join(path, "syn-fraud")
    return SyntheticFraudDataset(root=dataset_root)

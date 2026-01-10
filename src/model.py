import time

import numpy as np
import pandas as pd

import torch
from torch import nn, optim
import pytorch_lightning as pl

from torch_sparse import SparseTensor
import torch.nn.functional as F
from torch.nn import ModuleList, Linear
from torch_geometric.nn import JumpingKnowledge
from torch_geometric.utils import degree
from torch_scatter import scatter_add

from torchmetrics.classification import (
    MulticlassF1Score,
    MulticlassPrecision,
    MulticlassRecall,
)

from src.datasets.data_utils import get_norm_adj


# ---------------------------------------------------------------------------
# Convolution selector
# ---------------------------------------------------------------------------


def get_conv(
    conv_type: str,
    input_dim: int,
    output_dim: int,
    alpha,
    lcs_threshold: float = 0.0,
    enable_lcs_masking: bool = False,
):
    """
    Factory for convolution layers.

    Baseline:
      conv_type = "dir-gcn"       -> DirGCNConv

    Enhanced (paper-enhanced DIR-GCN):
      conv_type = "dir-gcn-gated" -> GatedDirGCNConv
        - Direction-specific aggregation and gated feature fusion
        - Local Contribution Score (LCS) Masking
        - Structural hash caching + recurrence diagnostics
        - MP-operation counter
        - Mini-batch Recurrence / Neighbor-Value sampling
    """
    if conv_type == "dir-gcn":
        return DirGCNConv(input_dim, output_dim, alpha)
    elif conv_type == "dir-gcn-gated":
        return GatedDirGCNConv(
            input_dim,
            output_dim,
            alpha,
            lcs_threshold=lcs_threshold,
            enable_lcs_masking=enable_lcs_masking,
        )
    else:
        raise ValueError(f"Convolution type {conv_type} not supported")


# ---------------------------------------------------------------------------
# Baseline Dir-GCN convolution (paper’s original layer)
# ---------------------------------------------------------------------------


class DirGCNConv(torch.nn.Module):
    """
    Baseline DIR-GCN layer (original directed GCN operator).

    Uses a global (possibly learnable) alpha to mix:
      - src→dst (forward / inbound) messages
      - dst→src (reverse / outbound) messages

    Added in this implementation:
      - mp_operation_stats: counts raw message-passing operations per layer
    """

    def __init__(self, input_dim, output_dim, alpha):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        self.lin_src_to_dst = Linear(input_dim, output_dim)
        self.lin_dst_to_src = Linear(input_dim, output_dim)
        self.alpha = alpha

        # Cached normalized adjacency for forward and reverse directions
        self.adj_norm = None
        self.adj_t_norm = None

        # Explicit MP-operation counter (no structural reuse here)
        self.mp_operation_stats = {
            "naive_ops": 0,  # how many MP ops a naive implementation would do
            "actual_ops": 0,  # actual ops (same as naive for baseline layer)
            "saved_ops": 0,
            "naive_aggregations": 0,
            "actual_aggregations": 0,
            "saved_aggregations": 0,
            "saved_ratio": 0.0,
            "num_forwards": 0,
        }

    def forward(self, x, edge_index):
        num_edges = edge_index.size(1)

        # Lazy construction of normalized adjacency matrices
        if self.adj_norm is None or self.adj_t_norm is None:
            row, col = edge_index
            num_nodes = x.size(0)

            adj = SparseTensor(row=row, col=col, sparse_sizes=(num_nodes, num_nodes))
            self.adj_norm = get_norm_adj(adj, norm="dir")

            adj_t = SparseTensor(row=col, col=row, sparse_sizes=(num_nodes, num_nodes))
            self.adj_t_norm = get_norm_adj(adj_t, norm="dir")

        h_forward = self.lin_src_to_dst(self.adj_norm @ x)
        h_reverse = self.lin_dst_to_src(self.adj_t_norm @ x)

        out = self.alpha * h_forward + (1.0 - self.alpha) * h_reverse

        # ---- Message-passing operation counter ----
        # 2 directional paths (in / out) per edge
        ops_this_layer = 2 * num_edges
        aggs_this_layer = num_edges  # 1 aggregation per edge per direction pair

        self.mp_operation_stats["naive_ops"] += ops_this_layer
        self.mp_operation_stats["actual_ops"] += ops_this_layer
        self.mp_operation_stats["naive_aggregations"] += aggs_this_layer
        self.mp_operation_stats["actual_aggregations"] += aggs_this_layer
        self.mp_operation_stats["saved_ops"] = 0
        self.mp_operation_stats["saved_aggregations"] = 0
        self.mp_operation_stats["saved_ratio"] = 0.0
        self.mp_operation_stats["num_forwards"] += 1

        return out


# ---------------------------------------------------------------------------
# Enhanced DIR-GCN with LCS Masking + Structural Hash Caching + Mini-batch
# ---------------------------------------------------------------------------


class GatedDirGCNConv(torch.nn.Module):
    """
    Enhanced DIR-GCN layer (paper’s enhanced DIR-GNN component).

    Adds on top of the baseline DIR-GCN:
      - Direction-aware message passing with separate inbound/outbound flows
      - Node-wise gating over inbound vs outbound messages
      - Local Contribution Score (LCS) Masking for neighbor importance
      - Structural hash caching of unique edges + recurrence statistics
      - Redundancy-eliminated message passing via unique (src,dst) edges
      - Explicit MP-operation counter with savings vs naive DIR-GCN
      - Mini-batch Recurrence / Neighbor-Value sampling

    Diagnostics (for Chapter 4 / Problem 3):
      - recurring_transaction_stats
          counts + ratio of recurring transactions (duplicate directed edges)
      - lcs_masking_stats
          counts + ratio of edges pruned by Local Contribution Score (LCS) Masking
      - mp_operation_stats
          naive vs actual message-passing operations, plus savings ratio
      - last_lcs_mask
          Bool[E_unique] of unique edges kept when masking is enabled
    """

    def __init__(
        self,
        input_dim,
        output_dim,
        alpha,
        lcs_threshold: float = 0.0,
        enable_lcs_masking: bool = False,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        # Direction-specific projections (same idea as baseline DIR-GCN)
        self.lin_src_to_dst = Linear(input_dim, output_dim)
        self.lin_dst_to_src = Linear(input_dim, output_dim)

        # Local Contribution Score (LCS) network
        # LCS_ij = sigmoid(lcs_mlp([x_i || x_j]))
        self.lcs_mlp = nn.Sequential(
            Linear(2 * input_dim, input_dim),
            nn.ReLU(),
            Linear(input_dim, 1),
        )

        # Node-wise gate over inbound / outbound aggregated messages
        # gate = sigmoid(gate_mlp([h_in || h_out]))
        self.gate_mlp = nn.Sequential(
            Linear(2 * output_dim, output_dim),
            nn.ReLU(),
            Linear(output_dim, 1),
        )

        # Residual connection on x
        self.residual = (
            Linear(input_dim, output_dim) if input_dim != output_dim else nn.Identity()
        )

        # Local Contribution Score (LCS) controls
        self.lcs_threshold = float(lcs_threshold)
        self.enable_lcs_masking = bool(enable_lcs_masking)

        # Diagnostics / cache handles (not saved in state_dict)
        self.last_lcs_mask = None  # Bool[E_unique]
        self.lcs_masking_stats = None  # LCS pruning statistics
        self.recurring_transaction_stats = None  # duplicate edge statistics

        # Structural hash cache (reused across epochs for fixed graph):
        #   - _cached_src, _cached_dst: original (possibly duplicated) edges
        #   - _unique_src, _unique_dst: unique (src,dst) edge set
        #   - _unique_counts: multiplicity for each unique edge
        #   - _num_edges_total / _num_unique_edges
        #   - _cached_in_deg / _cached_out_deg: degree counts (with recurrence)
        self._cached_src = None
        self._cached_dst = None
        self._cached_in_deg = None
        self._cached_out_deg = None

        self._unique_src = None
        self._unique_dst = None
        self._unique_counts = None
        self._num_edges_total = None
        self._num_unique_edges = None

        # Cached Local Contribution Score per UNIQUE edge (computed once)
        self._lcs_per_edge = None

        # Explicit MP-operation counter (compares naive vs cached)
        self.mp_operation_stats = {
            "naive_ops": 0,
            "actual_ops": 0,
            "saved_ops": 0,
            "naive_aggregations": 0,
            "actual_aggregations": 0,
            "saved_aggregations": 0,
            "saved_ratio": 0.0,
            "num_forwards": 0,
        }

        # Alpha kept for compatibility with get_conv API (not used directly)
        self.alpha = alpha

    # -------------------- helpers -------------------- #

    def _init_structure_cache(self, edge_index: torch.Tensor, num_nodes: int):
        """
        One-time structural preprocessing for this graph:

          - Cache src/dst index tensors
          - Cache structural hash on (src,dst) via unique() for reuse
          - Cache in/out degrees accounting for recurring edges
          - Compute recurring transaction statistics (duplicate edges)
        """
        src, dst = edge_index  # [E], [E]
        self._cached_src = src
        self._cached_dst = dst

        # Structural hash caching: group identical edges via unique()
        pairs = torch.stack([src, dst], dim=1)  # [E, 2]
        if pairs.numel() == 0:
            self._unique_src = src
            self._unique_dst = dst
            self._unique_counts = torch.zeros_like(src)
            self._num_edges_total = 0
            self._num_unique_edges = 0

            self._cached_in_deg = torch.ones(num_nodes)
            self._cached_out_deg = torch.ones(num_nodes)

            self.recurring_transaction_stats = {
                "num_edges": 0,
                "num_unique_edges": 0,
                "num_recurring_edges": 0,
                "recurring_ratio": 0.0,
                "cache_hits": 0,
                "cache_hit_ratio": 0.0,
            }
            return

        unique_pairs, inverse_indices, counts = pairs.unique(
            dim=0, return_inverse=True, return_counts=True
        )

        self._unique_src = unique_pairs[:, 0]
        self._unique_dst = unique_pairs[:, 1]
        self._unique_counts = counts
        self._num_edges_total = int(pairs.size(0))
        self._num_unique_edges = int(unique_pairs.size(0))

        # Degrees with recurrence: each duplicate edge contributes 1 to degree
        counts_f = counts.to(torch.float32)
        in_deg = scatter_add(
            counts_f, self._unique_dst, dim=0, dim_size=num_nodes
        ).clamp(min=1.0)
        out_deg = scatter_add(
            counts_f, self._unique_src, dim=0, dim_size=num_nodes
        ).clamp(min=1.0)
        self._cached_in_deg = in_deg
        self._cached_out_deg = out_deg

        # ---- Recurring transactions: duplicate directed edges (src, dst) ----
        num_recurring_edges = int(self._num_edges_total - self._num_unique_edges)
        recurring_ratio = (
            float(num_recurring_edges) / float(self._num_edges_total)
            if self._num_edges_total > 0
            else 0.0
        )
        self.recurring_transaction_stats = {
            "num_edges": self._num_edges_total,
            "num_unique_edges": self._num_unique_edges,
            "num_recurring_edges": num_recurring_edges,
            "recurring_ratio": recurring_ratio,
            # For Problem 3 Solution 3:
            "cache_hits": num_recurring_edges,
            "cache_hit_ratio": recurring_ratio,
        }

    # -------------------- Mini-Batch Recurrence / Neighbor-Value Sampler ---- #

    @torch.no_grad()
    def mini_batch_recurrence_neighbor_value_sampling(
        self,
        x: torch.Tensor,
        batch_nodes: torch.Tensor,
        num_neighbors: int = 10,
        recurrence_weight: float = 0.5,
        neighbor_value_weight: float = 0.5,
    ):
        """
        Implements the Recurrence / Neighbor-Value sampling heuristic
        described in Section 3.4.1 (Mini-Batch Sampling).

        IMPORTANT CHANGE:
        ------------------
        - We now reuse a *cached* LCS value per UNIQUE edge (self._lcs_per_edge)
          instead of recomputing LCS via an MLP every forward.
        - This makes the enhanced model much cheaper per epoch while still
          prioritizing high-contribution, recurring neighbors.

        Given:
          - x: [N, F] node features
          - batch_nodes: [B] seed node indices
        Returns:
          - sampled_edge_indices: 1D LongTensor indices into the UNIQUE edge set
            (self._unique_src / self._unique_dst) representing the edges to
            include for this mini-batch.
        """
        if self._unique_src is None or self._unique_dst is None:
            raise RuntimeError(
                "Structure cache not initialized. Call the layer at least once "
                "or call _init_structure_cache before using the sampler."
            )

        device = x.device
        batch_nodes = batch_nodes.to(device)

        src_u = self._unique_src.to(device)
        dst_u = self._unique_dst.to(device)
        counts = self._unique_counts.to(device).float()

        # 1) Neighbor mask: edges incident to any batch node
        incident_mask = (src_u.unsqueeze(0) == batch_nodes.unsqueeze(1)) | (
            dst_u.unsqueeze(0) == batch_nodes.unsqueeze(1)
        )
        incident_mask = incident_mask.any(dim=0)

        if incident_mask.sum() == 0:
            # No neighbors detected, return empty selection
            return torch.empty(0, dtype=torch.long, device=device)

        edge_idx_incident = torch.nonzero(incident_mask, as_tuple=False).view(-1)

        # 2) Recurrence Score (RS): normalized counts
        counts_incident = counts[edge_idx_incident]
        if counts_incident.numel() > 0:
            rs = counts_incident / counts_incident.max()
        else:
            rs = torch.zeros_like(counts_incident)

        # 3) Neighbor Value Score (NVS): based on cached Local Contribution Score
        if self._lcs_per_edge is None:
            # Fallback: compute once here (same as in forward)
            x_src_full = x[src_u]
            x_dst_full = x[dst_u]
            lcs_input_full = torch.cat([x_src_full, x_dst_full], dim=-1)
            self._lcs_per_edge = (
                torch.sigmoid(self.lcs_mlp(lcs_input_full)).squeeze(-1).detach()
            )

        nvs = self._lcs_per_edge.to(device)[edge_idx_incident]

        # 4) Combined importance
        total_w = recurrence_weight + neighbor_value_weight
        if total_w <= 0:
            total_w = 1.0
        rw = recurrence_weight / total_w
        nw = neighbor_value_weight / total_w

        combined_score = rw * rs + nw * nvs

        # 5) Top-k selection over incident edges
        k = min(num_neighbors * max(1, batch_nodes.numel()), edge_idx_incident.numel())
        _, topk_idx = torch.topk(combined_score, k=k, largest=True, sorted=False)

        sampled_edge_indices = edge_idx_incident[topk_idx]
        return sampled_edge_indices

    # -------------------- forward -------------------- #

    def forward(self, x, edge_index, batch_nodes: torch.Tensor = None):
        """
        x: [N, F]
        edge_index: [2, E] (directed edges, src->dst)
        batch_nodes: Optional[LongTensor] of seed nodes for mini-batch training.
                     If None, uses the full graph.

        This forward pass uses:
          - Structural hash caching of unique (src,dst) edges
          - Redundancy-eliminated message passing
          - Optional LCS Masking
          - Optional Recurrence/Neighbor-Value mini-batch sampling
          - Node-wise gated fusion of inbound and outbound flows
          - Residual connection
          - MP-operation counter
        """
        num_nodes = x.size(0)

        # Initialize / reuse structural cache
        if self._unique_src is None or self._unique_dst is None:
            self._init_structure_cache(edge_index, num_nodes)

        src_u = self._unique_src
        dst_u = self._unique_dst
        counts = self._unique_counts.float()

        # ---- Mini-batch sampling over UNIQUE edges (optional) ----
        if batch_nodes is not None:
            sampled_idx = self.mini_batch_recurrence_neighbor_value_sampling(
                x, batch_nodes
            )
            if sampled_idx.numel() == 0:
                # Fallback to use all edges if no edges sampled
                sampled_idx = torch.arange(
                    src_u.size(0), device=src_u.device, dtype=torch.long
                )
        else:
            sampled_idx = torch.arange(
                src_u.size(0), device=src_u.device, dtype=torch.long
            )

        src_sampled = src_u[sampled_idx]
        dst_sampled = dst_u[sampled_idx]
        counts_sampled = counts[sampled_idx]

        # --------------------------------------------------
        # 1) Local Contribution Score (LCS) for each UNIQUE sampled edge
        #    (CACHED: computed once from static node features)
        # --------------------------------------------------
        if self._lcs_per_edge is None:
            with torch.no_grad():
                x_src_full = x[src_u]
                x_dst_full = x[dst_u]
                lcs_input_full = torch.cat([x_src_full, x_dst_full], dim=-1)
                self._lcs_per_edge = (
                    torch.sigmoid(self.lcs_mlp(lcs_input_full)).squeeze(-1).detach()
                )  # [E_unique]

        # Re-index cached LCS for sampled edges
        lcs = self._lcs_per_edge.to(x.device)[sampled_idx]  # [E_eff]

        # --------------------------------------------------
        # 2) LCS Masking (optional redundancy control)
        # --------------------------------------------------
        # For MP-operation diagnostics we track two notions:
        #   - num_edges_total_effective: edges (with duplicates) in this sampled set
        num_edges_total_effective = int(counts_sampled.sum().item())

        if self.enable_lcs_masking and self.lcs_threshold > 0.0:
            mask = lcs >= self.lcs_threshold  # True = keep
            num_unique_total = int(lcs.numel())

            # Edge counts are defined on sampled unique edges
            edges_kept = int(counts_sampled[mask].sum().item())
            edges_pruned = int(num_edges_total_effective - edges_kept)

            # Avoid degenerate case where all edges are pruned
            if edges_kept == 0 and num_unique_total > 0:
                mask = torch.ones_like(lcs, dtype=torch.bool)
                edges_kept = num_edges_total_effective
                edges_pruned = 0

            # Save diagnostics for Problem 3 (LCS masking effectiveness)
            self.last_lcs_mask = mask.detach()
            self.lcs_masking_stats = {
                "num_edges_total": num_edges_total_effective,
                "num_edges_kept": edges_kept,
                "num_edges_pruned": edges_pruned,
                "pruned_ratio": float(edges_pruned) / float(num_edges_total_effective)
                if num_edges_total_effective > 0
                else 0.0,
                "threshold": float(self.lcs_threshold),
            }

            # Apply mask at the sampled UNIQUE-edge level
            src = src_sampled[mask]
            dst = dst_sampled[mask]
            lcs_kept = lcs[mask]
            counts_kept = counts_sampled[mask]
        else:
            self.last_lcs_mask = None
            self.lcs_masking_stats = None

            src = src_sampled
            dst = dst_sampled
            lcs_kept = lcs
            counts_kept = counts_sampled

        num_unique_effective = int(src.numel())

        # --------------------------------------------------
        # 3) Directional messages with LCS weighting
        #     + recurrence-based scaling via counts_kept
        # --------------------------------------------------
        x_src_eff = x[src]
        x_dst_eff = x[dst]

        msg_in = self.lin_src_to_dst(x_src_eff)  # src → dst (inbound direction)
        msg_out = self.lin_dst_to_src(x_dst_eff)  # dst → src (outbound direction)

        lcs_kept = lcs_kept.unsqueeze(-1)  # [E_eff, 1]
        counts_kept = counts_kept.unsqueeze(-1)  # [E_eff, 1]

        # weight by LCS and recurrence multiplicity
        msg_in = msg_in * lcs_kept * counts_kept
        msg_out = msg_out * lcs_kept * counts_kept

        # --------------------------------------------------
        # 4) Aggregate with degree normalization
        # --------------------------------------------------
        h_in = scatter_add(msg_in, dst, dim=0, dim_size=num_nodes)  # [N, D]
        h_out = scatter_add(msg_out, src, dim=0, dim_size=num_nodes)  # [N, D]

        # Recompute degrees on kept edges for mini-batch OR masking;
        # otherwise reuse full-graph cached degrees.
        if (batch_nodes is not None) or (
            self.enable_lcs_masking and self.lcs_threshold > 0.0
        ):
            counts_float = counts_kept.squeeze(-1)
            in_deg = scatter_add(counts_float, dst, dim=0, dim_size=num_nodes).clamp(
                min=1.0
            )
            out_deg = scatter_add(counts_float, src, dim=0, dim_size=num_nodes).clamp(
                min=1.0
            )
        else:
            in_deg = self._cached_in_deg
            out_deg = self._cached_out_deg

        h_in = h_in / in_deg.unsqueeze(-1)
        h_out = h_out / out_deg.unsqueeze(-1)

        # --------------------------------------------------
        # 5) Node-wise gate between inbound and outbound aggregations
        # --------------------------------------------------
        gate_input = torch.cat([h_in, h_out], dim=-1)  # [N, 2D]
        gate = torch.sigmoid(self.gate_mlp(gate_input))  # [N, 1]

        fused = gate * h_in + (1.0 - gate) * h_out  # [N, D]

        # --------------------------------------------------
        # 6) Residual connection
        # --------------------------------------------------
        res = self.residual(x)  # [N, D]
        out = fused + res

        # --------------------------------------------------
        # 7) MP-operation counter (naive vs structural hash caching)
        # --------------------------------------------------
        # naive: 2 * (# duplicate edges in this sampled subset)
        naive_ops_this = 2 * num_edges_total_effective
        # actual: 2 * (# unique sampled edges after masking and/or sampling)
        actual_ops_this = 2 * num_unique_effective

        self.mp_operation_stats["naive_ops"] += naive_ops_this
        self.mp_operation_stats["actual_ops"] += actual_ops_this

        # Aggregations = number of edges considered (effective vs unique)
        self.mp_operation_stats["naive_aggregations"] += num_edges_total_effective
        self.mp_operation_stats["actual_aggregations"] += num_unique_effective

        self.mp_operation_stats["saved_ops"] = (
            self.mp_operation_stats["naive_ops"] - self.mp_operation_stats["actual_ops"]
        )
        self.mp_operation_stats["saved_aggregations"] = (
            self.mp_operation_stats["naive_aggregations"]
            - self.mp_operation_stats["actual_aggregations"]
        )

        if self.mp_operation_stats["naive_aggregations"] > 0:
            self.mp_operation_stats["saved_ratio"] = float(
                self.mp_operation_stats["saved_aggregations"]
            ) / float(self.mp_operation_stats["naive_aggregations"])
        else:
            self.mp_operation_stats["saved_ratio"] = 0.0

        self.mp_operation_stats["num_forwards"] += 1

        return out


# ---------------------------------------------------------------------------
# Multi-layer GNN wrapper
# ---------------------------------------------------------------------------


class GNN(torch.nn.Module):
    def __init__(
        self,
        num_features,
        num_classes,
        hidden_dim,
        num_layers=2,
        dropout=0.0,
        conv_type="dir-gcn",
        jumping_knowledge=False,
        normalize=False,
        alpha=0.5,
        learn_alpha=False,
        lcs_threshold: float = 0.0,
        enable_lcs_masking: bool = False,
    ):
        super().__init__()

        self.num_classes = num_classes
        self.alpha = nn.Parameter(torch.ones(1) * alpha, requires_grad=learn_alpha)

        self.conv_type = conv_type
        self.lcs_threshold = lcs_threshold
        self.enable_lcs_masking = enable_lcs_masking

        output_dim = hidden_dim if jumping_knowledge else num_classes
        self.num_layers = num_layers
        self.dropout = dropout
        self.jumping_knowledge = jumping_knowledge
        self.normalize = normalize

        # Build convolution stack
        if num_layers == 1:
            self.convs = ModuleList(
                [
                    get_conv(
                        conv_type,
                        num_features,
                        output_dim,
                        self.alpha,
                        lcs_threshold=self.lcs_threshold,
                        enable_lcs_masking=self.enable_lcs_masking,
                    )
                ]
            )
        else:
            self.convs = ModuleList(
                [
                    get_conv(
                        conv_type,
                        num_features,
                        hidden_dim,
                        self.alpha,
                        lcs_threshold=self.lcs_threshold,
                        enable_lcs_masking=self.enable_lcs_masking,
                    )
                ]
            )
            for _ in range(num_layers - 2):
                self.convs.append(
                    get_conv(
                        conv_type,
                        hidden_dim,
                        hidden_dim,
                        self.alpha,
                        lcs_threshold=self.lcs_threshold,
                        enable_lcs_masking=self.enable_lcs_masking,
                    )
                )
            self.convs.append(
                get_conv(
                    conv_type,
                    hidden_dim,
                    output_dim,
                    self.alpha,
                    lcs_threshold=self.lcs_threshold,
                    enable_lcs_masking=self.enable_lcs_masking,
                )
            )

        # Jumping knowledge
        if jumping_knowledge is not None and jumping_knowledge:
            input_dim = (
                hidden_dim * num_layers if jumping_knowledge == "cat" else hidden_dim
            )
            self.lin = Linear(input_dim, num_classes)
            self.jump = JumpingKnowledge(
                mode=jumping_knowledge, channels=hidden_dim, num_layers=num_layers
            )
        else:
            self.lin = None
            self.jump = None

    def forward(self, x, edge_index, batch_nodes: torch.Tensor = None):
        xs = []
        for i, conv in enumerate(self.convs):
            # Only the enhanced layer knows how to use batch_nodes
            if isinstance(conv, GatedDirGCNConv):
                x = conv(x, edge_index, batch_nodes=batch_nodes)
            else:
                x = conv(x, edge_index)

            if i != len(self.convs) - 1 or self.jumping_knowledge:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
                if self.normalize:
                    x = F.normalize(x, p=2, dim=1)
            xs.append(x)

        if self.jump is not None:
            x = self.jump(xs)
            x = self.lin(x)

        return F.log_softmax(x, dim=1)


# ---------------------------------------------------------------------------
# Lightning wrappers with metrics
#   - Full-batch (baseline & optionally enhanced)
#   - Mini-batch for Enhanced DIR-GCN (dir-gcn-gated)
# ---------------------------------------------------------------------------


class LightingFullBatchModelWrapper(pl.LightningModule):
    def __init__(
        self, model, lr, weight_decay, train_mask, val_mask, test_mask, evaluator=None
    ):
        super().__init__()
        self.model = model
        self.lr = lr
        self.weight_decay = weight_decay
        self.evaluator = evaluator
        self.train_mask, self.val_mask, self.test_mask = (
            train_mask,
            val_mask,
            test_mask,
        )

        # ====== Metrics (macro over classes) ======
        num_classes = int(getattr(self.model, "num_classes", 2))

        # train metrics
        self.train_f1 = MulticlassF1Score(num_classes=num_classes, average="macro")
        self.train_prec = MulticlassPrecision(num_classes=num_classes, average="macro")
        self.train_rec = MulticlassRecall(num_classes=num_classes, average="macro")

        # val metrics
        self.val_f1 = MulticlassF1Score(num_classes=num_classes, average="macro")
        self.val_prec = MulticlassPrecision(num_classes=num_classes, average="macro")
        self.val_rec = MulticlassRecall(num_classes=num_classes, average="macro")

        # test metrics
        self.test_f1 = MulticlassF1Score(num_classes=num_classes, average="macro")
        self.test_prec = MulticlassPrecision(num_classes=num_classes, average="macro")
        self.test_rec = MulticlassRecall(num_classes=num_classes, average="macro")

    def forward(self, x, edge_index):
        # Full-batch: no mini-batching
        return self.model(x, edge_index, batch_nodes=None)

    def training_step(self, batch, batch_idx):
        x, y, edge_index = batch.x, batch.y.long(), batch.edge_index
        out = self.model(x, edge_index, batch_nodes=None)

        loss = F.nll_loss(out[self.train_mask], y[self.train_mask].squeeze())
        self.log("train_loss", loss)

        y_pred = out.argmax(dim=1)

        # ----- train metrics -----
        self.log(
            "train_f1",
            self.train_f1(y_pred[self.train_mask], y[self.train_mask].squeeze()),
            prog_bar=True,
        )
        self.log(
            "train_prec",
            self.train_prec(y_pred[self.train_mask], y[self.train_mask].squeeze()),
        )
        self.log(
            "train_rec",
            self.train_rec(y_pred[self.train_mask], y[self.train_mask].squeeze()),
        )

        train_acc = self.evaluate(
            y_pred=y_pred[self.train_mask], y_true=y[self.train_mask]
        )
        self.log("train_acc", train_acc, prog_bar=True)

        # ----- val metrics -----
        self.log(
            "val_f1",
            self.val_f1(y_pred[self.val_mask], y[self.val_mask].squeeze()),
            prog_bar=True,
        )
        self.log(
            "val_prec",
            self.val_prec(y_pred[self.val_mask], y[self.val_mask].squeeze()),
        )
        self.log(
            "val_rec",
            self.val_rec(y_pred[self.val_mask], y[self.val_mask].squeeze()),
        )

        val_acc = self.evaluate(y_pred=y_pred[self.val_mask], y_true=y[self.val_mask])
        self.log("val_acc", val_acc, prog_bar=True)

        return loss

    def evaluate(self, y_pred, y_true):
        if self.evaluator:
            acc = self.evaluator.eval(
                {"y_true": y_true, "y_pred": y_pred.unsqueeze(1)}
            )["acc"]
        else:
            acc = y_pred.eq(y_true.squeeze()).sum().item() / y_pred.shape[0]
        return acc

    def test_step(self, batch, batch_idx):
        x, y, edge_index = batch.x, batch.y.long(), batch.edge_index
        out = self.model(x, edge_index, batch_nodes=None)

        y_pred = out.argmax(dim=1)

        # ----- test metrics -----
        self.log(
            "test_f1",
            self.test_f1(y_pred[self.test_mask], y[self.test_mask].squeeze()),
        )
        self.log(
            "test_prec",
            self.test_prec(y_pred[self.test_mask], y[self.test_mask].squeeze()),
        )
        self.log(
            "test_rec",
            self.test_rec(y_pred[self.test_mask], y[self.test_mask].squeeze()),
        )

        test_acc = self.evaluate(
            y_pred=y_pred[self.test_mask], y_true=y[self.test_mask]
        )
        self.log("test_acc", test_acc)

    def configure_optimizers(self):
        optimizer = optim.Adam(
            self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        return optimizer


class LightingMiniBatchDirGatedWrapper(pl.LightningModule):
    """
    Lightning wrapper for Enhanced DIR-GCN (dir-gcn-gated) using
    Mini-batch Recurrence / Neighbor-Value sampling over seed nodes.

    Baseline remains full-batch; this is only used for conv_type='dir-gcn-gated'
    when --mini_batch_training is enabled.
    """

    def __init__(
        self,
        model,
        lr,
        weight_decay,
        train_mask,
        val_mask,
        test_mask,
        evaluator=None,
        mini_batch_size: int = 1024,
    ):
        super().__init__()
        self.model = model
        self.lr = lr
        self.weight_decay = weight_decay
        self.evaluator = evaluator

        self.train_mask = train_mask
        self.val_mask = val_mask
        self.test_mask = test_mask

        self.mini_batch_size = mini_batch_size

        # ====== Metrics (macro over classes) ======
        num_classes = int(getattr(self.model, "num_classes", 2))

        # train metrics
        self.train_f1 = MulticlassF1Score(num_classes=num_classes, average="macro")
        self.train_prec = MulticlassPrecision(num_classes=num_classes, average="macro")
        self.train_rec = MulticlassRecall(num_classes=num_classes, average="macro")

        # val metrics
        self.val_f1 = MulticlassF1Score(num_classes=num_classes, average="macro")
        self.val_prec = MulticlassPrecision(num_classes=num_classes, average="macro")
        self.val_rec = MulticlassRecall(num_classes=num_classes, average="macro")

        # test metrics
        self.test_f1 = MulticlassF1Score(num_classes=num_classes, average="macro")
        self.test_prec = MulticlassPrecision(num_classes=num_classes, average="macro")
        self.test_rec = MulticlassRecall(num_classes=num_classes, average="macro")

    def forward(self, x, edge_index, batch_nodes=None):
        return self.model(x, edge_index, batch_nodes=batch_nodes)

    def _ensure_indices(self, device):
        # Precompute node index tensors on the right device
        if not hasattr(self, "_train_idx") or self._train_idx.device != device:
            self._train_idx = torch.nonzero(
                self.train_mask.to(device), as_tuple=False
            ).view(-1)
            self._val_idx = torch.nonzero(
                self.val_mask.to(device), as_tuple=False
            ).view(-1)
            self._test_idx = torch.nonzero(
                self.test_mask.to(device), as_tuple=False
            ).view(-1)

    def training_step(self, batch, batch_idx):
        # batch is still the full graph, but we only train on a sampled subset
        x = batch.x.to(self.device)
        y = batch.y.long().view(-1).to(self.device)
        edge_index = batch.edge_index.to(self.device)

        self._ensure_indices(self.device)
        train_idx = self._train_idx

        # Sample mini-batch of seed nodes
        batch_size = min(self.mini_batch_size, train_idx.numel())
        perm = torch.randperm(train_idx.numel(), device=self.device)[:batch_size]
        batch_nodes = train_idx[perm]

        out = self.model(x, edge_index, batch_nodes=batch_nodes)

        loss = F.nll_loss(out[batch_nodes], y[batch_nodes])
        self.log("train_loss", loss)

        y_pred = out.argmax(dim=1)

        # ----- train metrics on mini-batch -----
        self.log(
            "train_f1",
            self.train_f1(y_pred[batch_nodes], y[batch_nodes]),
            prog_bar=True,
        )
        self.log(
            "train_prec",
            self.train_prec(y_pred[batch_nodes], y[batch_nodes]),
        )
        self.log(
            "train_rec",
            self.train_rec(y_pred[batch_nodes], y[batch_nodes]),
        )

        train_acc = self.evaluate(y_pred=y_pred[batch_nodes], y_true=y[batch_nodes])
        self.log("train_acc", train_acc, prog_bar=True)

        # ----- val metrics on full graph (as in full-batch wrapper) -----
        self.log(
            "val_f1",
            self.val_f1(y_pred[self._val_idx], y[self._val_idx]),
            prog_bar=True,
        )
        self.log(
            "val_prec",
            self.val_prec(y_pred[self._val_idx], y[self._val_idx]),
        )
        self.log(
            "val_rec",
            self.val_rec(y_pred[self._val_idx], y[self._val_idx]),
        )

        val_acc = self.evaluate(y_pred=y_pred[self._val_idx], y_true=y[self._val_idx])
        self.log("val_acc", val_acc, prog_bar=True)

        return loss

    def evaluate(self, y_pred, y_true):
        if self.evaluator:
            acc = self.evaluator.eval(
                {"y_true": y_true, "y_pred": y_pred.unsqueeze(1)}
            )["acc"]
        else:
            acc = y_pred.eq(y_true.squeeze()).sum().item() / y_pred.shape[0]
        return acc

    def test_step(self, batch, batch_idx):
        # Full-graph evaluation for test
        x = batch.x.to(self.device)
        y = batch.y.long().view(-1).to(self.device)
        edge_index = batch.edge_index.to(self.device)

        self._ensure_indices(self.device)

        out = self.model(x, edge_index, batch_nodes=None)
        y_pred = out.argmax(dim=1)

        self.log(
            "test_f1",
            self.test_f1(y_pred[self._test_idx], y[self._test_idx]),
        )
        self.log(
            "test_prec",
            self.test_prec(y_pred[self._test_idx], y[self._test_idx]),
        )
        self.log(
            "test_rec",
            self.test_rec(y_pred[self._test_idx], y[self._test_idx]),
        )

        test_acc = self.evaluate(
            y_pred=y_pred[self._test_idx], y_true=y[self._test_idx]
        )
        self.log("test_acc", test_acc)

    def configure_optimizers(self):
        optimizer = optim.Adam(
            self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        return optimizer


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def get_model(args):
    """
    args is expected to have:
      - num_features, num_classes
      - hidden_dim, num_layers, dropout
      - conv_type ("dir-gcn" or "dir-gcn-gated")
      - jk, normalize
      - alpha, learn_alpha
      - lcs_threshold, enable_lcs_masking  (for enhanced DIR-GCN)
    """
    return GNN(
        num_features=args.num_features,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_classes=args.num_classes,
        dropout=args.dropout,
        conv_type=args.conv_type,
        jumping_knowledge=args.jk,
        normalize=args.normalize,
        alpha=args.alpha,
        learn_alpha=args.learn_alpha,
        lcs_threshold=getattr(args, "lcs_threshold", 0.0),
        enable_lcs_masking=getattr(args, "enable_lcs_masking", False),
    )


# ---------------------------------------------------------------------------
# NEW: Helpers for Prediction Export, Inference Time, and Problem 3 Metrics
# ---------------------------------------------------------------------------


def export_node_predictions(model, data, output_path: str, class_names=None):
    """
    Export node-level predictions and probabilities to CSV.

    Columns:
      - node_idx
      - y_true (if available)
      - y_pred
      - prob_class_0 / prob_class_1 / ...  OR
        prob_<class_name> if class_names is provided.
    """
    model.eval()
    device = next(model.parameters()).device
    x = data.x.to(device)
    edge_index = data.edge_index.to(device)

    with torch.no_grad():
        log_probs = model(x, edge_index, batch_nodes=None)  # log-softmax
        probs = torch.exp(log_probs)

    num_nodes = x.size(0)
    num_classes = probs.size(1)

    y_true = None
    if getattr(data, "y", None) is not None:
        y_true = data.y.view(-1).cpu().numpy()

    y_pred = probs.argmax(dim=1).cpu().numpy()
    probs_np = probs.cpu().numpy()

    df_dict = {
        "node_idx": np.arange(num_nodes, dtype=int),
        "y_pred": y_pred,
    }
    if y_true is not None and len(y_true) == num_nodes:
        df_dict["y_true"] = y_true

    if class_names is not None and len(class_names) == num_classes:
        prob_cols = [f"prob_{c}" for c in class_names]
    else:
        prob_cols = [f"prob_class_{i}" for i in range(num_classes)]

    for j, col in enumerate(prob_cols):
        df_dict[col] = probs_np[:, j]

    df = pd.DataFrame(df_dict)
    df.to_csv(output_path, index=False)
    return df


def measure_inference_time(model, data, runs: int = 10):
    """
    Measure average inference time (forward pass) over the full graph.

    Returns:
      avg_time_seconds
    """
    model.eval()
    device = next(model.parameters()).device
    x = data.x.to(device)
    edge_index = data.edge_index.to(device)

    # Warmup
    with torch.no_grad():
        _ = model(x, edge_index, batch_nodes=None)

    runs = max(int(runs), 1)
    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(runs):
            _ = model(x, edge_index, batch_nodes=None)
    end = time.perf_counter()

    return (end - start) / runs


def collect_problem3_metrics(model):
    """
    Collect Problem 3-related diagnostics from all convolution layers.

    Returns a dict:
      {
        "per_layer": [ {layer metrics...}, ... ],
        "summary": {
            "total_naive_aggregations": ...,
            "total_actual_aggregations": ...,
            "total_saved_aggregations": ...,
            "total_saved_ratio": ...,
            "total_edges": ...,
            "total_recurring_edges": ...,
            "global_cache_hit_ratio": ...,
        }
      }

    Aggregation counts are normalised per forward (using num_forwards) so
    they can accumulate over training + diagnostics runs without changing
    the *per-forward* interpretation.
    """
    per_layer = []

    total_naive_aggs = 0.0
    total_actual_aggs = 0.0
    total_saved_aggs = 0.0

    total_edges = 0
    total_recurring_edges = 0

    if not hasattr(model, "convs"):
        return {"per_layer": [], "summary": {}}

    for idx, conv in enumerate(model.convs):
        layer_entry = {
            "layer_idx": idx,
            "conv_type": type(conv).__name__,
        }

        # --- MP / aggregation stats ---
        mp = getattr(conv, "mp_operation_stats", None)
        if mp is not None:
            num_forwards = max(int(mp.get("num_forwards", 1)), 1)

            naive_aggs_total = float(
                mp.get(
                    "naive_aggregations",
                    mp.get("naive_ops", 0) / 2.0,
                )
            )
            actual_aggs_total = float(
                mp.get(
                    "actual_aggregations",
                    mp.get("actual_ops", 0) / 2.0,
                )
            )

            naive_aggs = naive_aggs_total / num_forwards
            actual_aggs = actual_aggs_total / num_forwards
            saved_aggs = naive_aggs - actual_aggs

            saved_ratio = float(mp.get("saved_ratio", 0.0))

            layer_entry.update(
                {
                    "naive_aggregations": naive_aggs,
                    "actual_aggregations": actual_aggs,
                    "saved_aggregations": saved_aggs,
                    "saved_ratio": saved_ratio,
                }
            )

            total_naive_aggs += naive_aggs
            total_actual_aggs += actual_aggs
            total_saved_aggs += saved_aggs

        # --- Recurring transaction stats + cache hit ratio ---
        rec = getattr(conv, "recurring_transaction_stats", None)
        if rec is not None:
            num_edges = int(rec.get("num_edges", 0))
            num_unique = int(rec.get("num_unique_edges", 0))
            num_rec = int(rec.get("num_recurring_edges", 0))
            rec_ratio = float(rec.get("recurring_ratio", 0.0))
            cache_hits = int(rec.get("cache_hits", num_rec))
            cache_hit_ratio = float(rec.get("cache_hit_ratio", rec_ratio))

            layer_entry.update(
                {
                    "rec_num_edges": num_edges,
                    "rec_num_unique_edges": num_unique,
                    "rec_num_recurring_edges": num_rec,
                    "rec_recurring_ratio": rec_ratio,
                    "cache_hits": cache_hits,
                    "cache_hit_ratio": cache_hit_ratio,
                }
            )

            total_edges += num_edges
            total_recurring_edges += num_rec

        # --- LCS masking stats ---
        lcs = getattr(conv, "lcs_masking_stats", None)
        if lcs is not None:
            for k, v in lcs.items():
                layer_entry[f"lcs_{k}"] = v

        per_layer.append(layer_entry)

    if total_naive_aggs > 0:
        total_saved_ratio = float(total_saved_aggs) / float(total_naive_aggs)
    else:
        total_saved_ratio = 0.0

    if total_edges > 0:
        global_cache_hit_ratio = float(total_recurring_edges) / float(total_edges)
    else:
        global_cache_hit_ratio = 0.0

    summary = {
        "total_naive_aggregations": total_naive_aggs,
        "total_actual_aggregations": total_actual_aggs,
        "total_saved_aggregations": total_saved_aggs,
        "total_saved_ratio": total_saved_ratio,
        "total_edges": int(total_edges),
        "total_recurring_edges": int(total_recurring_edges),
        "global_cache_hit_ratio": global_cache_hit_ratio,
    }

    return {"per_layer": per_layer, "summary": summary}


def problem3_metrics_to_dataframe(metrics_dict):
    """
    Convert collect_problem3_metrics(...) output into a pandas DataFrame
    (per-layer view) for easy CSV export and Chapter 4 tables.
    """
    per_layer = metrics_dict.get("per_layer", [])
    if not per_layer:
        return pd.DataFrame()
    return pd.DataFrame(per_layer)

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


# ===========================================================================
# Convolution Layer Selector
# ===========================================================================

def get_conv(
        conv_type: str,
        input_dim: int,
        output_dim: int,
        alpha,
        lcs_threshold: float = 0.0,
        enable_lcs_masking: bool = False,
):
    """
    Factory method to select the appropriate convolution layer.

    - "dir-gcn": Returns the Baseline DirGCNConv.
    - "dir-gcn-gated": Returns the Enhanced GatedDirGCNConv which includes:
         1. Mini-Batch Sampling (3.4.1)
         2. Gated Feature Fusion (3.4.2)
         3. LCS Masking & Edge Caching (3.4.3)
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


# ===========================================================================
# Baseline Model: Dir-GCN Convolution
# ===========================================================================

class DirGCNConv(torch.nn.Module):
    """
    Baseline DIR-GCN layer (original directed GCN operator).

    Uses a global alpha to linearly mix:
      - src->dst (forward/inbound) messages
      - dst->src (reverse/outbound) messages
    """

    def __init__(self, input_dim, output_dim, alpha):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        self.lin_src_to_dst = Linear(input_dim, output_dim)
        self.lin_dst_to_src = Linear(input_dim, output_dim)
        self.alpha = alpha

        # Caches for normalized adjacency matrices
        self.adj_norm = None
        self.adj_t_norm = None

        # Statistics to track message passing operations (for comparison with Enhanced model)
        self.mp_operation_stats = {
            "naive_ops": 0,  # Operations a naive implementation performs
            "actual_ops": 0,  # Actual operations (same as naive for baseline)
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

        # Standard Dir-GCN propagation
        h_forward = self.lin_src_to_dst(self.adj_norm @ x)
        h_reverse = self.lin_dst_to_src(self.adj_t_norm @ x)

        # Fixed alpha fusion (Problem 2 limitation: fixed weighting)
        out = self.alpha * h_forward + (1.0 - self.alpha) * h_reverse

        # ---- Statistics Logging ----
        ops_this_layer = 2 * num_edges
        aggs_this_layer = num_edges  # 1 aggregation per edge per direction

        self.mp_operation_stats["naive_ops"] += ops_this_layer
        self.mp_operation_stats["actual_ops"] += ops_this_layer
        self.mp_operation_stats["naive_aggregations"] += aggs_this_layer
        self.mp_operation_stats["actual_aggregations"] += aggs_this_layer
        self.mp_operation_stats["num_forwards"] += 1

        return out


# ===========================================================================
# Enhanced Model: Gated Dir-GCN
# Implements Sections 3.4.1, 3.4.2, and 3.4.3
# ===========================================================================

class GatedDirGCNConv(torch.nn.Module):
    """
    Enhanced DIR-GCN layer.

    Key Innovations:
    1. Mini-Batch Sampling (Sec 3.4.1): Prioritizes neighbors via recurrence/value scores.
    2. Gated Feature Fusion (Sec 3.4.2): Learnable gate for inbound vs outbound flows.
    3. LCS Masking & Edge Caching (Sec 3.4.3): Reduces redundancy by grouping identical edges.
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

        # Direction-specific projections
        self.lin_src_to_dst = Linear(input_dim, output_dim)
        self.lin_dst_to_src = Linear(input_dim, output_dim)

        # -------------------------------------------------------------------
        # Components for Section 3.4.3: Local Contribution Score (LCS)
        # -------------------------------------------------------------------
        # Network to compute LCS score from node features
        self.lcs_mlp = nn.Sequential(
            Linear(2 * input_dim, input_dim),
            nn.ReLU(),
            Linear(input_dim, 1),
        )

        # -------------------------------------------------------------------
        # Components for Section 3.4.2: Gated Feature Fusion
        # -------------------------------------------------------------------
        # Node-wise gate to fuse inbound and outbound aggregations
        # gate = sigmoid(gate_mlp([h_in || h_out]))
        self.gate_mlp = nn.Sequential(
            Linear(2 * output_dim, output_dim),
            nn.ReLU(),
            Linear(output_dim, 1),
        )

        # Residual connection
        self.residual = (
            Linear(input_dim, output_dim) if input_dim != output_dim else nn.Identity()
        )

        # Configuration for LCS Masking
        self.lcs_threshold = float(lcs_threshold)
        self.enable_lcs_masking = bool(enable_lcs_masking)

        # -------------------------------------------------------------------
        # Structural Cache (Section 3.4.3 Edge Multiplicity Caching)
        # -------------------------------------------------------------------
        # Caches unique edge sets to avoid re-aggregating recurring transactions
        self._cached_src = None
        self._cached_dst = None
        self._cached_in_deg = None
        self._cached_out_deg = None

        self._unique_src = None
        self._unique_dst = None
        self._unique_counts = None
        self._num_edges_total = None
        self._num_unique_edges = None

        # Cached LCS values (computed once per unique edge)
        self._lcs_per_edge = None

        # Diagnostics stats
        self.last_lcs_mask = None
        self.lcs_masking_stats = None
        self.recurring_transaction_stats = None
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

        # Alpha kept for API compatibility, though Gated Fusion replaces it
        self.alpha = alpha

    # -----------------------------------------------------------------------
    # Implementation of Section 3.4.3: Edge Multiplicity Caching Logic
    # -----------------------------------------------------------------------
    def _init_structure_cache(self, edge_index: torch.Tensor, num_nodes: int):
        """
        One-time structural preprocessing.
        Groups identical edges (recurring transactions) into unique (src, dst) pairs.
        """
        src, dst = edge_index
        self._cached_src = src
        self._cached_dst = dst

        pairs = torch.stack([src, dst], dim=1)  # [E, 2]
        if pairs.numel() == 0:
            # Handle empty graph edge case
            self._unique_src = src
            self._unique_dst = dst
            self._unique_counts = torch.zeros_like(src)
            self._num_edges_total = 0
            self._num_unique_edges = 0
            self._cached_in_deg = torch.ones(num_nodes)
            self._cached_out_deg = torch.ones(num_nodes)
            return

        # Identify unique edges and count multiplicity
        unique_pairs, inverse_indices, counts = pairs.unique(
            dim=0, return_inverse=True, return_counts=True
        )

        self._unique_src = unique_pairs[:, 0]
        self._unique_dst = unique_pairs[:, 1]
        self._unique_counts = counts
        self._num_edges_total = int(pairs.size(0))
        self._num_unique_edges = int(unique_pairs.size(0))

        # Precompute degrees considering edge multiplicity
        counts_f = counts.to(torch.float32)
        in_deg = scatter_add(
            counts_f, self._unique_dst, dim=0, dim_size=num_nodes
        ).clamp(min=1.0)
        out_deg = scatter_add(
            counts_f, self._unique_src, dim=0, dim_size=num_nodes
        ).clamp(min=1.0)
        self._cached_in_deg = in_deg
        self._cached_out_deg = out_deg

        # Store cache hit statistics
        num_recurring_edges = int(self._num_edges_total - self._num_unique_edges)
        self.recurring_transaction_stats = {
            "num_edges": self._num_edges_total,
            "num_unique_edges": self._num_unique_edges,
            "num_recurring_edges": num_recurring_edges,
            "recurring_ratio": (
                        float(num_recurring_edges) / self._num_edges_total) if self._num_edges_total > 0 else 0.0,
            "cache_hits": num_recurring_edges,
            "cache_hit_ratio": (
                        float(num_recurring_edges) / self._num_edges_total) if self._num_edges_total > 0 else 0.0,
        }

    # -----------------------------------------------------------------------
    # Implementation of Section 3.4.1: Mini-Batch Sampling
    # -----------------------------------------------------------------------
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
        Implements criteria-based mini-batch sampling.

        Criteria 1: Recurrence Score (RS) - Frequency of node appearance in transaction paths.
        Criteria 2: Neighbor Value Score (NVS) - Derived from LCS ranking.
        """
        if self._unique_src is None:
            raise RuntimeError("Structure cache not initialized. Run forward() once first.")

        device = x.device
        batch_nodes = batch_nodes.to(device)

        src_u = self._unique_src.to(device)
        dst_u = self._unique_dst.to(device)
        counts = self._unique_counts.to(device).float()

        # 1. Identify potential neighbors (incident to batch nodes)
        incident_mask = (src_u.unsqueeze(0) == batch_nodes.unsqueeze(1)) | (
                dst_u.unsqueeze(0) == batch_nodes.unsqueeze(1)
        )
        incident_mask = incident_mask.any(dim=0)

        if incident_mask.sum() == 0:
            return torch.empty(0, dtype=torch.long, device=device)

        edge_idx_incident = torch.nonzero(incident_mask, as_tuple=False).view(-1)

        # 2. Compute Recurrence Score (RS)
        counts_incident = counts[edge_idx_incident]
        if counts_incident.numel() > 0:
            rs = counts_incident / counts_incident.max()
        else:
            rs = torch.zeros_like(counts_incident)

        # 3. Compute Neighbor Value Score (NVS) using Cached LCS
        if self._lcs_per_edge is None:
            # Fallback if LCS not yet computed
            x_src_full = x[src_u]
            x_dst_full = x[dst_u]
            lcs_input_full = torch.cat([x_src_full, x_dst_full], dim=-1)
            self._lcs_per_edge = (
                torch.sigmoid(self.lcs_mlp(lcs_input_full)).squeeze(-1).detach()
            )

        nvs = self._lcs_per_edge.to(device)[edge_idx_incident]

        # 4. Combine Scores
        total_w = recurrence_weight + neighbor_value_weight or 1.0
        combined_score = (recurrence_weight / total_w) * rs + (neighbor_value_weight / total_w) * nvs

        # 5. Select Top-K neighbors
        k = min(num_neighbors * max(1, batch_nodes.numel()), edge_idx_incident.numel())
        _, topk_idx = torch.topk(combined_score, k=k, largest=True, sorted=False)

        return edge_idx_incident[topk_idx]

    def forward(self, x, edge_index, batch_nodes: torch.Tensor = None):
        """
        Enhanced Forward Pass combining all three methods.
        """
        num_nodes = x.size(0)

        # --- [3.4.3] Initialize Edge Multiplicity Cache (First run only) ---
        if self._unique_src is None or self._unique_dst is None:
            self._init_structure_cache(edge_index, num_nodes)

        src_u = self._unique_src
        dst_u = self._unique_dst
        counts = self._unique_counts.float()

        # --- [3.4.1] Apply Mini-Batch Sampling (If batch_nodes provided) ---
        if batch_nodes is not None:
            sampled_idx = self.mini_batch_recurrence_neighbor_value_sampling(
                x, batch_nodes
            )
            if sampled_idx.numel() == 0:
                sampled_idx = torch.arange(src_u.size(0), device=src_u.device, dtype=torch.long)
        else:
            # Use full graph if no batching (Standard Dir-GCN behavior)
            sampled_idx = torch.arange(src_u.size(0), device=src_u.device, dtype=torch.long)

        src_sampled = src_u[sampled_idx]
        dst_sampled = dst_u[sampled_idx]
        counts_sampled = counts[sampled_idx]

        # --- [3.4.3] Compute Local Contribution Score (LCS) ---
        # Computed ONCE per unique edge and cached (self._lcs_per_edge)
        if self._lcs_per_edge is None:
            with torch.no_grad():
                x_src_full = x[src_u]
                x_dst_full = x[dst_u]
                lcs_input_full = torch.cat([x_src_full, x_dst_full], dim=-1)
                self._lcs_per_edge = (
                    torch.sigmoid(self.lcs_mlp(lcs_input_full)).squeeze(-1).detach()
                )  # [E_unique]

        # Retrieve cached LCS for currently sampled edges
        lcs = self._lcs_per_edge.to(x.device)[sampled_idx]

        # --- [3.4.3] LCS Masking Logic ---
        num_edges_total_effective = int(counts_sampled.sum().item())

        if self.enable_lcs_masking and self.lcs_threshold > 0.0:
            mask = lcs >= self.lcs_threshold  # Filter low-contribution edges

            # Diagnostics tracking
            edges_kept = int(counts_sampled[mask].sum().item())
            edges_pruned = int(num_edges_total_effective - edges_kept)

            if edges_kept == 0 and lcs.numel() > 0:
                # Safety fallback: Keep all if pruning removes everything
                mask = torch.ones_like(lcs, dtype=torch.bool)
                edges_kept = num_edges_total_effective
                edges_pruned = 0

            self.lcs_masking_stats = {
                "num_edges_total": num_edges_total_effective,
                "num_edges_kept": edges_kept,
                "num_edges_pruned": edges_pruned,
                "pruned_ratio": float(
                    edges_pruned) / num_edges_total_effective if num_edges_total_effective > 0 else 0.0,
                "threshold": float(self.lcs_threshold),
            }

            # Apply mask
            src = src_sampled[mask]
            dst = dst_sampled[mask]
            lcs_kept = lcs[mask]
            counts_kept = counts_sampled[mask]
        else:
            # No masking applied
            self.lcs_masking_stats = None
            src = src_sampled
            dst = dst_sampled
            lcs_kept = lcs
            counts_kept = counts_sampled

        # --- [3.4.2] Direction-Specific Aggregation ---
        # 1. Separate Inbound vs Outbound projections
        x_src_eff = x[src]
        x_dst_eff = x[dst]

        msg_in = self.lin_src_to_dst(x_src_eff)  # src → dst
        msg_out = self.lin_dst_to_src(x_dst_eff)  # dst → src

        # 2. Weight by LCS and Recurrence Count (Edge Multiplicity)
        lcs_kept = lcs_kept.unsqueeze(-1)
        counts_kept = counts_kept.unsqueeze(-1)

        msg_in = msg_in * lcs_kept * counts_kept
        msg_out = msg_out * lcs_kept * counts_kept

        # 3. Aggregate (Scatter Add)
        h_in = scatter_add(msg_in, dst, dim=0, dim_size=num_nodes)
        h_out = scatter_add(msg_out, src, dim=0, dim_size=num_nodes)

        # 4. Normalize by Degree
        # Use dynamic degrees if we are batching/masking, else cached degrees
        if (batch_nodes is not None) or (self.enable_lcs_masking and self.lcs_threshold > 0.0):
            counts_float = counts_kept.squeeze(-1)
            in_deg = scatter_add(counts_float, dst, dim=0, dim_size=num_nodes).clamp(min=1.0)
            out_deg = scatter_add(counts_float, src, dim=0, dim_size=num_nodes).clamp(min=1.0)
        else:
            in_deg = self._cached_in_deg
            out_deg = self._cached_out_deg

        h_in = h_in / in_deg.unsqueeze(-1)
        h_out = h_out / out_deg.unsqueeze(-1)

        # --- [3.4.2] Gated Feature Fusion ---
        # Learnable gate determines importance of inbound vs outbound for each node
        gate_input = torch.cat([h_in, h_out], dim=-1)
        gate = torch.sigmoid(self.gate_mlp(gate_input))  # [N, 1]

        fused = gate * h_in + (1.0 - gate) * h_out  # Adaptive fusion

        # Residual connection
        out = fused + self.residual(x)

        # --- Update Diagnostics (MP Operation Counter) ---
        num_unique_effective = int(src.numel())
        naive_ops_this = 2 * num_edges_total_effective
        actual_ops_this = 2 * num_unique_effective

        self.mp_operation_stats["naive_ops"] += naive_ops_this
        self.mp_operation_stats["actual_ops"] += actual_ops_this
        self.mp_operation_stats["naive_aggregations"] += num_edges_total_effective
        self.mp_operation_stats["actual_aggregations"] += num_unique_effective
        self.mp_operation_stats["saved_ops"] = self.mp_operation_stats["naive_ops"] - self.mp_operation_stats[
            "actual_ops"]
        self.mp_operation_stats["saved_aggregations"] = self.mp_operation_stats["naive_aggregations"] - \
                                                        self.mp_operation_stats["actual_aggregations"]

        if self.mp_operation_stats["naive_aggregations"] > 0:
            self.mp_operation_stats["saved_ratio"] = float(self.mp_operation_stats["saved_aggregations"]) / float(
                self.mp_operation_stats["naive_aggregations"])

        self.mp_operation_stats["num_forwards"] += 1

        return out


# ===========================================================================
# Multi-Layer GNN Wrapper
# ===========================================================================

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

        # Construct Layers
        self.convs = ModuleList()

        # Helper to simplify conv creation
        def create_layer(in_d, out_d):
            return get_conv(
                conv_type,
                in_d,
                out_d,
                self.alpha,
                lcs_threshold=self.lcs_threshold,
                enable_lcs_masking=self.enable_lcs_masking,
            )

        if num_layers == 1:
            self.convs.append(create_layer(num_features, output_dim))
        else:
            # Input Layer
            self.convs.append(create_layer(num_features, hidden_dim))
            # Hidden Layers
            for _ in range(num_layers - 2):
                self.convs.append(create_layer(hidden_dim, hidden_dim))
            # Output Layer
            self.convs.append(create_layer(hidden_dim, output_dim))

        # Jumping Knowledge (Optional)
        if jumping_knowledge:
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
            # Pass batch_nodes only if the layer supports it (GatedDirGCNConv)
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


# ===========================================================================
# Lightning Wrappers (Training Logic)
# ===========================================================================

class LightingFullBatchModelWrapper(pl.LightningModule):
    """
    Standard Full-Batch training wrapper.
    Used for Baseline Dir-GCN and optionally for Enhanced if batching is disabled.
    """

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

        # Setup Metrics
        num_classes = int(getattr(self.model, "num_classes", 2))
        metric_args = {"num_classes": num_classes, "average": "macro"}

        self.train_f1 = MulticlassF1Score(**metric_args)
        self.train_prec = MulticlassPrecision(**metric_args)
        self.train_rec = MulticlassRecall(**metric_args)

        self.val_f1 = MulticlassF1Score(**metric_args)
        self.val_prec = MulticlassPrecision(**metric_args)
        self.val_rec = MulticlassRecall(**metric_args)

        self.test_f1 = MulticlassF1Score(**metric_args)
        self.test_prec = MulticlassPrecision(**metric_args)
        self.test_rec = MulticlassRecall(**metric_args)

    def forward(self, x, edge_index):
        return self.model(x, edge_index, batch_nodes=None)

    def training_step(self, batch, batch_idx):
        x, y, edge_index = batch.x, batch.y.long(), batch.edge_index
        out = self.model(x, edge_index, batch_nodes=None)

        loss = F.nll_loss(out[self.train_mask], y[self.train_mask].squeeze())
        self.log("train_loss", loss)

        y_pred = out.argmax(dim=1)
        self.log("train_f1", self.train_f1(y_pred[self.train_mask], y[self.train_mask].squeeze()), prog_bar=True)
        self.log("train_prec", self.train_prec(y_pred[self.train_mask], y[self.train_mask].squeeze()))
        self.log("train_rec", self.train_rec(y_pred[self.train_mask], y[self.train_mask].squeeze()))

        train_acc = self.evaluate(y_pred[self.train_mask], y[self.train_mask])
        self.log("train_acc", train_acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        # ... (validation logic mirroring training step) ...
        # Simplified for brevity as logic is identical to training metrics
        pass  # Actual implementation handled by standard PL hooks if needed,
        # but here we usually run metrics calculation similar to training_step
        # but using self.val_mask.
        # (See full implementation in `training_step` block style below for consistency)

        x, y, edge_index = batch.x, batch.y.long(), batch.edge_index
        out = self.model(x, edge_index, batch_nodes=None)
        y_pred = out.argmax(dim=1)

        self.log("val_f1", self.val_f1(y_pred[self.val_mask], y[self.val_mask].squeeze()), prog_bar=True)
        self.log("val_prec", self.val_prec(y_pred[self.val_mask], y[self.val_mask].squeeze()))
        self.log("val_rec", self.val_rec(y_pred[self.val_mask], y[self.val_mask].squeeze()))
        self.log("val_acc", self.evaluate(y_pred[self.val_mask], y[self.val_mask]), prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, y, edge_index = batch.x, batch.y.long(), batch.edge_index
        out = self.model(x, edge_index, batch_nodes=None)
        y_pred = out.argmax(dim=1)

        self.log("test_f1", self.test_f1(y_pred[self.test_mask], y[self.test_mask].squeeze()))
        self.log("test_prec", self.test_prec(y_pred[self.test_mask], y[self.test_mask].squeeze()))
        self.log("test_rec", self.test_rec(y_pred[self.test_mask], y[self.test_mask].squeeze()))
        self.log("test_acc", self.evaluate(y_pred[self.test_mask], y[self.test_mask]))

    def evaluate(self, y_pred, y_true):
        if self.evaluator:
            acc = self.evaluator.eval({"y_true": y_true, "y_pred": y_pred.unsqueeze(1)})["acc"]
        else:
            acc = y_pred.eq(y_true.squeeze()).sum().item() / y_pred.shape[0]
        return acc

    def configure_optimizers(self):
        return optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)


class LightingMiniBatchDirGatedWrapper(pl.LightningModule):
    """
    Wrapper for Enhanced Dir-GCN using Mini-Batch Sampling (Sec 3.4.1).
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

        # Setup Metrics
        num_classes = int(getattr(self.model, "num_classes", 2))
        metric_args = {"num_classes": num_classes, "average": "macro"}

        self.train_f1 = MulticlassF1Score(**metric_args)
        self.train_prec = MulticlassPrecision(**metric_args)
        self.train_rec = MulticlassRecall(**metric_args)

        self.val_f1 = MulticlassF1Score(**metric_args)
        self.val_prec = MulticlassPrecision(**metric_args)
        self.val_rec = MulticlassRecall(**metric_args)

        self.test_f1 = MulticlassF1Score(**metric_args)
        self.test_prec = MulticlassPrecision(**metric_args)
        self.test_rec = MulticlassRecall(**metric_args)

    def forward(self, x, edge_index, batch_nodes=None):
        return self.model(x, edge_index, batch_nodes=batch_nodes)

    def _ensure_indices(self, device):
        if not hasattr(self, "_train_idx") or self._train_idx.device != device:
            self._train_idx = torch.nonzero(self.train_mask.to(device), as_tuple=False).view(-1)
            self._val_idx = torch.nonzero(self.val_mask.to(device), as_tuple=False).view(-1)
            self._test_idx = torch.nonzero(self.test_mask.to(device), as_tuple=False).view(-1)

    def training_step(self, batch, batch_idx):
        x = batch.x.to(self.device)
        y = batch.y.long().view(-1).to(self.device)
        edge_index = batch.edge_index.to(self.device)

        self._ensure_indices(self.device)
        train_idx = self._train_idx

        # --- Randomly Sample Seed Nodes for Mini-Batch ---
        batch_size = min(self.mini_batch_size, train_idx.numel())
        perm = torch.randperm(train_idx.numel(), device=self.device)[:batch_size]
        batch_nodes = train_idx[perm]

        # Forward pass with batch_nodes triggers Section 3.4.1 logic in the model
        out = self.model(x, edge_index, batch_nodes=batch_nodes)

        loss = F.nll_loss(out[batch_nodes], y[batch_nodes])
        self.log("train_loss", loss)

        y_pred = out.argmax(dim=1)

        # Metrics computed only on the sampled batch
        self.log("train_f1", self.train_f1(y_pred[batch_nodes], y[batch_nodes]), prog_bar=True)
        self.log("train_prec", self.train_prec(y_pred[batch_nodes], y[batch_nodes]))
        self.log("train_rec", self.train_rec(y_pred[batch_nodes], y[batch_nodes]))
        self.log("train_acc", self.evaluate(y_pred[batch_nodes], y[batch_nodes]), prog_bar=True)

        return loss

    # Validation and Test steps use full-graph evaluation (batch_nodes=None)
    def validation_step(self, batch, batch_idx):
        self._ensure_indices(self.device)
        x, y, edge_index = batch.x, batch.y.long(), batch.edge_index
        out = self.model(x, edge_index, batch_nodes=None)
        y_pred = out.argmax(dim=1)

        self.log("val_f1", self.val_f1(y_pred[self._val_idx], y[self._val_idx]), prog_bar=True)
        self.log("val_prec", self.val_prec(y_pred[self._val_idx], y[self._val_idx]))
        self.log("val_rec", self.val_rec(y_pred[self._val_idx], y[self._val_idx]))
        self.log("val_acc", self.evaluate(y_pred[self._val_idx], y[self._val_idx]), prog_bar=True)

    def test_step(self, batch, batch_idx):
        self._ensure_indices(self.device)
        x, y, edge_index = batch.x, batch.y.long(), batch.edge_index
        out = self.model(x, edge_index, batch_nodes=None)
        y_pred = out.argmax(dim=1)

        self.log("test_f1", self.test_f1(y_pred[self._test_idx], y[self._test_idx]))
        self.log("test_prec", self.test_prec(y_pred[self._test_idx], y[self._test_idx]))
        self.log("test_rec", self.test_rec(y_pred[self._test_idx], y[self._test_idx]))
        self.log("test_acc", self.evaluate(y_pred[self._test_idx], y[self._test_idx]))

    def evaluate(self, y_pred, y_true):
        if self.evaluator:
            acc = self.evaluator.eval({"y_true": y_true, "y_pred": y_pred.unsqueeze(1)})["acc"]
        else:
            acc = y_pred.eq(y_true.squeeze()).sum().item() / y_pred.shape[0]
        return acc

    def configure_optimizers(self):
        return optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)


# ===========================================================================
# Helpers
# ===========================================================================

def get_model(args):
    """
    Factory to create GNN instance from args.
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


def export_node_predictions(model, data, output_path: str, class_names=None):
    """
    Exports node-level predictions to CSV.
    """
    model.eval()
    device = next(model.parameters()).device
    x = data.x.to(device)
    edge_index = data.edge_index.to(device)

    with torch.no_grad():
        log_probs = model(x, edge_index, batch_nodes=None)
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
    Measures average inference time.
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
    Collects diagnostics related to Section 3.4.3 (Redundancy & Masking).
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

        # MP Statistics
        mp = getattr(conv, "mp_operation_stats", None)
        if mp is not None:
            num_forwards = max(int(mp.get("num_forwards", 1)), 1)
            naive_aggs = float(mp.get("naive_aggregations", 0)) / num_forwards
            actual_aggs = float(mp.get("actual_aggregations", 0)) / num_forwards
            saved_aggs = naive_aggs - actual_aggs

            layer_entry.update({
                "naive_aggregations": naive_aggs,
                "actual_aggregations": actual_aggs,
                "saved_aggregations": saved_aggs,
                "saved_ratio": float(mp.get("saved_ratio", 0.0)),
            })
            total_naive_aggs += naive_aggs
            total_actual_aggs += actual_aggs
            total_saved_aggs += saved_aggs

        # Recurring Edge Statistics
        rec = getattr(conv, "recurring_transaction_stats", None)
        if rec is not None:
            num_rec = int(rec.get("num_recurring_edges", 0))
            num_edges = int(rec.get("num_edges", 0))
            layer_entry.update({
                "rec_num_edges": num_edges,
                "rec_num_unique_edges": int(rec.get("num_unique_edges", 0)),
                "rec_num_recurring_edges": num_rec,
                "rec_recurring_ratio": float(rec.get("recurring_ratio", 0.0)),
                "cache_hits": num_rec,
                "cache_hit_ratio": float(rec.get("cache_hit_ratio", 0.0)),
            })
            total_edges += num_edges
            total_recurring_edges += num_rec

        # LCS Masking Statistics
        lcs = getattr(conv, "lcs_masking_stats", None)
        if lcs is not None:
            for k, v in lcs.items():
                layer_entry[f"lcs_{k}"] = v

        per_layer.append(layer_entry)

    total_saved_ratio = (total_saved_aggs / total_naive_aggs) if total_naive_aggs > 0 else 0.0
    global_cache_hit_ratio = (total_recurring_edges / total_edges) if total_edges > 0 else 0.0

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
    Helper to convert metrics dictionary to DataFrame.
    """
    per_layer = metrics_dict.get("per_layer", [])
    if not per_layer:
        return pd.DataFrame()
    return pd.DataFrame(per_layer)

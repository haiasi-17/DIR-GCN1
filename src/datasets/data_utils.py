import math
import torch
from torch_sparse import SparseTensor
from torch_geometric.utils import degree


# ---------------------------------------------------------
# 1. Helper: boolean mask from indices  (used in data_loading.py)
# ---------------------------------------------------------
def get_mask(idx: torch.Tensor, size: int) -> torch.Tensor:
    """
    Create a boolean mask of length `size`, where positions in `idx`
    are set to True and the rest are False.

    Parameters
    ----------
    idx : LongTensor
        Indices to be marked as True.
    size : int
        Total number of nodes.

    Returns
    -------
    mask : BoolTensor of shape [size]
    """
    mask = torch.zeros(size, dtype=torch.bool)
    if idx.numel() > 0:
        mask[idx] = True
    return mask


# ---------------------------------------------------------
# 2. Normalized adjacency for Dir-GCN / GNN layers
#    (used in src/model.py)
# ---------------------------------------------------------
def get_norm_adj(adj: SparseTensor, norm: str = "dir") -> SparseTensor:
    """
    Compute a normalized adjacency matrix from a SparseTensor adjacency.

    The original Dir-GCN code calls:
        get_norm_adj(adj, norm="dir")

    Here we implement a simple, standard normalization:

        - If norm == "dir": row-normalized adjacency  A_norm = D^{-1} A
        - If norm == "sym": symmetric normalization   A_norm = D^{-1/2} A D^{-1/2}

    This returns a SparseTensor that can be used as:
        A_norm @ x

    which is exactly what the current DirGCNConv / GatedDirGCNConv expect.
    """
    # obtain COO representation (row, col, value)
    row, col, value = adj.coo()
    if value is None:
        value = torch.ones_like(row, dtype=torch.float32)

    # out-degree (per row) based on unnormalized adjacency
    deg = degree(row, num_nodes=adj.sizes()[0], dtype=torch.float32)

    if norm == "sym":
        # symmetric normalization: D^{-1/2} A D^{-1/2}
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float("inf")] = 0.0

        # value_ij = A_ij * (deg_i^-1/2 * deg_j^-1/2)
        new_val = value * deg_inv_sqrt[row] * deg_inv_sqrt[col]

    else:
        # "dir" (default): row-normalized adjacency D^{-1} A
        deg_inv = deg.pow(-1.0)
        deg_inv[deg_inv == float("inf")] = 0.0

        # value_ij = A_ij * (deg_i^-1)
        new_val = value * deg_inv[row]

    return SparseTensor(
        row=row,
        col=col,
        value=new_val,
        sparse_sizes=adj.sizes(),
    )


# ---------------------------------------------------------
# 3. Edge homophily + uniform summary (used by prepare_*.py)
# ---------------------------------------------------------
def compute_edge_homophily(
    edge_index: torch.Tensor,
    y: torch.Tensor,
    ignore_label: int | None = -1,
) -> float:
    """
    Fraction of edges whose endpoints share the same (non-ignored) label.

    Parameters
    ----------
    edge_index : LongTensor, shape [2, E]
        Graph edges (src, dst).
    y : Tensor, shape [N]
        Node labels.
    ignore_label : int or None
        If not None, edges incident to nodes with this label are ignored.

    Returns
    -------
    h : float
        Edge homophily in [0,1].
    """
    src, dst = edge_index

    if ignore_label is not None:
        mask = (y[src] != ignore_label) & (y[dst] != ignore_label)
        src = src[mask]
        dst = dst[mask]

    if src.numel() == 0:
        return 0.0

    same = (y[src] == y[dst]).float()
    return same.mean().item()


def print_graph_summary(
    name: str,
    data,
    unknown_label: int | None = None,
    positive_label: int = 1,
) -> None:
    """
    Uniform summary printer for all fraud datasets.

    Prints:
      - #nodes, #edges, #features
      - label distribution (positive / negative)
      - fraud ratio
      - edge homophily (if computable)
    """
    y = data.y.view(-1)

    if unknown_label is not None:
        labeled_mask = y != unknown_label
        y_labeled = y[labeled_mask]
    else:
        labeled_mask = torch.ones_like(y, dtype=torch.bool)
        y_labeled = y

    num_nodes = int(data.num_nodes)
    num_edges = int(data.edge_index.size(1))
    num_node_feats = int(data.x.size(1))

    edge_attr = getattr(data, "edge_attr", None)
    num_edge_feats = 0 if edge_attr is None else int(edge_attr.size(1))

    num_labeled = int(labeled_mask.sum().item())
    num_pos = int((y_labeled == positive_label).sum().item())
    num_neg = int((y_labeled != positive_label).sum().item())

    fraud_ratio = num_pos / max(num_labeled, 1)

    try:
        h = compute_edge_homophily(data.edge_index, y, ignore_label=unknown_label)
    except Exception:
        h = None

    print(f"\n=== {name} Dataset Info ===")
    print(f"Num nodes        : {num_nodes}")
    print(f"Num edges        : {num_edges}")
    print(f"Num node feats   : {num_node_feats}")
    print(f"Num edge feats   : {num_edge_feats}")
    print(f"Labeled nodes    : {num_labeled}")
    print(f"  - positive ({positive_label}): {num_pos}")
    print(f"  - negative          : {num_neg}")
    print(f"Fraud ratio      : {fraud_ratio:.4f}")
    if h is not None:
        print(f"Edge homophily   : {h:.4f}")

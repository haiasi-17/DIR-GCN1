import os
import numpy as np
import math

import torch
import torch_geometric
from torch_geometric.data import download_url
from torch_geometric.datasets import (
    WikipediaNetwork,
    CitationFull,
)
import torch_geometric.transforms as transforms
from ogb.nodeproppred import PygNodePropPredDataset, Evaluator

from src.datasets.directed_heterophilous_graphs import DirectedHeterophilousGraphDataset
from src.datasets.telegram import Telegram
from src.datasets.data_utils import get_mask
from src.utils.third_party import (
    load_snap_patents_mat,
    even_quantile_labels,
)

from src.datasets.synthetic import get_syn_dataset
from src.datasets.synthetic_fraud import get_synthetic_fraud_dataset
from src.datasets.elliptic import EllipticDataset
from src.datasets.ulb_credit import get_ulb_credit_dataset
from src.datasets.online_payments import get_online_payments_dataset


def get_dataset(
    name: str,
    root_dir: str,
    homophily=None,
    undirected: bool = False,
    self_loops: bool = False,
    transpose: bool = False,
):
    path = f"{root_dir}/"
    evaluator = None

    # Wikipedia (heterophilous)
    if name in ["chameleon", "squirrel"]:
        dataset = WikipediaNetwork(
            root=path, name=name, transform=transforms.NormalizeFeatures()
        )
        # make y shape [N, 1]
        dataset._data.y = dataset._data.y.unsqueeze(-1)

    # OGBN-Arxiv
    elif name in ["ogbn-arxiv"]:
        dataset = PygNodePropPredDataset(
            name=name, transform=transforms.ToSparseTensor(), root=path
        )
        evaluator = Evaluator(name=name)
        split_idx = dataset.get_idx_split()
        dataset._data.train_mask = get_mask(split_idx["train"], dataset._data.num_nodes)
        dataset._data.val_mask = get_mask(split_idx["valid"], dataset._data.num_nodes)
        dataset._data.test_mask = get_mask(split_idx["test"], dataset._data.num_nodes)

    # Directed Roman Empire
    elif name in ["directed-roman-empire"]:
        dataset = DirectedHeterophilousGraphDataset(
            name=name, transform=transforms.NormalizeFeatures(), root=path
        )

    # SNAP patents
    elif name == "snap-patents":
        dataset = load_snap_patents_mat(n_classes=5, root=path)

    # Arxiv-year
    elif name == "arxiv-year":
        # arxiv-year uses the same graph and features as ogbn-arxiv, but with different labels
        dataset = PygNodePropPredDataset(
            name="ogbn-arxiv", transform=transforms.ToSparseTensor(), root=path
        )
        evaluator = Evaluator(name="ogbn-arxiv")
        y = even_quantile_labels(
            dataset._data.node_year.flatten().numpy(), nclasses=5, verbose=False
        )
        dataset._data.y = torch.as_tensor(y).reshape(-1, 1)
        # Dummy masks; real splits handled in get_dataset_split
        dataset._data.train_mask = 0
        dataset._data.val_mask = 0
        dataset._data.test_mask = 0
        os.makedirs(os.path.join(path, name.replace("-", "_"), "raw"), exist_ok=True)

    # Original synthetic directed graph
    elif name == "syn-dir":
        dataset = get_syn_dataset(path)

    # Cora / Citeseer full
    elif name in ["cora_ml", "citeseer_full"]:
        ds_name = "citeseer" if name == "citeseer_full" else name
        dataset = CitationFull(path, ds_name)

    # Telegram
    elif name == "telegram":
        dataset = Telegram(path)

    # Synthetic fraud (our custom dataset)
    elif name == "fraud-syn":
        dataset = get_synthetic_fraud_dataset(path)

    # Elliptic dataset (already provides splits)
    elif name == "elliptic":
        dataset = EllipticDataset(root=os.path.join(path, "elliptic"))
        evaluator = None

    # ULB credit card fraud (our tabular→graph conversion)
    elif name == "ulb-credit":
        dataset = get_ulb_credit_dataset(path)
        evaluator = None

    elif name == "online-payments":
        dataset = get_online_payments_dataset(root_dir)
        evaluator = None  # you use manual accuracy/F1 already

    else:
        raise Exception(f"Unknown dataset: {name}")

    # optional transforms
    if undirected:
        dataset._data.edge_index = torch_geometric.utils.to_undirected(
            dataset._data.edge_index
        )
    if self_loops:
        dataset._data.edge_index, _ = torch_geometric.utils.add_self_loops(
            dataset._data.edge_index
        )
    if transpose:
        ei = dataset._data.edge_index
        dataset._data.edge_index = torch.stack([ei[1], ei[0]])

    return dataset, evaluator


def get_dataset_split(name, data, root_dir, split_number):
    """
    Return train/val/test masks for a dataset.

    Some datasets have precomputed multi-splits in data["train_mask"][:, k],
    others have a single fixed split, and some we split uniformly (50/25/25).
    """

    # Datasets with multi-splits stored in mask matrices
    if name in [
        "snap-patents",
        "chameleon",
        "squirrel",
        "telegram",
        "directed-roman-empire",
    ]:
        return (
            data["train_mask"][:, split_number],
            data["val_mask"][:, split_number],
            data["test_mask"][:, split_number],
        )

    # OGBN-Arxiv: single split from OGB
    if name in ["ogbn-arxiv"]:
        return data["train_mask"], data["val_mask"], data["test_mask"]

    # Arxiv-year (external split file from CUAI repo)
    if name in ["arxiv-year"]:
        num_nodes = data["y"].shape[0]
        github_url = (
            "https://github.com/CUAI/Non-Homophily-Large-Scale/raw/master/data/splits/"
        )
        split_file_name = f"{name}-splits.npy"
        local_dir = os.path.join(root_dir, name.replace("-", "_"), "raw")

        download_url(os.path.join(github_url, split_file_name), local_dir, log=False)
        splits = np.load(os.path.join(local_dir, split_file_name), allow_pickle=True)
        split_idx = splits[split_number % len(splits)]

        train_mask = get_mask(split_idx["train"], num_nodes)
        val_mask = get_mask(split_idx["valid"], num_nodes)
        test_mask = get_mask(split_idx["test"], num_nodes)

        return train_mask, val_mask, test_mask

    # Uniform split datasets (we generate 50/25/25 splits from labels)
    if name in ["syn-dir", "cora_ml", "citeseer_full", "fraud-syn"]:
        return set_uniform_train_val_test_split(
            split_number, data, train_ratio=0.5, val_ratio=0.25
        )

    # ULB credit: masks are already in the Data object
    if name == "ulb-credit":
        return data.train_mask, data.val_mask, data.test_mask

    # Elliptic: also already has masks (single split)
    if name == "elliptic":
        return data.train_mask, data.val_mask, data.test_mask

    # Online Payments (PaySim) – create a random 60/20/20 split
    if name == "online-payments":
        return set_uniform_train_val_test_split(
            split_number,
            data,
            train_ratio=0.6,
            val_ratio=0.2,  # remaining 0.2 becomes test
        )

    raise RuntimeError(f"get_dataset_split: no split rule defined for dataset '{name}'")


def set_uniform_train_val_test_split(seed, data, train_ratio=0.5, val_ratio=0.25):
    """
    Create a random train/val/test split over labeled nodes (y != -1).

    Default: 50% train, 25% val, 25% test.
    """
    rnd_state = np.random.RandomState(seed)
    num_nodes = data.y.shape[0]

    # Only use nodes with valid labels
    labeled_nodes = torch.where(data.y != -1)[0]
    num_labeled_nodes = labeled_nodes.shape[0]

    num_train = math.floor(num_labeled_nodes * train_ratio)
    num_val = math.floor(num_labeled_nodes * val_ratio)

    idxs = list(range(num_labeled_nodes))
    rnd_state.shuffle(idxs)

    train_idx = idxs[:num_train]
    val_idx = idxs[num_train : num_train + num_val]
    test_idx = idxs[num_train + num_val :]

    train_idx = labeled_nodes[train_idx]
    val_idx = labeled_nodes[val_idx]
    test_idx = labeled_nodes[test_idx]

    train_mask = get_mask(train_idx, num_nodes)
    val_mask = get_mask(val_idx, num_nodes)
    test_mask = get_mask(test_idx, num_nodes)

    return train_mask, val_mask, test_mask

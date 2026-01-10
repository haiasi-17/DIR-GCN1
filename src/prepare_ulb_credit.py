import os

from src.datasets.ulb_credit import ULBCCFDataset
from src.datasets.ulb_credit import compute_edge_homophily  # reuse helper


def main():
    root_dir = "dataset"  # same root you use in args.dataset_directory
    dataset_root = os.path.join(root_dir, "ulb_credit")

    print("Processing ULB Credit Card Fraud Detection dataset...")
    dataset = ULBCCFDataset(root=dataset_root, window=5)
    data = dataset.data  # or dataset._data depending on your style

    num_nodes = data.num_nodes
    num_edges = data.edge_index.size(1)
    num_feats = data.x.size(1)

    y = data.y
    num_fraud = int((y == 1).sum().item())
    num_norm = int((y == 0).sum().item())
    total_labeled = num_fraud + num_norm
    fraud_ratio = num_fraud / max(total_labeled, 1)

    h = compute_edge_homophily(data.edge_index, y)

    print("\n=== ULB Credit Card Fraud Graph ===")
    print(f"Num nodes       : {num_nodes}")
    print(f"Num edges       : {num_edges}")
    print(f"Num node feats  : {num_feats}")
    print(f"Fraud (1)       : {num_fraud}")
    print(f"Normal (0)      : {num_norm}")
    print(f"Fraud ratio     : {fraud_ratio:.4f}")
    print(f"Edge homophily  : {h:.4f}")
    print("\nTrain / Val / Test sizes:")
    print(f"  train: {int(data.train_mask.sum().item())}")
    print(f"  val  : {int(data.val_mask.sum().item())}")
    print(f"  test : {int(data.test_mask.sum().item())}")


if __name__ == "__main__":
    main()

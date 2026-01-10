import os
from src.datasets.elliptic import EllipticDataset
from src.datasets.data_utils import print_graph_summary


def main():
    root_dir = "dataset"  # align with others
    elliptic_root = os.path.join(root_dir, "elliptic")

    dataset = EllipticDataset(root=elliptic_root)
    data = dataset._data

    print_graph_summary(
        name="Elliptic Bitcoin Transactions",
        data=data,
        unknown_label=-1,  # -1 = unlabeled in Elliptic
        positive_label=1,  # 1 = illicit
    )

    # Optional: also print labeled split sizes
    y = data.y
    print("\nTrain / Val / Test sizes (labeled only):")
    print(f"  train: {(data.train_mask & (y != -1)).sum().item()}")
    print(f"  val  : {(data.val_mask & (y != -1)).sum().item()}")
    print(f"  test : {(data.test_mask & (y != -1)).sum().item()}")


if __name__ == "__main__":
    main()

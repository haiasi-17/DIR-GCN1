from src.datasets.synthetic_fraud import get_synthetic_fraud_dataset
from src.datasets.data_utils import print_graph_summary


def main():
    root = "dataset"  # or use your args.dataset_directory
    dataset = get_synthetic_fraud_dataset(root)
    data = dataset._data

    print_graph_summary(
        name="Synthetic Fraud (fraud-syn)",
        data=data,
        unknown_label=None,  # all nodes are labeled
        positive_label=1,
    )


if __name__ == "__main__":
    main()

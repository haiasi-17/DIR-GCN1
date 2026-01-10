import os
from src.datasets.online_payments import get_online_payments_dataset
from src.datasets.data_utils import print_graph_summary


def main():
    dataset_root = os.path.join("dataset")  # same as other datasets
    dataset = get_online_payments_dataset(dataset_root, max_rows=300_000)
    data = dataset._data

    print_graph_summary(
        name="Online Payments Fraud",
        data=data,
        unknown_label=None,  # all accounts have 0/1 label
        positive_label=1,
    )


if __name__ == "__main__":
    main()

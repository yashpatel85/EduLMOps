from datasets import load_from_disk
from pathlib import Path
import pandas as pd


DATASET_PATH = Path(
    "data/raw/squad_v2"
).resolve()


def analyze_dataset():
    dataset = load_from_disk(
        str(DATASET_PATH)
    )

    train_df = pd.DataFrame(dataset["train"])

    print("\nDataset Overview")
    print("=" * 50)

    print(f"Rows: {len(train_df)}")
    print(f"Columns: {list(train_df.columns)}")

    print("\nMissing Values:")
    print(train_df.isnull().sum())

    print("\nQuestion Length Statistics")

    train_df["question_length"] = (
        train_df["question"]
        .astype(str)
        .apply(lambda x: len(x.split()))
    )

    print(train_df["question_length"].describe())

    print("\nContext Length Statistics")

    train_df["context_length"] = (
        train_df["context"]
        .astype(str)
        .apply(lambda x: len(x.split()))
    )

    print(train_df["context_length"].describe())


if __name__ == "__main__":
    analyze_dataset()
from datasets import load_from_disk
from pathlib import Path


RAW_PATH = Path("data/raw/squad_v2")

PROCESSED_PATH = Path("data/interim/squad_v2")


def create_splits():

    dataset = load_from_disk(
        str(RAW_PATH.resolve())
    )

    train_dataset = dataset["train"]

    train_val_test = train_dataset.train_test_split(
        test_size=0.2,
        seed=42
    )

    train_set = train_val_test["train"]

    temp_set = train_val_test["test"]

    val_test_split = temp_set.train_test_split(
        test_size=0.5,
        seed=42
    )

    val_set = val_test_split["train"]

    test_set = val_test_split["test"]

    split_dataset = {
        "train": train_set,
        "validation": val_set,
        "test": test_set
    }

    from datasets import DatasetDict

    final_dataset = DatasetDict(split_dataset)

    PROCESSED_PATH.mkdir(
        parents=True,
        exist_ok=True
    )

    final_dataset.save_to_disk(
        str(PROCESSED_PATH.resolve())
    )

    print("Dataset splits created.")

    print(
        f"Train: {len(train_set)}"
    )

    print(
        f"Validation: {len(val_set)}"
    )

    print(
        f"Test: {len(test_set)}"
    )


if __name__ == "__main__":
    create_splits()
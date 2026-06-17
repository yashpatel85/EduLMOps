from datasets import load_from_disk
from pathlib import Path


DATASET_PATH = Path("data/raw/squad_v2")


REQUIRED_COLUMNS = [
    "question",
    "context",
    "answers"
]


def validate_dataset():

    dataset = load_from_disk(str(DATASET_PATH.resolve()))

    train_split = dataset["train"]

    print("Running validation...")

    for column in REQUIRED_COLUMNS:

        if column not in train_split.column_names:
            raise ValueError(
                f"Missing column: {column}"
            )

    print("Required columns verified.")

    empty_questions = sum(
        1 for q in train_split["question"]
        if not str(q).strip()
    )

    print(
        f"Empty questions found: "
        f"{empty_questions}"
    )

    print("Validation complete.")


if __name__ == "__main__":
    validate_dataset()
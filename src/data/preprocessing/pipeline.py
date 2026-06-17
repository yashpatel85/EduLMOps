from pathlib import Path

from datasets import (
    load_from_disk,
    DatasetDict
)

from cleaner import clean_dataset
from formatter import format_example
from tokenizer import tokenize_example
from version_manager import (
    get_next_version
)


INPUT_PATH = Path(
    "data/interim/squad_v2"
)

OUTPUT_DIR = Path(
    "data/processed"
)


def process_split(dataset):

    dataset = clean_dataset(dataset)

    dataset = dataset.map(
        format_example
    )

    dataset = dataset.map(
        tokenize_example
    )

    return dataset


def main():

    dataset = load_from_disk(
        str(INPUT_PATH.resolve())
    )

    processed_dataset = DatasetDict({
        split:
            process_split(dataset[split])
        for split in dataset.keys()
    })

    version_name = get_next_version()

    save_path = (
        OUTPUT_DIR / version_name
    )

    processed_dataset.save_to_disk(
        str(save_path.resolve())
    )

    print(
        f"Dataset saved to {save_path}"
    )

    for split in processed_dataset:

        print(
            f"{split}: "
            f"{len(processed_dataset[split])}"
        )


if __name__ == "__main__":
    main()
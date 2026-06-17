from datasets import load_from_disk


DATASET_PATH = (
    "data/processed/educational_qa_v1"
)


def prepare_dataset():

    dataset = load_from_disk(
        DATASET_PATH
    )

    columns_to_keep = [
        "instruction",
        "input",
        "output"
    ]

    for split in dataset.keys():

        cols_to_remove = [
            c
            for c in dataset[split].column_names
            if c not in columns_to_keep
        ]

        dataset[split] = dataset[
            split
        ].remove_columns(
            cols_to_remove
        )

    return dataset
from datasets import load_dataset
from pathlib import Path

RAW_DATA_DIR = Path("data/raw")


def download_dataset(
    dataset_name: str = "squad_v2"
):
    print(f"Downloading {dataset_name}...")

    dataset = load_dataset(dataset_name)

    RAW_DATA_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    save_path = RAW_DATA_DIR / dataset_name

    dataset.save_to_disk(
        str(save_path)
    )

    print(
        f"Dataset saved to: {save_path}"
    )

    return dataset


if __name__ == "__main__":
    download_dataset()
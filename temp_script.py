from datasets import load_from_disk

dataset = load_from_disk(
    "data/processed/educational_qa_v1"
)

print(dataset["train"][0].keys())
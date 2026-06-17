from datasets import Dataset


def clean_dataset(dataset: Dataset):

    def is_valid(example):

        question = example["question"]
        context = example["context"]
        answers = example["answers"]["text"]

        return (
            example["question"] is not None
            and example["context"] is not None
            and len(str(example["question"]).strip()) > 0
            and len(str(example["context"]).strip()) > 0
        )

    cleaned_dataset = dataset.filter(is_valid)

    return cleaned_dataset
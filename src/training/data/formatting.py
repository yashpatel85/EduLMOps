def formatting_func(example):

    text = (
        f"### Instruction:\n"
        f"{example['instruction']}\n\n"
        f"### Input:\n"
        f"{example['input']}\n\n"
        f"### Response:\n"
        f"{example['output']}"
    )

    return text
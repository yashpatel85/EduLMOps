from transformers import AutoTokenizer


MODEL_NAME = "microsoft/Phi-3-mini-4k-instruct"

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)


def tokenize_example(example):

    prompt = (
        f"Instruction: {example['instruction']}\n\n"
        f"Input: {example['input']}\n\n"
        f"Response: {example['output']}"
    )

    tokens = tokenizer(
        prompt,
        truncation=True,
        max_length=512,
        padding=False
    )

    return tokens
def create_prompt(example):

    text = f"""
### Instruction:
{example['instruction']}

### Input:
{example['input']}

### Response:
{example['output']}
"""

    return {
        "text": text.strip()
    }
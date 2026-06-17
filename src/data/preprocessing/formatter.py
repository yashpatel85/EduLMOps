def format_example(example):

    answers = example["answers"]["text"]

    answer = (
        answers[0]
        if len(answers) > 0
        else "Answer not found in context."
    )

    return {
        "instruction":
            "Answer the educational question using the provided context.",

        "input":
            f"Question: {example['question']}\n\n"
            f"Context: {example['context']}",

        "output":
            answer
    }
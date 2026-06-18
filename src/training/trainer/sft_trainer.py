from transformers import TrainingArguments


def get_training_args():

    return TrainingArguments(
        output_dir="artifacts/models",

        num_train_epochs=1,

        per_device_train_batch_size=1,

        gradient_accumulation_steps=8,

        learning_rate=2e-4,

        logging_steps=10,

        save_strategy="epoch",

        fp16=False,
        bf16=False,

        optim="paged_adamw_32bit",

        report_to="wandb"
    )
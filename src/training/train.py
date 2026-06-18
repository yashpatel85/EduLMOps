from trl import SFTTrainer

from src.training.models.model_loader import (
    load_model
)

from src.training.data.prepare_dataset import (
    prepare_dataset
)

from src.training.data.formatting import (
    formatting_func
)

from src.training.configs.qlora_config import (
    get_lora_config
)

from src.training.trainer.sft_trainer import (
    get_training_args
)

from src.monitoring.wandb_manager import (
    initialize_wandb
)

from src.monitoring.experiment_logger import (
    finish_run
)


def main():

    initialize_wandb()

    dataset = prepare_dataset()

    train_dataset = (
        dataset["train"]
        .shuffle(seed=42)
        .select(range(500))
    )

    model, tokenizer = load_model()

    trainer = SFTTrainer(
        model=model,

        args=get_training_args(),

        train_dataset=train_dataset,

        processing_class=tokenizer,

        peft_config=get_lora_config(),

        formatting_func=formatting_func
    )

    trainer.train()

    trainer.save_model(
        "artifacts/models/tinyllama_lora"
    )

    finish_run()


if __name__ == "__main__":
    main()
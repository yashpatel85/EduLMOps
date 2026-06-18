import wandb


def initialize_wandb():

    wandb.init(
        project="EduLMOps",
        
        name="tinyllama-qlora-r16-lr2e4-v1",

        job_type="qlora-training",

        tags=[
            "TinyLlama",
            "QLoRA",
            "EducationalQA"
        ],

        config={

            "model":
                "TinyLlama-1.1B-Chat-v1.0",

            "dataset":
                "educational_qa_v1",

            "epochs":
                1,

            "batch_size":
                1,

            "learning_rate":
                2e-4,

            "lora_rank":
                16
        }
    )
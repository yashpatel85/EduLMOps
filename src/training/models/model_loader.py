import torch

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer
)

from src.training.configs.qlora_config import (
    get_bnb_config
)


MODEL_NAME = (
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
)


def load_model():

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,

        quantization_config=get_bnb_config(),

        device_map="auto",

        dtype=torch.float16
    )

    model.config.use_cache = False
    model.gradient_checkpointing_enable()

    return model, tokenizer
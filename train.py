import os
from dataclasses import dataclass

import torch
from datasets import load_dataset
from peft import LoraConfig, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import SFTTrainer


MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
DATASET_NAME = "b-mc2/sql-create-context"
OUTPUT_DIR = "adapter"
SPLIT_SEED = 42
TEST_SIZE = 0.01


@dataclass
class TrainConfig:
    model_name: str = MODEL_NAME
    dataset_name: str = DATASET_NAME
    output_dir: str = OUTPUT_DIR
    max_steps: int = 2000
    per_device_train_batch_size: int = 2
    learning_rate: float = 2e-4
    max_seq_length: int = 512
    gradient_checkpointing: bool = True
    split_seed: int = SPLIT_SEED
    test_size: float = TEST_SIZE


def _ensure_list(value):
    if isinstance(value, list):
        return value
    return [value]


def format_prompt(example: dict) -> list[str]:
    questions = _ensure_list(example["question"])
    contexts = _ensure_list(example["context"])
    answers = _ensure_list(example["answer"])

    prompts = []
    for question, context, answer in zip(questions, contexts, answers):
        prompts.append(
            f"Generate SQL:\nQuestion: {question.strip()}\nContext: {context.strip()}\nSQL: {answer.strip()}"
        )
    return prompts


def build_quantization_config() -> BitsAndBytesConfig:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for 4-bit QLoRA training.")

    compute_dtype = torch.float16
    if torch.cuda.is_bf16_supported():
        compute_dtype = torch.bfloat16

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
    )


def load_tokenizer(model_name: str) -> AutoTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def load_model(config: TrainConfig) -> AutoModelForCausalLM:
    quant_config = build_quantization_config()
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        quantization_config=quant_config,
        device_map="auto",
        trust_remote_code=False,
    )
    model.config.use_cache = False
    if config.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)
    return model


def build_peft_config() -> LoraConfig:
    return LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "v_proj"],
    )


def load_train_split(config: TrainConfig):
    dataset = load_dataset(config.dataset_name, split="train")
    split_dataset = dataset.train_test_split(test_size=config.test_size, seed=config.split_seed)
    return split_dataset["train"], split_dataset["test"]


def main() -> None:
    config = TrainConfig()
    os.makedirs(config.output_dir, exist_ok=True)

    print(f"Loading dataset: {config.dataset_name}")
    train_dataset, eval_dataset = load_train_split(config)
    print(f"Training samples: {len(train_dataset)}")
    print(f"Held-out eval samples: {len(eval_dataset)}")

    tokenizer = load_tokenizer(config.model_name)
    model = load_model(config)
    peft_config = build_peft_config()

    training_args = TrainingArguments(
        output_dir=config.output_dir,
        max_steps=config.max_steps,
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=4,
        learning_rate=config.learning_rate,
        logging_steps=10,
        save_strategy="epoch",
        optim="paged_adamw_8bit",
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        report_to="none",
        remove_unused_columns=True,
        gradient_checkpointing=config.gradient_checkpointing,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        max_grad_norm=0.3,
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        peft_config=peft_config,
        args=training_args,
        formatting_func=format_prompt,
        tokenizer=tokenizer,
        max_seq_length=config.max_seq_length,
    )

    print("Starting QLoRA fine-tuning...")
    trainer.train()
    trainer.model.save_pretrained(config.output_dir)
    tokenizer.save_pretrained(config.output_dir)
    print(f"Adapter saved to .\\{config.output_dir}")
    print(f"Training used 99% split with seed {config.split_seed}; 1% held out for evaluation.")


if __name__ == "__main__":
    main()


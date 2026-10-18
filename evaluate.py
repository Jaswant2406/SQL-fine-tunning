import json
import re
import sqlite3
from pathlib import Path

import torch
from datasets import load_dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
DATASET_NAME = "b-mc2/sql-create-context"
ADAPTER_DIR = "adapter"
RESULTS_PATH = "results.json"
MAX_NEW_TOKENS = 128
SPLIT_SEED = 42
TEST_SIZE = 0.01


def normalize_sql(text: str) -> str:
    text = text.strip().rstrip(";")
    text = re.sub(r"\s+", " ", text)
    return text.lower()


def extract_sql(output: str) -> str:
    if "SQL:" in output:
        return output.split("SQL:")[-1].strip()
    return output.strip()


def build_prompt(question: str, context: str) -> str:
    return f"Generate SQL:\nQuestion: {question.strip()}\nContext: {context.strip()}\nSQL:"


def build_quantization_config() -> BitsAndBytesConfig:
    compute_dtype = torch.float16
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        compute_dtype = torch.bfloat16

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
    )


def load_tokenizer() -> AutoTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return tokenizer


def load_base_model():
    return AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=build_quantization_config(),
        device_map="auto",
    )


def load_finetuned_model():
    base_model = load_base_model()
    return PeftModel.from_pretrained(base_model, ADAPTER_DIR)


def get_model_device(model) -> torch.device:
    return next(model.parameters()).device


def generate_sql(model, tokenizer, question: str, context: str) -> str:
    prompt = build_prompt(question, context)
    inputs = tokenizer(prompt, return_tensors="pt").to(get_model_device(model))

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            temperature=0.0,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    return tokenizer.decode(outputs[0], skip_special_tokens=True).strip()


def create_schema_db(context: str) -> sqlite3.Connection:
    connection = sqlite3.connect(":memory:")
    cursor = connection.cursor()

    statements = [stmt.strip() for stmt in context.split(";") if stmt.strip()]
    for statement in statements:
        cursor.execute(f"{statement};")

    connection.commit()
    return connection


def executes_without_error(sql: str, context: str) -> bool:
    try:
        connection = create_schema_db(context)
        cursor = connection.cursor()
        cursor.execute(sql)
        cursor.fetchall()
        connection.close()
        return True
    except Exception:
        return False


def load_eval_dataset():
    dataset = load_dataset(DATASET_NAME, split="train")
    split_dataset = dataset.train_test_split(test_size=TEST_SIZE, seed=SPLIT_SEED)
    return split_dataset["test"]


def main() -> None:
    dataset = load_eval_dataset()
    tokenizer = load_tokenizer()

    print(f"Loaded {len(dataset)} held-out evaluation samples from 1% split")
    print("Loading base model...")
    base_model = load_base_model()
    base_model.eval()

    print("Loading fine-tuned adapter...")
    finetuned_model = load_finetuned_model()
    finetuned_model.eval()

    base_exact = 0
    ft_exact = 0
    base_exec = 0
    ft_exec = 0
    examples = []

    for idx, sample in enumerate(dataset, start=1):
        question = sample["question"]
        context = sample["context"]
        gold_sql = sample["answer"]

        base_output = generate_sql(base_model, tokenizer, question, context)
        ft_output = generate_sql(finetuned_model, tokenizer, question, context)

        base_sql = extract_sql(base_output)
        ft_sql = extract_sql(ft_output)

        base_exact += int(normalize_sql(base_sql) == normalize_sql(gold_sql))
        ft_exact += int(normalize_sql(ft_sql) == normalize_sql(gold_sql))
        base_exec += int(executes_without_error(base_sql, context))
        ft_exec += int(executes_without_error(ft_sql, context))

        examples.append(
            {
                "id": idx,
                "question": question,
                "gold_sql": gold_sql,
                "base_output": base_output,
                "finetuned_output": ft_output,
                "base_sql": base_sql,
                "finetuned_sql": ft_sql,
            }
        )
        print(f"Evaluated {idx}/{len(dataset)}")

    total = len(dataset)
    results = {
        "model_name": MODEL_NAME,
        "dataset_name": DATASET_NAME,
        "evaluation_split": f"held-out 1% split from train with seed {SPLIT_SEED}",
        "samples_evaluated": total,
        "base_model": {
            "exact_match_accuracy": round(base_exact / total, 4),
            "execution_accuracy": round(base_exec / total, 4),
        },
        "finetuned_model": {
            "exact_match_accuracy": round(ft_exact / total, 4),
            "execution_accuracy": round(ft_exec / total, 4),
        },
        "examples": examples[:10],
    }

    Path(RESULTS_PATH).write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))
    print(f"Saved evaluation results to .\\{RESULTS_PATH}")


if __name__ == "__main__":
    main()

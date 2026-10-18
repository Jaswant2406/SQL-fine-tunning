import argparse
import json
import os
import re
import sqlite3

import torch
from huggingface_hub import InferenceClient
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
ADAPTER_DIR = "adapter"
MAX_NEW_TOKENS = 128
# This model is available through Hugging Face Inference Providers. The cloud
# app uses the provider router rather than loading model weights itself.
HOSTED_MODEL_NAME = "deepseek-ai/DeepSeek-V3-0324"


class HostedInferenceModel:
    """Text-generation client used by the deployed Streamlit application."""

    def __init__(self, token: str, model_name: str):
        self.client = InferenceClient(provider="auto", token=token)
        self.model_name = model_name

    def generate(self, prompt: str, max_new_tokens: int) -> str:
        try:
            completion = self.client.chat_completion(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_new_tokens,
                temperature=0,
            )
            return completion.choices[0].message.content or ""
        except Exception as exc:
            raise RuntimeError(
                "Hosted inference failed. Check HF_TOKEN and TEXT_TO_SQL_MODEL in Streamlit Secrets."
            ) from exc


def get_hosted_inference_model(
    token: str | None = None, model_name: str | None = None
) -> HostedInferenceModel | None:
    """Return a hosted client when deployment credentials are available."""
    token = token or os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN")
    if not token:
        return None
    model_name = model_name or os.getenv("TEXT_TO_SQL_MODEL", HOSTED_MODEL_NAME)
    return HostedInferenceModel(token, model_name)


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


def build_prompt(question: str, context: str) -> str:
    return f"Generate SQL:\nQuestion: {question.strip()}\nContext: {context.strip()}\nSQL:"


def load_base_model():
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=build_quantization_config(),
        device_map="auto",
    )
    model.eval()
    return model


def load_finetuned_model():
    base_model = load_base_model()
    peft_model = PeftModel.from_pretrained(base_model, ADAPTER_DIR)
    peft_model.eval()
    try:
        merged_model = peft_model.merge_and_unload()
        merged_model.eval()
        return merged_model
    except Exception:
        return peft_model


def get_model_device(model) -> torch.device:
    return next(model.parameters()).device


def _decode_generation(model, tokenizer, prompt: str, max_new_tokens: int = MAX_NEW_TOKENS) -> str:
    if isinstance(model, HostedInferenceModel) or tokenizer is None:
        return model.generate(prompt, max_new_tokens)
    inputs = tokenizer(prompt, return_tensors="pt").to(get_model_device(model))

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=0.0,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def _clean_sql(text: str) -> str:
    """Extract the first SQL statement when the model echoes its prompt or alternatives."""
    text = re.sub(r"```(?:sql)?|```", "", text, flags=re.IGNORECASE).strip()

    # Generated text may include the original prompt (Question/Context/SQL:) and
    # several proposed statements.  Start at the first SQL command, not at the
    # first line of decoded model output.
    command = re.search(r"\b(?:SELECT|WITH|INSERT|UPDATE|DELETE)\b", text, flags=re.IGNORECASE)
    if not command:
        return ""
    sql = text[command.start():]

    # A later prompt label or a second `SQL:` marks the end of the first answer.
    sql = re.split(
        r"\s+(?:SQL\s*:|Question\s*:|Context\s*:|Database\s+context\s*:|Original\s+SQL\s*:|Fix\s+the\s+SQL\b)",
        sql,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0]
    sql = " ".join(line.strip() for line in sql.splitlines() if line.strip())

    # If the model used a semicolon, it is the unambiguous statement boundary.
    if ";" in sql:
        sql = sql.split(";", 1)[0]
    return sql.strip()


def generate_sql(model, tokenizer, question: str, context: str) -> str:
    prompt = build_prompt(question, context)
    decoded = _decode_generation(model, tokenizer, prompt)
    return _clean_sql(decoded)


def self_heal_sql(question: str, context: str, sql: str, error: str, model, tokenizer) -> str:
    prompt = (
        f"The SQL query failed with error: {error}\n"
        f"Original question: {question}\n"
        f"Database context: {context}\n"
        f"Original SQL: {sql}\n"
        "Fix the SQL query and return only the corrected SQL."
    )
    decoded = _decode_generation(model, tokenizer, prompt)
    return _clean_sql(decoded)


def suggest_chart_config(question: str, column_names: list[str], model, tokenizer) -> dict:
    prompt = (
        f"Given this question: {question}\n"
        f"And this SQL result with columns: {', '.join(column_names)}\n"
        "Return JSON only, no explanation:\n"
        "{\n"
        '  "chart_type": "bar" or "line" or "pie" or "table",\n'
        '  "x_column": "column name for x axis",\n'
        '  "y_column": "column name for y axis",\n'
        '  "title": "chart title"\n'
        "}\n"
        "If the question does not request a visualization, return chart_type as table."
    )
    decoded = _decode_generation(model, tokenizer, prompt, max_new_tokens=96)
    match = re.search(r"\{.*\}", decoded, flags=re.DOTALL)
    raw_json = match.group(0) if match else "{}"
    try:
        config = json.loads(raw_json)
    except json.JSONDecodeError:
        lowered = question.lower()
        chart_type = "table"
        if "bar" in lowered:
            chart_type = "bar"
        elif "line" in lowered:
            chart_type = "line"
        elif "pie" in lowered:
            chart_type = "pie"
        config = {
            "chart_type": chart_type,
            "x_column": column_names[0] if column_names else "",
            "y_column": column_names[1] if len(column_names) > 1 else "",
            "title": question.strip(),
        }

    return {
        "chart_type": str(config.get("chart_type", "table")).lower(),
        "x_column": str(config.get("x_column", "")),
        "y_column": str(config.get("y_column", "")),
        "title": str(config.get("title", question.strip() or "Query Result")),
    }


def create_schema_db(context: str) -> sqlite3.Connection:
    connection = sqlite3.connect(":memory:")
    cursor = connection.cursor()
    statements = [stmt.strip() for stmt in context.split(";") if stmt.strip()]
    for statement in statements:
        cursor.execute(f"{statement};")
    connection.commit()
    return connection


def validate_sql(sql: str, context: str) -> tuple[bool, str | None]:
    try:
        connection = create_schema_db(context)
        cursor = connection.cursor()
        cursor.execute(sql)
        cursor.fetchall()
        connection.close()
        return True, None
    except Exception as exc:
        return False, str(exc)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare base and fine-tuned SQL generation.")
    parser.add_argument("--question", required=True, help="Natural language question")
    parser.add_argument("--context", required=True, help="Table schema / SQL CREATE TABLE context")
    args = parser.parse_args()

    tokenizer = load_tokenizer()

    print("Loading base model...")
    base_model = load_base_model()
    print("Loading fine-tuned model...")
    finetuned_model = load_finetuned_model()

    base_sql = generate_sql(base_model, tokenizer, args.question, args.context)
    ft_sql = generate_sql(finetuned_model, tokenizer, args.question, args.context)

    print("\n=== Base Model SQL ===")
    print(base_sql)
    base_ok, base_error = validate_sql(base_sql, args.context)
    if not base_ok:
        base_fixed = self_heal_sql(args.question, args.context, base_sql, base_error, base_model, tokenizer)
        print("\nBase SQL Error:")
        print(base_error)
        print("\nBase Corrected SQL:")
        print(base_fixed)

    print("\n=== Fine-Tuned Model SQL ===")
    print(ft_sql)
    ft_ok, ft_error = validate_sql(ft_sql, args.context)
    if not ft_ok:
        ft_fixed = self_heal_sql(args.question, args.context, ft_sql, ft_error, finetuned_model, tokenizer)
        print("\nFine-Tuned SQL Error:")
        print(ft_error)
        print("\nFine-Tuned Corrected SQL:")
        print(ft_fixed)


if __name__ == "__main__":
    main()

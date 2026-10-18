# TinyLlama SQL Dashboard

A Text-to-SQL project that fine-tunes `TinyLlama/TinyLlama-1.1B-Chat-v1.0` with QLoRA. Ask a question in plain English, provide a database schema (or upload a CSV), and the model generates SQL, runs it against SQLite, and returns a table or chart.

## What it does

- Converts natural-language questions into SQLite SQL.
- Uses a fine-tuned LoRA adapter instead of the base model alone.
- Runs generated SQL against a built-in sample database or an uploaded CSV file.
- Attempts to repair SQL automatically when the first query fails.
- Renders table results and optional bar, line, or pie charts.
- Compares the base and fine-tuned models using exact-match and execution accuracy.

## Dashboard

The Streamlit dashboard uses a retro terminal interface with a database schema tree, generated-SQL view, execution status, data table, and model-evaluation view.

Keyboard shortcuts:

- `/` focuses the question box.
- `Ctrl + Enter` (or `Cmd + Enter` on macOS) runs the query.

## Quick start

### 1. Create and activate a virtual environment

```powershell
cd D:\Sql-finetunning
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

> The model is configured for 4-bit QLoRA and requires a CUDA-capable GPU. If `bitsandbytes` does not work on native Windows, use WSL2 with CUDA enabled.

### 2. Start the dashboard

```powershell
streamlit run app.py
```

Open the address shown in the terminal, usually `http://localhost:8501`.

## Free cloud deployment

Deploy the dashboard on [Streamlit Community Cloud](https://share.streamlit.io/). The deployed app uses hosted inference because the free cloud environment does not provide a CUDA GPU for local 4-bit model loading.

1. Push this repository to GitHub.
2. Sign in to Streamlit Community Cloud with GitHub and select this repository.
3. Set the entry point to `app.py`.
4. In **App settings → Secrets**, copy the values from `.streamlit/secrets.toml.example` and replace `hf_your_access_token` with your Hugging Face access token.
5. Deploy the app.

The token is kept in Streamlit Secrets and is not committed to GitHub. Create a **fine-grained** Hugging Face token with the **Make calls to Inference Providers** permission; a Read-only token cannot run cloud inference. The cloud version uses `deepseek-ai/DeepSeek-V3-0324` by default; set `TEXT_TO_SQL_MODEL` in Secrets to use a different compatible hosted chat model.

### 3. Ask a question

With **Sample Database** selected, try:

```text
Show average salary by department as a bar chart
```

Or:

```text
List all employees in Engineering
```

The dashboard shows the generated SQL, whether it executed successfully, and the query result.

## Using your own CSV

1. Choose **Upload CSV** in the left navigation.
2. Upload a `.csv` file.
3. Ask a question using the CSV column names.

The app normalizes column names, loads the file into an in-memory SQLite table named `uploaded_data`, and builds the schema context automatically. Uploaded files are not saved by the application.

## Sample database

Run this command if you want to recreate the local sample database:

```powershell
python sample_db.py
```

It creates `sample.db` with these tables:

| Table | Columns |
| --- | --- |
| `employees` | `id`, `name`, `salary`, `department` |
| `products` | `id`, `name`, `price`, `category` |
| `orders` | `id`, `product_id`, `quantity`, `date` |

## Training

The adapter is trained on the Hugging Face dataset `b-mc2/sql-create-context`.

```powershell
python train.py
```

Training saves the LoRA adapter and tokenizer files to `adapter/`, including:

```text
adapter/adapter_model.safetensors
adapter/adapter_config.json
```

Training configuration:

- 4-bit NF4 quantization with `BitsAndBytesConfig`
- LoRA rank `16`, alpha `32`, dropout `0.05`
- Target modules: `q_proj` and `v_proj`
- Gradient checkpointing enabled to reduce VRAM use

## Evaluation

Run the evaluation script to compare the base model with the adapter:

```powershell
python evaluate.py
```

It writes results to `results.json`.

Current saved evaluation results:

| Model | Exact-match accuracy | Execution accuracy |
| --- | ---: | ---: |
| Base TinyLlama | 0.00% | 2.80% |
| Fine-tuned adapter | 56.49% | 91.35% |

`exact_match_accuracy` compares normalized SQL text. `execution_accuracy` checks whether generated SQL successfully executes against an SQLite database created from the held-out schema context.

## Command-line inference

Compare the base and fine-tuned models for a single question:

```powershell
python infer.py --question "List all employees in Engineering" --context "CREATE TABLE employees (id INTEGER PRIMARY KEY, name TEXT, salary INTEGER, department TEXT);"
```

The script prints SQL from both models. It also validates the generated query and tries a correction when execution fails.

## Project structure

```text
Sql-finetunning/
├── app.py          # Streamlit terminal dashboard
├── infer.py        # Model loading, SQL generation, repair, and validation
├── train.py        # QLoRA fine-tuning
├── evaluate.py     # Base vs. fine-tuned evaluation
├── sample_db.py    # Local SQLite sample database
├── adapter/        # Trained LoRA adapter and checkpoints
├── results.json    # Saved evaluation output
└── requirements.txt
```

## Requirements

- Python 3.11+
- NVIDIA GPU with CUDA for model loading/training
- PyTorch, Transformers, PEFT, TRL, BitsAndBytes, Streamlit, Pandas, and Plotly (listed in `requirements.txt`)

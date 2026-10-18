import json
import os
import sqlite3
import tempfile
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st
import streamlit.components.v1 as components

from infer import generate_sql, get_hosted_inference_model, load_finetuned_model, load_tokenizer, self_heal_sql, suggest_chart_config
from sample_db import create_database

RESULTS_PATH = Path("results.json")
# Streamlit Community Cloud mounts the repository as read-only. Keep the
# regenerated demo database in the runtime's writable temporary directory.
SAMPLE_DB_PATH = Path(tempfile.gettempdir()) / "tinyllama_sql_sample.db"
SAMPLE_SCHEMA = """CREATE TABLE employees (id INTEGER PRIMARY KEY, name TEXT NOT NULL, salary INTEGER NOT NULL, department TEXT NOT NULL);
CREATE TABLE products (id INTEGER PRIMARY KEY, name TEXT NOT NULL, price REAL NOT NULL, category TEXT NOT NULL);
CREATE TABLE orders (id INTEGER PRIMARY KEY, product_id INTEGER NOT NULL, quantity INTEGER NOT NULL, date TEXT NOT NULL, FOREIGN KEY (product_id) REFERENCES products(id));"""
CHART_KEYWORDS = ("chart", "plot", "graph", "bar", "line", "pie", "distribution", "trend")


def inject_terminal_theme():
    st.markdown("""<style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto+Mono:wght@400;500;700&display=swap');
    :root{--void:#0A0A0A;--panel:#121212;--line:#2A2A2A;--paper:#E8E8E8;--mist:#989898;--phosphor:#F5E400;--verified:#3FCF6E;--alert:#FF5C5C}
    html,body,[class*="css"],[data-testid="stAppViewContainer"],[data-testid="stSidebar"]{font-family:'Roboto Mono',monospace!important}
    [data-testid="stAppViewContainer"]{background:var(--void);color:var(--paper)}
    [data-testid="stAppViewContainer"]:before{content:"";position:fixed;inset:0;pointer-events:none;opacity:.05;z-index:100;background-image:url("data:image/svg+xml,%3Csvg viewBox='0 0 180 180' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='.9' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)'/%3E%3C/svg%3E")}
    [data-testid="stHeader"]{background:var(--void);border-bottom:1px solid var(--line)}.block-container{max-width:none!important;padding:1.25rem 2rem 3rem!important}
    h1,h2,h3,p,label,span,div,button,input,textarea{font-family:'Roboto Mono',monospace!important}h1{font-size:20px!important;letter-spacing:-.6px;font-weight:700!important}h2,h3{font-size:13px!important;font-weight:500!important;text-transform:lowercase}p,label,[data-testid="stCaptionContainer"]{font-size:14px!important;color:var(--paper)!important}
    [data-testid="stSidebar"]{background:var(--panel);border-right:1px solid var(--line)}[data-testid="stSidebar"]>div:first-child{padding-top:1.35rem}[data-testid="stSidebar"] [data-testid="stFileUploader"]{border:1px solid var(--line);border-radius:0;padding:.45rem}[data-testid="stSidebar"] [data-testid="stRadio"] label{padding:0!important;margin:0 0 .35rem!important}[data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) p{color:var(--phosphor)!important}[data-testid="stSidebar"] [data-testid="stRadio"] input{display:none}
    [data-testid="stTextInput"] input,[data-testid="stTextArea"] textarea{background:var(--void)!important;color:var(--paper)!important;border:1px solid var(--line)!important;border-radius:0!important;box-shadow:none!important;caret-color:var(--phosphor)!important;font-size:14px!important}[data-testid="stTextInput"] input:focus,[data-testid="stTextArea"] textarea:focus{border-color:var(--phosphor)!important;box-shadow:0 0 0 1px var(--phosphor)!important}[data-testid="stTextInput"] input::placeholder,textarea::placeholder{color:var(--mist)!important}
    .stButton>button,[data-testid="stFormSubmitButton"] button{border-radius:0!important;background:var(--phosphor)!important;border:1px solid var(--phosphor)!important;color:var(--void)!important;font-weight:700!important;box-shadow:none!important}.stButton>button:hover,[data-testid="stFormSubmitButton"] button:hover{background:var(--paper)!important;border-color:var(--paper)!important}
    [data-testid="stTabs"] [data-baseweb="tab-list"]{gap:1.1rem;border-bottom:1px solid var(--line)}[data-testid="stTabs"] button{color:var(--mist)!important;font-size:14px!important;padding:.55rem 0!important}[data-testid="stTabs"] button[aria-selected="true"]{color:var(--phosphor)!important;border-bottom-color:var(--phosphor)!important}[data-testid="stTabs"] button:before{content:"[ "}[data-testid="stTabs"] button:after{content:" ]"}
    .terminal-label{font-size:13px;color:var(--paper);font-weight:500;margin:1.25rem 0 .55rem}.tree{white-space:pre;color:var(--mist);font-size:13px;line-height:1.62;margin:.35rem 0 1rem}.terminal-status{color:var(--mist);font-size:13px;margin:.65rem 0}.terminal-status .ok{color:var(--verified)}.terminal-status .ai{color:var(--phosphor)}.sql-block{border-left:1px solid var(--phosphor);padding:.1rem 0 .1rem .85rem;margin:1.15rem 0;background:var(--void)}.sql-block .label{color:var(--phosphor);font-size:13px;margin-bottom:.5rem}.sql-block pre{white-space:pre-wrap;color:var(--paper);font-size:14px;margin:0}.error-line{color:var(--alert);font-size:13px;margin:.4rem 0}
    [data-testid="stDataFrame"],[data-testid="stDataFrame"]>div{border-radius:0!important;border-color:var(--line)!important}[data-testid="stDataFrame"] *{font-family:'Roboto Mono',monospace!important}.metric-row{display:flex;align-items:stretch;gap:2.25rem;margin:1.7rem 0 1.6rem}.metric{min-width:220px}.metric+.metric{border-left:1px solid var(--line);padding-left:2.25rem}.metric .value{font-size:64px;line-height:1;font-weight:700;letter-spacing:-4px;color:var(--paper)}.metric.fine .value{color:var(--phosphor)}.metric .meta{font-size:12px;color:var(--mist);margin-top:.55rem}.terminal-copy{color:var(--mist);max-width:780px;line-height:1.6;font-size:14px}.topline{display:flex;justify-content:space-between;align-items:baseline;margin-bottom:.8rem}.topline .hint{color:var(--mist);font-size:12px}[data-testid="stAlert"]{border-radius:0!important;border:1px solid var(--line)!important;background:var(--panel)!important}@media(max-width:700px){.block-container{padding:1rem!important}.metric-row{gap:1rem}.metric+.metric{padding-left:1rem}.metric .value{font-size:42px}}
    </style>""", unsafe_allow_html=True)


def install_keyboard_shortcuts():
    components.html("""<script>
    const d=window.parent.document;
    if(!d.querySelector('[data-terminal-shortcuts]')){const m=d.createElement('span');m.dataset.terminalShortcuts='true';m.style.display='none';d.body.appendChild(m);d.addEventListener('keydown',(e)=>{const t=e.target;if(e.key==='/'&&t.tagName!=='INPUT'&&t.tagName!=='TEXTAREA'){e.preventDefault();d.querySelector('textarea[aria-label="Ask a question or request a chart"]')?.focus()}if((e.ctrlKey||e.metaKey)&&e.key==='Enter'){const b=[...d.querySelectorAll('button')].find(x=>x.innerText.includes('Generate SQL & Run'));if(b){e.preventDefault();b.click()}}})}
    </script>""", height=0)


@st.cache_resource
def get_query_model():
    hosted_model = get_hosted_inference_model()
    if hosted_model is not None:
        return None, hosted_model
    return load_tokenizer(), load_finetuned_model()


def read_results():
    if RESULTS_PATH.exists(): return json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
    return {"base_model":{"exact_match_accuracy":0,"execution_accuracy":0},"finetuned_model":{"exact_match_accuracy":0,"execution_accuracy":0}}


def infer_sqlite_type(dtype):
    if pd.api.types.is_integer_dtype(dtype): return "INTEGER"
    return "REAL" if pd.api.types.is_float_dtype(dtype) else "TEXT"


def get_connection(source, uploaded):
    if source == "Sample Database":
        create_database(SAMPLE_DB_PATH); connection = sqlite3.connect(SAMPLE_DB_PATH); connection.row_factory = sqlite3.Row
        return connection, SAMPLE_SCHEMA, None
    dataframe = pd.read_csv(uploaded); dataframe.columns = [str(c).strip().replace(" ", "_") for c in dataframe.columns]
    connection = sqlite3.connect(":memory:"); connection.row_factory = sqlite3.Row; dataframe.to_sql("uploaded_data", connection, if_exists="replace", index=False)
    columns = ",\n    ".join(f"{name} {infer_sqlite_type(dtype)}" for name, dtype in dataframe.dtypes.items())
    return connection, f"CREATE TABLE uploaded_data (\n    {columns}\n);", dataframe


def execute_sql(connection, sql):
    try: return pd.read_sql_query(sql, connection), None
    except Exception as exc: return pd.DataFrame(), str(exc)


def run_query(question, context, connection, model, tokenizer):
    original = generate_sql(model, tokenizer, question, context); result, error = execute_sql(connection, original)
    corrected = corrected_error = None
    if error:
        corrected = self_heal_sql(question, context, original, error, model, tokenizer); result, corrected_error = execute_sql(connection, corrected)
    return original, error, corrected, corrected_error, result


def sql_panel(label, sql):
    safe = sql.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    st.markdown(f'<div class="sql-block"><div class="label">&gt; {label} <span style="float:right;color:#989898">[ copy ]</span></div><pre>{safe}</pre></div>', unsafe_allow_html=True)


def render_chart(question, dataframe, model, tokenizer):
    if dataframe.empty or not any(k in question.lower() for k in CHART_KEYWORDS): return None
    config = suggest_chart_config(question, list(dataframe.columns), model, tokenizer); kind, x, y = config["chart_type"], config["x_column"], config["y_column"]
    if x not in dataframe.columns or y not in dataframe.columns: return None
    if kind == "bar": fig = px.bar(dataframe, x=x, y=y, title=config["title"], color_discrete_sequence=["#F5E400"])
    elif kind == "line": fig = px.line(dataframe, x=x, y=y, title=config["title"], markers=True, color_discrete_sequence=["#F5E400"])
    elif kind == "pie": fig = px.pie(dataframe, names=x, values=y, title=config["title"], color_discrete_sequence=["#F5E400", "#E8E8E8", "#2A2A2A"])
    else: return None
    fig.update_layout(paper_bgcolor="#0A0A0A",plot_bgcolor="#0A0A0A",font={"family":"Roboto Mono","color":"#E8E8E8"},title_font={"size":14})
    fig.update_xaxes(gridcolor="#2A2A2A",linecolor="#2A2A2A"); fig.update_yaxes(gridcolor="#2A2A2A",linecolor="#2A2A2A")
    return fig


def main():
    st.set_page_config(page_title="TinyLlama SQL Dashboard", layout="wide")
    inject_terminal_theme(); install_keyboard_shortcuts()
    st.markdown('<div class="topline"><h1>TinyLlama SQL Dashboard</h1><span class="hint">[ / focus input ] &nbsp; [ ctrl+enter run ]</span></div>', unsafe_allow_html=True)
    tokenizer, model = get_query_model(); results = read_results()
    query_tab, eval_tab = st.tabs(["Query your data", "Model evaluation"])
    with query_tab:
        with st.sidebar:
            st.markdown('<div class="terminal-label">/ data_source</div>', unsafe_allow_html=True)
            source = st.radio("Data source", ["Sample Database", "Upload CSV"], label_visibility="collapsed")
            upload = st.file_uploader("upload_csv", type=["csv"]) if source == "Upload CSV" else None
            st.markdown('<div class="terminal-label">/ schema_tree</div>', unsafe_allow_html=True)
            if source == "Sample Database": st.markdown('<div class="tree">├─ employees\n│  ├─ id: INTEGER\n│  ├─ name: TEXT\n│  ├─ salary: INTEGER\n│  └─ department: TEXT\n├─ products\n│  ├─ id: INTEGER\n│  ├─ name: TEXT\n│  ├─ price: REAL\n│  └─ category: TEXT\n└─ orders\n   ├─ product_id: INTEGER\n   ├─ quantity: INTEGER\n   └─ date: TEXT</div>', unsafe_allow_html=True)
            else: st.markdown('<div class="tree">└─ uploaded_data</div>', unsafe_allow_html=True)
        if source == "Upload CSV" and upload is None:
            st.info("[ awaiting csv upload ]"); return
        connection, context, preview = get_connection(source, upload)
        hosted_mode = bool(os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN"))
        st.markdown(f'<div class="terminal-status">[ source: {"sample.sqlite" if preview is None else "uploaded_data"} ]</div>', unsafe_allow_html=True)
        if hosted_mode:
            st.markdown('<div class="terminal-status">[ inference: hosted ]</div>', unsafe_allow_html=True)
        if preview is not None: st.dataframe(preview.head(10), use_container_width=True, hide_index=True)
        with st.form("query_form"):
            question = st.text_area("Ask a question or request a chart", value="Show me average salary by department as a bar chart", height=96)
            submitted = st.form_submit_button("Generate SQL & Run [ CTRL+ENTER ]", type="primary", use_container_width=True)
        if submitted:
            try:
                with st.spinner("[ generating_sql... ]"):
                    original, error, corrected, corrected_error, dataframe = run_query(question, context, connection, model, tokenizer)
            except RuntimeError as exc:
                st.markdown(f'<div class="error-line">[ inference_error ] {exc}</div>', unsafe_allow_html=True)
                connection.close()
                return
            sql_panel("generated_sql", original)
            if error:
                st.markdown(f'<div class="error-line">[ execution_error ] {error}</div>', unsafe_allow_html=True); sql_panel("ai_corrected_sql", corrected)
            if corrected_error: st.markdown(f'<div class="error-line">[ correction_failed ] {corrected_error}</div>', unsafe_allow_html=True)
            else:
                state = '<span class="ok">[ executed ]</span>' if not error else '<span class="ai">[ auto-corrected ]</span> <span class="ok">[ executed ]</span>'
                st.markdown(f'<div class="terminal-status">[ sql generated ] {state}</div>', unsafe_allow_html=True)
                if dataframe.empty: st.markdown('<div class="terminal-status">[ result: no rows ]</div>', unsafe_allow_html=True)
                else:
                    chart = render_chart(question, dataframe, model, tokenizer)
                    if chart: st.plotly_chart(chart, use_container_width=True)
                    st.markdown('<div class="terminal-label">&gt; query_result</div>', unsafe_allow_html=True); st.dataframe(dataframe, use_container_width=True, hide_index=True)
        connection.close()
    with eval_tab:
        base, fine = results["base_model"]["execution_accuracy"]*100, results["finetuned_model"]["execution_accuracy"]*100
        st.markdown(f'<div class="metric-row"><div class="metric"><div class="value">{base:.1f}%</div><div class="meta">base model / execution accuracy</div></div><div class="metric fine"><div class="value">{fine:.2f}%</div><div class="meta">fine-tuned model / execution accuracy</div></div></div><p class="terminal-copy">TinyLlama was fine-tuned with QLoRA on natural-language questions paired with database schemas and SQL. The adapter improves the chance that generated SQL can execute against the supplied schema.</p>', unsafe_allow_html=True)
        data = pd.DataFrame({"Metric":["Exact match accuracy","Execution accuracy"],"Base model":[results["base_model"]["exact_match_accuracy"],results["base_model"]["execution_accuracy"]],"Fine-tuned model":[results["finetuned_model"]["exact_match_accuracy"],results["finetuned_model"]["execution_accuracy"]]})
        fig = px.bar(data.melt(id_vars="Metric",var_name="Model",value_name="Score"),x="Metric",y="Score",color="Model",barmode="group",color_discrete_map={"Base model":"#E8E8E8","Fine-tuned model":"#F5E400"})
        fig.update_layout(paper_bgcolor="#0A0A0A",plot_bgcolor="#0A0A0A",font={"family":"Roboto Mono","color":"#E8E8E8"},legend_title_text="")
        fig.update_xaxes(gridcolor="#2A2A2A",linecolor="#2A2A2A"); fig.update_yaxes(gridcolor="#2A2A2A",linecolor="#2A2A2A",range=[0,1]); st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__": main()

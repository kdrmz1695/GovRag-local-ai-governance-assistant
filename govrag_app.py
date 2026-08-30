"""Local Streamlit UI for the existing GovRAG pipeline.

Place beside start_govrag.ps1 in the project root. Run: python govrag_app.py
No changes to src/generation/rag_answer.py or retrieval settings are needed.
"""
from __future__ import annotations

import contextlib
import html
import json
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
WORKER_FLAG = "--govrag-worker"
MAX_QUESTION_LENGTH = 2000
REQUEST_TIMEOUT_SECONDS = 900
EXAMPLES = {
    "GDPR · Impact assessments": (
        "When is a data protection impact assessment required under the GDPR?"
    ),
    "AI Act · High-risk systems": (
        "How does the EU AI Act determine whether an AI system is high-risk?"
    ),
    "EDPB · Anonymous models": (
        "According to EDPB Opinion 28/2024, what conditions must be met for an AI "
        "model trained on personal data to be considered anonymous?"
    ),
}


def safe_error(error: Exception) -> str:
    """Do not expose database credentials, HTTP bodies or tracebacks in the UI."""
    message = str(error).lower()
    if isinstance(error, ModuleNotFoundError):
        return (
            "A GovRAG dependency is missing. Use the same activated .venv313 "
            "environment as the working CLI."
        )
    if "missing required environment variable" in message:
        return "A required setting is missing. Check the project-root .env file."
    if type(error).__module__.startswith("psycopg2"):
        return "PostgreSQL is unavailable or a database operation failed. Check the service and DB settings."
    if "citation" in message or "source markers" in message:
        return (
            "The generated answer did not pass the existing citation-marker check. "
            "No answer is shown. Try a more focused question."
        )
    if "cuda" in message or "out of memory" in message:
        return "GPU inference failed. Close other GPU-heavy applications and check the working CLI."
    if any(word in message for word in ("local_files_only", "huggingface", "offline", "cached")):
        return "The reranker could not load from its local cache. Check the existing offline setup."
    if "foundry" in message or "timed out" in message:
        return "Foundry Local did not complete the request. Run start_govrag.ps1 and retry."
    if "candidates" in message or "sources" in message:
        return "No usable source passages were returned. Check the loaded corpus and Gold embeddings."
    return "The RAG request failed. Run the same question with rag_answer.py to inspect the original error."


def run_worker() -> int:
    """One request, one call to the unchanged API, then release the process."""
    try:
        request = json.load(sys.stdin)
        question = request.get("question", "")
        if not isinstance(question, str) or not question.strip():
            raise ValueError("Question cannot be empty.")
        if len(question) > MAX_QUESTION_LENGTH:
            raise ValueError("Question is too long.")
        # Keep the JSON channel separate from the pipeline's normal progress logs.
        with contextlib.redirect_stdout(sys.stderr):
            sys.path.insert(0, str(ROOT / "src"))
            from generation.rag_answer import (
                DEFAULT_CANDIDATE_K,
                DEFAULT_TOP_K,
                generate_rag_answer,
                source_label,
            )

            result = generate_rag_answer(question=question.strip())
            result["source_labels"] = [source_label(source) for source in result["sources"]]
            result["candidate_k"] = DEFAULT_CANDIDATE_K
            result["top_k"] = DEFAULT_TOP_K
        print(json.dumps({"ok": True, "result": result}, ensure_ascii=False))
        return 0
    except Exception as error:
        print(json.dumps({"ok": False, "error": safe_error(error)}, ensure_ascii=False))
        return 1


def request_answer(question: str) -> dict:
    """Fresh imports read .env again after a Foundry port change; no shell use."""
    worker_env = os.environ.copy()
    worker_env.update({"PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"})
    try:
        completed = subprocess.run(
            [sys.executable, "-u", str(Path(__file__).resolve()), WORKER_FLAG],
            input=json.dumps({"question": question}, ensure_ascii=False),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            cwd=ROOT,
            env=worker_env,
            timeout=REQUEST_TIMEOUT_SECONDS,
            check=False,
        )
    except subprocess.TimeoutExpired as error:
        raise RuntimeError("The request timed out. Check Foundry Local and PostgreSQL, then retry.") from error
    except OSError as error:
        raise RuntimeError("The Python worker could not start. Check the project path and virtual environment.") from error
    try:
        payload = json.loads(completed.stdout)
    except (json.JSONDecodeError, TypeError) as error:
        raise RuntimeError("The inference process exited without a result. Check the CLI and available GPU memory.") from error
    if not isinstance(payload, dict):
        raise RuntimeError("The inference process returned an unexpected response.")
    if not payload.get("ok"):
        raise RuntimeError(payload.get("error") or "The RAG request failed.")
    if completed.returncode != 0:
        raise RuntimeError("The inference process did not finish successfully.")
    result = payload.get("result")
    if not isinstance(result, dict) or not result.get("answer") or not result.get("sources"):
        raise RuntimeError("The inference process returned an incomplete result.")
    return result


def text_html(text: str, *, citations: bool = False) -> str:
    """Escape all model/document text; never render its remote images or HTML."""
    escaped = html.escape(str(text))
    if citations:
        escaped = re.sub(r"\[S\d+\]", lambda match: f'<span class="cite">{match[0]}</span>', escaped)
    return f'<div class="passage">{escaped}</div>'


def report_text(result: dict) -> str:
    lines = ["GovRAG — research output", "", result["question"], "", result["answer"], "", "Sources"]
    used = set(result.get("used_source_numbers", []))
    for number, label in enumerate(result["source_labels"], start=1):
        usage = "cited" if number in used else "retrieved, not cited"
        lines.append(f"[S{number}] {label} ({usage})")
    lines.extend(["", "Research prototype, not legal advice. Citation markers do not verify legal correctness."])
    return "\n".join(lines)


def show_app() -> None:
    import streamlit as st

    st.set_page_config(page_title="GovRAG | Legal research", page_icon="G", layout="wide")
    st.markdown(
        """<style>
        .stApp {background:#f7f8f4;color:#172d28;}
        .block-container {max-width:1200px;padding-top:2.3rem;padding-bottom:2rem;}
        [data-testid="stSidebar"] {background:#edf1eb;border-right:1px solid #d8e1d7;}
        h1,h2,h3 {color:#143d30;letter-spacing:-.025em;}
        .eyebrow {font:600 .75rem system-ui;letter-spacing:.15em;color:#517366;margin-bottom:.6rem;}
        .intro {font-size:1.1rem;color:#52645c;margin-bottom:1.4rem;}
        .passage {white-space:pre-wrap;overflow-wrap:anywhere;font-size:1rem;line-height:1.8;}
        .cite {color:#165d45;background:#e4f1e9;border-radius:4px;padding:1px 4px;font-weight:600;}
        .empty {padding:2rem;border:1px dashed #b8c9bd;border-radius:12px;color:#5c7064;}
        [data-testid="stForm"] {background:white;border:1px solid #d8e1d7;border-radius:12px;}
        button[kind="primary"] {background:#165541;border-color:#165541;}
        @media(max-width:700px){.block-container{padding-top:1.2rem;}}
        </style>""",
        unsafe_allow_html=True,
    )

    @st.cache_resource
    def inference_lock():
        # Shared across tabs/sessions in this Streamlit server, not across CLI runs.
        return threading.Lock()

    def choose_example(question: str) -> None:
        st.session_state["question"] = question
        st.session_state.pop("result", None)
        st.session_state.pop("error", None)

    with st.sidebar:
        st.title("GovRAG")
        st.caption("AI GOVERNANCE ASSISTANT")
        st.divider()
        st.markdown("**Document collection**")
        st.write("GDPR")
        st.write("EU AI Act")
        st.write("EDPB Opinion 28/2024")
        st.divider()
        st.markdown("**Local pipeline**")
        st.caption("PostgreSQL + pgvector retrieval\n\nQwen3 reranking\n\nFoundry Local answer generation")
        st.caption("Retrieval settings are read from your existing .env. The UI does not change them.")
        st.divider()
        st.caption("Research prototype. Verify legal conclusions against the original documents.")
        with st.expander("Startup help"):
            st.write("Start PostgreSQL and run your existing start_govrag.ps1 before asking a question.")
            st.code("python govrag_app.py", language="powershell")
            st.caption("Keep the terminal open. Stop the app with Ctrl+C.")

    st.markdown('<div class="eyebrow">LOCAL LEGAL RESEARCH · V1</div>', unsafe_allow_html=True)
    st.title("Ask a question. Inspect the evidence.")
    st.markdown('<div class="intro">Explore AI governance documents with source-linked answers.</div>', unsafe_allow_html=True)

    for column, (label, question) in zip(st.columns(3), EXAMPLES.items()):
        column.button(label, on_click=choose_example, args=(question,), width="stretch")

    with st.form("question_form"):
        question = st.text_area(
            "Your question",
            key="question",
            placeholder="What does the GDPR require before high-risk processing?",
            height=110,
            max_chars=MAX_QUESTION_LENGTH,
        )
        submitted = st.form_submit_button("Generate answer", type="primary")
    st.caption("One question at a time. The first request may take longer while local models load.")

    if submitted:
        st.session_state.pop("result", None)
        st.session_state.pop("error", None)
        clean_question = question.strip()
        if not clean_question:
            st.session_state["error"] = "Enter a question first."
        elif not (ROOT / "src" / "generation" / "rag_answer.py").is_file():
            st.session_state["error"] = (
                "Place govrag_app.py in the gov_rag project root, beside start_govrag.ps1. "
                "The existing src/generation/rag_answer.py could not be found."
            )
        else:
            lock = inference_lock()
            if not lock.acquire(blocking=False):
                st.session_state["error"] = "Another request is using the GPU. Wait for it to finish, then retry."
            else:
                try:
                    started = time.perf_counter()
                    with st.spinner("Retrieving evidence, reranking and generating an answer…"):
                        result = request_answer(clean_question)
                    result["elapsed_seconds"] = time.perf_counter() - started
                    st.session_state["result"] = result
                except RuntimeError as error:
                    st.session_state["error"] = str(error)
                finally:
                    lock.release()

    if st.session_state.get("error"):
        st.error(st.session_state["error"])

    result = st.session_state.get("result")
    if result:
        st.divider()
        st.markdown(
            f'<div class="eyebrow">RESULT FOR: {html.escape(result["question"])}</div>',
            unsafe_allow_html=True,
        )
        answer_column, sources_column = st.columns([1.45, 1], gap="large")
        with answer_column:
            st.subheader("Answer")
            st.caption(
                f"{result['elapsed_seconds']:.1f} seconds · "
                f"{len(result['sources'])} evidence passages · "
                f"top-{result['candidate_k']} → top-{result['top_k']}"
            )
            st.markdown(text_html(result["answer"], citations=True), unsafe_allow_html=True)
            st.caption("Citation-marker checks passed. This does not verify the legal accuracy or support for each claim.")
            st.download_button(
                "Download answer and sources",
                data=report_text(result),
                file_name="govrag_answer.txt",
                mime="text/plain",
                on_click="ignore",
            )
        with sources_column:
            st.subheader("Evidence")
            used = set(result["used_source_numbers"])
            for number, (source, label) in enumerate(zip(result["sources"], result["source_labels"]), start=1):
                usage = "Cited" if number in used else "Retrieved · not cited"
                reference = source.get("section_reference") or "Source passage"
                with st.expander(f"[S{number}] {reference} · {usage}", expanded=number == 1):
                    st.caption(label)
                    st.markdown(text_html(source["chunk_text"]), unsafe_allow_html=True)
                    st.caption(
                        f"Chunk {source.get('chunk_id', '—')} · "
                        f"Vector {source['vector_similarity']:.4f} · "
                        f"Reranker {source['reranker_score']:.4f}"
                    )
            st.caption("Scores describe retrieval relevance, not confidence in legal correctness.")
    elif not st.session_state.get("error"):
        st.markdown(
            '<div class="empty">Your answer and its source passages will appear here. '
            'Choose an example above or write your own question.</div>',
            unsafe_allow_html=True,
        )


def main() -> int:
    if WORKER_FLAG in sys.argv[1:]:
        return run_worker()
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
    except ModuleNotFoundError:
        print('Install the UI dependency first: python -m pip install streamlit==1.62.0')
        return 1
    if get_script_run_ctx(suppress_warning=True) is not None:
        show_app()
        return 0
    # The short python command is also safe for the IDE's Run button.
    command = [
        sys.executable, "-m", "streamlit", "run", str(Path(__file__).resolve()),
        "--server.address=127.0.0.1", "--server.port=8501",
        "--server.headless=false", "--server.fileWatcherType=none",
        "--browser.gatherUsageStats=false", "--client.toolbarMode=minimal",
        "--theme.base=light", "--theme.primaryColor=#165541",
        "--theme.backgroundColor=#f7f8f4", "--theme.secondaryBackgroundColor=#edf1eb",
        "--theme.textColor=#172d28", "--theme.font=sans serif",
    ]
    try:
        return subprocess.call(command, cwd=ROOT)
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    exit_code = main()
    if exit_code:
        raise SystemExit(exit_code)

from __future__ import annotations

import argparse
import gc
import json
import re
import sys
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import torch


SRC_ROOT = Path(__file__).resolve().parents[1]

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


from retrieval.semantic_search import (
    DEFAULT_CANDIDATE_K,
    DEFAULT_TOP_K,
    FOUNDRY_BASE_URL,
    format_pages,
    get_db_connection,
    load_reranker,
    request_query_embedding,
    required_env,
    rerank_candidates,
    retrieve_vector_candidates,
)


CHAT_MODEL_ID = required_env("CHAT_MODEL_ID")
CHAT_TEMPERATURE = float(required_env("CHAT_TEMPERATURE"))
CHAT_MAX_TOKENS = int(required_env("CHAT_MAX_TOKENS"))


DOCUMENT_LABELS = {
    "gdpr_2016_679": "GDPR (Regulation (EU) 2016/679)",
    "eu_ai_act_2024_1689": "EU AI Act (Regulation (EU) 2024/1689)",
    "edpb_opinion_202428_ai-models_en": "EDPB Opinion 28/2024 on AI Models",
}


SYSTEM_PROMPT = """You are GovRAG, a local AI governance assistant.

Answer the user's question using only the supplied source passages.

Rules:
1. Do not use outside knowledge or invent legal rules, article numbers, pages, quotations, or facts.
2. Every paragraph or bullet containing a legal claim must end with one or more exact source markers such as [S1] or [S2].
3. The only permitted citation format is [S<number>]. Never use [Article 35], [Recital 91], document names, or page numbers as citation markers.
4. Article and Recital names may appear as normal text, but the supporting citation must still use [S1], [S2], and so on.
5. Prefer binding provisions such as Articles when stating legal obligations. Use Recitals only as explanatory context and clearly distinguish them from binding provisions.
6. If the supplied passages are insufficient, explicitly say so. Do not guess.
7. Answer only the question asked. Do not add unrelated requirements or procedures.
8. Do not present illustrative examples as an exhaustive legal list.
9. Answer in the same language as the user's question.
10. Before returning the answer, verify that every legal paragraph or bullet contains at least one valid [S#] marker.
"""


def document_label(document_name: str) -> str:
    return DOCUMENT_LABELS.get(document_name, document_name)


def source_label(source: dict) -> str:
    section_reference = source.get("section_reference") or "unknown section"
    pages = format_pages(source.get("page_start"), source.get("page_end"))

    return (
        f"{document_label(source['document_name'])} — "
        f"{section_reference} — page(s) {pages}"
    )


def build_context(sources: list[dict]) -> str:
    context_blocks = []

    for source_number, source in enumerate(sources, start=1):
        section_type = source.get("section_type") or "unknown"
        section_reference = source.get("section_reference") or "unknown"
        section_title = source.get("section_title") or "unknown"
        pages = format_pages(source.get("page_start"), source.get("page_end"))
        chunk_text = " ".join(source["chunk_text"].split())

        context_blocks.append(
            "\n".join(
                [
                    f"[S{source_number}]",
                    f"Document: {document_label(source['document_name'])}",
                    f"Section type: {section_type}",
                    f"Section reference: {section_reference}",
                    f"Section title: {section_title}",
                    f"Pages: {pages}",
                    f"Passage: {chunk_text}",
                ]
            )
        )

    return "\n\n".join(context_blocks)


def build_user_prompt(question: str, sources: list[dict]) -> str:
    allowed_markers = ", ".join(
        f"[S{number}]"
        for number in range(1, len(sources) + 1)
    )
    return (
        f"Question:\n{question.strip()}\n\n"
        "Source passages:\n"
        f"{build_context(sources)}\n\n"
        f"Allowed citation markers: {allowed_markers}\n"
        "Use only these exact markers for citations. "
        "Every legal paragraph or bullet must contain at least one marker. "
        "Answer only the question that was asked."
    )


def request_chat_completion(question: str, sources: list[dict]) -> str:
    payload = json.dumps(
        {
            "model": CHAT_MODEL_ID,
            "messages": [
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": build_user_prompt(question, sources),
                },
            ],
            "temperature": CHAT_TEMPERATURE,
            "max_tokens": CHAT_MAX_TOKENS,
        }
    ).encode("utf-8")

    request = Request(
        url=f"{FOUNDRY_BASE_URL}/chat/completions",
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
        method="POST",
    )

    try:
        with urlopen(request, timeout=600) as response:
            response_body = response.read().decode("utf-8")

    except HTTPError as error:
        error_body = error.read().decode("utf-8", errors="replace")

        raise RuntimeError(
            f"Foundry chat request failed with HTTP "
            f"{error.code}: {error_body}"
        ) from error

    except URLError as error:
        raise RuntimeError(
            f"Could not reach Foundry Local at {FOUNDRY_BASE_URL}."
        ) from error

    response_json = json.loads(response_body)
    choices = response_json.get("choices")

    if not isinstance(choices, list) or not choices:
        raise RuntimeError(
            f"Unexpected Foundry chat response: {response_json}"
        )

    message = choices[0].get("message")

    if not isinstance(message, dict):
        raise RuntimeError(
            f"Foundry did not return a valid message: {response_json}"
        )

    answer = message.get("content")

    if not isinstance(answer, str) or not answer.strip():
        raise RuntimeError("Foundry returned an empty answer.")

    return answer.strip()


def validate_answer_citations(answer: str, source_count: int) -> list[int]:
    citation_numbers = [
        int(value)
        for value in re.findall(r"\[S(\d+)]", answer)
    ]

    if not citation_numbers:
        raise RuntimeError(
            "The generated answer contains no valid [S#] citation markers."
        )
    invalid_citations = sorted(
        {
            number
            for number in citation_numbers
            if number < 1 or number > source_count
        }
    )

    if invalid_citations:
        raise RuntimeError(
            "The generated answer contains invalid source markers: "
            f"{invalid_citations}"
        )

    return sorted(set(citation_numbers))


def retrieve_evidence(
    question: str,
    candidate_k: int,
    top_k: int,
) -> list[dict]:
    print("Creating query embedding...")
    query_embedding = request_query_embedding(question)
    print(f"Query embedding created: {len(query_embedding)} dimensions")

    connection = get_db_connection()
    cursor = connection.cursor()

    try:
        print(f"Retrieving {candidate_k} pgvector candidates...")
        candidates = retrieve_vector_candidates(
            cursor=cursor,
            query_embedding=query_embedding,
            candidate_k=candidate_k,
        )

    finally:
        cursor.close()
        connection.close()

    if not candidates:
        raise RuntimeError("pgvector did not return any candidates.")

    print(f"Retrieved {len(candidates)} vector candidates.")
    reranker = load_reranker()

    try:
        print(f"Reranking candidates and selecting top-{top_k}...")
        sources = rerank_candidates(
            question=question,
            candidates=candidates,
            model=reranker,
            top_k=top_k,
        )

    finally:
        del reranker
        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if not sources:
        raise RuntimeError("Reranker did not return any sources.")

    print(f"Selected {len(sources)} evidence passages.")
    print("Reranker released from the Python GPU process.")
    return sources


def generate_rag_answer(
    question: str,
    candidate_k: int = DEFAULT_CANDIDATE_K,
    top_k: int = DEFAULT_TOP_K,
) -> dict:
    clean_question = question.strip()

    if not clean_question:
        raise ValueError("Question cannot be empty.")

    if candidate_k <= 0:
        raise ValueError("candidate_k must be greater than zero.")

    if top_k <= 0:
        raise ValueError("top_k must be greater than zero.")

    if top_k > candidate_k:
        raise ValueError("top_k cannot be greater than candidate_k.")

    sources = retrieve_evidence(
        question=clean_question,
        candidate_k=candidate_k,
        top_k=top_k,
    )

    print(f"Generating grounded answer with {CHAT_MODEL_ID}...")
    answer = request_chat_completion(clean_question, sources)
    used_source_numbers = validate_answer_citations(
        answer=answer,
        source_count=len(sources),
    )

    return {
        "question": clean_question,
        "answer": answer,
        "sources": sources,
        "used_source_numbers": used_source_numbers,
        "chat_model_id": CHAT_MODEL_ID,
    }


def print_rag_result(result: dict) -> None:
    print()
    print("=" * 100)
    print("GOVRAG ANSWER")
    print("=" * 100)
    print(result["answer"])
    print()
    print("SOURCES")

    used_source_numbers = set(result["used_source_numbers"])

    for source_number, source in enumerate(result["sources"], start=1):
        usage_label = "cited" if source_number in used_source_numbers else "retrieved"
        print(
            f"[S{source_number}] {source_label(source)} "
            f"({usage_label}; reranker={source['reranker_score']:.4f}; "
            f"vector={source['vector_similarity']:.4f})"
        )

    if not used_source_numbers:
        print()
        print(
            "WARNING: The model returned an answer without [S#] citation markers."
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a citation-ready GovRAG answer using "
            "pgvector retrieval, Qwen3 reranking, and a Foundry Local chat model."
        )
    )
    parser.add_argument(
        "question",
        type=str,
        help="Legal AI governance question.",
    )
    parser.add_argument(
        "--candidate-k",
        type=int,
        default=DEFAULT_CANDIDATE_K,
        help="Number of pgvector candidates.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help="Number of evidence passages sent to the chat model.",
    )
    arguments = parser.parse_args()

    try:
        result = generate_rag_answer(
            question=arguments.question,
            candidate_k=arguments.candidate_k,
            top_k=arguments.top_k,
        )

    except (RuntimeError, ValueError) as error:
        parser.exit(status=1, message=f"GovRAG failed: {error}\n")

    print_rag_result(result)


if __name__ == "__main__":
    main()
from __future__ import annotations

import argparse
import json
import math
import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import numpy as np
import psycopg2
import torch
from dotenv import load_dotenv
from sentence_transformers import CrossEncoder


PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")


QUERY_TASK = (
    "Given a legal AI governance question, retrieve relevant passages "
    "from authoritative AI governance and data protection documents."
)
RERANK_TASK = (
    "Given a legal AI governance or data protection question, "
    "determine whether the document passage directly answers the "
    "question. Prioritize authoritative legal provisions and passages "
    "that contain the applicable rule, requirement, condition, "
    "obligation, exception, or definition."
)

def required_env(name: str) -> str:
    value = os.getenv(name)

    if not value:
        raise RuntimeError(
            f"Missing required environment variable: {name}"
        )

    return value


FOUNDRY_BASE_URL = required_env("FOUNDRY_BASE_URL").rstrip("/")
EMBEDDING_MODEL_ID = required_env("EMBEDDING_MODEL_ID")
EMBEDDING_MODEL_VERSION = required_env(
    "EMBEDDING_MODEL_VERSION"
)
EXPECTED_DIMENSION = int(
    required_env("EMBEDDING_DIMENSION")
)

RERANKER_MODEL_ID = required_env("RERANKER_MODEL_ID")
RERANKER_DEVICE = required_env("RERANKER_DEVICE")

DEFAULT_CANDIDATE_K = int(
    required_env("RETRIEVAL_CANDIDATE_K")
)
DEFAULT_TOP_K = int(
    required_env("RERANK_TOP_K")
)


def get_db_connection():
    return psycopg2.connect(
        host=required_env("DB_HOST"),
        port=required_env("DB_PORT"),
        dbname=required_env("DB_NAME"),
        user=required_env("DB_USER"),
        password=required_env("DB_PASSWORD"),
    )


def format_embedding_query(question: str) -> str:
    return (
        f"Instruct: {QUERY_TASK}\n"
        f"Query: {question.strip()}"
    )


def request_query_embedding(question: str) -> list[float]:
    formatted_query = format_embedding_query(question)

    payload = json.dumps(
        {
            "model": EMBEDDING_MODEL_ID,
            "input": [formatted_query],
        }
    ).encode("utf-8")

    request = Request(
        url=f"{FOUNDRY_BASE_URL}/embeddings",
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
        error_body = error.read().decode(
            "utf-8",
            errors="replace",
        )

        raise RuntimeError(
            f"Foundry request failed with HTTP "
            f"{error.code}: {error_body}"
        ) from error

    except URLError as error:
        raise RuntimeError(
            f"Could not reach Foundry Local at "
            f"{FOUNDRY_BASE_URL}."
        ) from error

    response_json = json.loads(response_body)
    embedding_items = response_json.get("data")

    if not isinstance(embedding_items, list):
        raise RuntimeError(
            f"Unexpected Foundry response: {response_json}"
        )

    if len(embedding_items) != 1:
        raise RuntimeError(
            "Expected one query embedding, "
            f"received {len(embedding_items)}."
        )

    embedding = embedding_items[0].get("embedding")

    if not isinstance(embedding, list):
        raise RuntimeError(
            "Foundry did not return a valid embedding."
        )

    if len(embedding) != EXPECTED_DIMENSION:
        raise RuntimeError(
            f"Expected {EXPECTED_DIMENSION} dimensions, "
            f"received {len(embedding)}."
        )

    if not all(
        isinstance(value, (int, float))
        and math.isfinite(value)
        for value in embedding
    ):
        raise RuntimeError(
            "Query embedding contains invalid values."
        )

    return [float(value) for value in embedding]


def vector_literal(vector: list[float]) -> str:
    return json.dumps(vector, separators=(",", ":"))


def retrieve_vector_candidates(
    cursor,
    query_embedding: list[float],
    candidate_k: int,
) -> list[dict]:
    cursor.execute(
        """
        WITH query_vector AS (
            SELECT %s::vector AS embedding
        )
        SELECT
            s.chunk_id,
            s.document_id,
            s.document_name,
            s.chunk_order,
            s.chunk_text,
            s.section_type,
            s.section_reference,
            s.section_title,
            s.page_start,
            s.page_end,
            s.chunking_version,
            g.embedding <=> q.embedding AS cosine_distance,
            1 - (g.embedding <=> q.embedding) AS vector_similarity
        FROM gold.chunk_embeddings AS g
        JOIN silver.document_chunks AS s
            ON s.chunk_id = g.chunk_id
        CROSS JOIN query_vector AS q
        WHERE g.model_id = %s
          AND g.model_version = %s
        ORDER BY cosine_distance ASC
        LIMIT %s;
        """,
        (
            vector_literal(query_embedding),
            EMBEDDING_MODEL_ID,
            EMBEDDING_MODEL_VERSION,
            candidate_k,
        ),
    )

    candidates = []

    for vector_rank, row in enumerate(
        cursor.fetchall(),
        start=1,
    ):
        (
            chunk_id,
            document_id,
            document_name,
            chunk_order,
            chunk_text,
            section_type,
            section_reference,
            section_title,
            page_start,
            page_end,
            chunking_version,
            cosine_distance,
            vector_similarity,
        ) = row

        candidates.append(
            {
                "chunk_id": chunk_id,
                "document_id": document_id,
                "document_name": document_name,
                "chunk_order": chunk_order,
                "chunk_text": chunk_text,
                "section_type": section_type,
                "section_reference": section_reference,
                "section_title": section_title,
                "page_start": page_start,
                "page_end": page_end,
                "chunking_version": chunking_version,
                "cosine_distance": float(cosine_distance),
                "vector_similarity": float(vector_similarity),
                "vector_rank": vector_rank,
            }
        )

    return candidates


def load_reranker() -> CrossEncoder:
    if (
        RERANKER_DEVICE.startswith("cuda")
        and not torch.cuda.is_available()
    ):
        raise RuntimeError(
            "RERANKER_DEVICE is cuda, but CUDA is unavailable."
        )

    print(f"Loading reranker: {RERANKER_MODEL_ID}")

    model = CrossEncoder(
        RERANKER_MODEL_ID,
        device=RERANKER_DEVICE,
        local_files_only=True,
        max_length=1024,
        prompts={
            "legal_retrieval": RERANK_TASK,
        },
        default_prompt_name="legal_retrieval",
        model_kwargs={
            "torch_dtype": torch.float16,
        },
    )

    actual_device = next(model.model.parameters()).device
    print(f"Actual reranker device: {actual_device}")

    if (
        RERANKER_DEVICE.startswith("cuda")
        and actual_device.type != "cuda"
    ):
        raise RuntimeError(
            f"Reranker loaded on {actual_device}, not CUDA."
        )

    return model

def rerank_candidates(
    question: str,
    candidates: list[dict],
    model: CrossEncoder,
    top_k: int,
) -> list[dict]:
    sentence_pairs = [
        (question, candidate["chunk_text"])
        for candidate in candidates
    ]

    scores = model.predict(
        sentence_pairs,
        batch_size=8,
        show_progress_bar=False,
        convert_to_numpy=True,
    )

    score_values = (
        np.asarray(scores)
        .reshape(-1)
        .tolist()
    )

    if len(score_values) != len(candidates):
        raise RuntimeError(
            "Reranker score count does not match "
            "candidate count."
        )

    for candidate, score in zip(
        candidates,
        score_values,
        strict=True,
    ):
        candidate["reranker_score"] = float(score)

    reranked = sorted(
        candidates,
        key=lambda item: item["reranker_score"],
        reverse=True,
    )

    for rerank_rank, candidate in enumerate(
        reranked,
        start=1,
    ):
        candidate["rerank_rank"] = rerank_rank

    return reranked[:top_k]


def format_pages(page_start, page_end) -> str:
    if page_start is None or page_end is None:
        return "unknown"

    if page_start == page_end:
        return str(page_start)

    return f"{page_start}-{page_end}"


def print_results(question: str, results: list[dict]):
    print()
    print(f"Question: {question}")
    print(f"Final reranked chunks: {len(results)}")

    for result in results:
        text_preview = " ".join(
            result["chunk_text"].split()
        )

        if len(text_preview) > 700:
            text_preview = text_preview[:700] + "..."

        print()
        print("=" * 100)
        print(
            f"Final rank {result['rerank_rank']} | "
            f"Original vector rank {result['vector_rank']}"
        )
        print(
            f"Reranker score: "
            f"{result['reranker_score']:.4f}"
        )
        print(
            f"Vector similarity: "
            f"{result['vector_similarity']:.4f}"
        )
        print(
            f"Document: {result['document_name']} "
            f"(document_id={result['document_id']})"
        )
        print(
            f"Chunk: {result['chunk_id']} "
            f"(order={result['chunk_order']})"
        )
        print(
            f"Section type: "
            f"{result['section_type'] or 'unknown'}"
        )
        print(
            f"Section reference: "
            f"{result['section_reference'] or 'unknown'}"
        )
        print(
            f"Section title: "
            f"{result['section_title'] or 'unknown'}"
        )
        print(
            f"Pages: "
            f"{format_pages(result['page_start'], result['page_end'])}"
        )
        print("-" * 100)
        print(text_preview)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Retrieve candidates with pgvector and rerank "
            "them with a Cross-Encoder."
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
        help="Number of final reranked chunks.",
    )

    arguments = parser.parse_args()

    if not arguments.question.strip():
        parser.error("Question cannot be empty.")

    if arguments.candidate_k <= 0:
        parser.error("--candidate-k must be greater than zero.")

    if arguments.top_k <= 0:
        parser.error("--top-k must be greater than zero.")

    if arguments.top_k > arguments.candidate_k:
        parser.error(
            "--top-k cannot be greater than --candidate-k."
        )

    print(f"Embedding model: {EMBEDDING_MODEL_ID}")
    print(
        f"Retrieval plan: pgvector top-{arguments.candidate_k} "
        f"-> reranker top-{arguments.top_k}"
    )
    print("Creating query embedding...")

    query_embedding = request_query_embedding(
        arguments.question
    )

    print(
        f"Query embedding created: "
        f"{len(query_embedding)} dimensions"
    )

    connection = get_db_connection()
    cursor = connection.cursor()

    try:
        print("Retrieving pgvector candidates...")

        candidates = retrieve_vector_candidates(
            cursor=cursor,
            query_embedding=query_embedding,
            candidate_k=arguments.candidate_k,
        )

    finally:
        cursor.close()
        connection.close()

    if not candidates:
        raise RuntimeError(
            "pgvector did not return any candidates."
        )

    print(
        f"Retrieved {len(candidates)} vector candidates."
    )

    reranker = load_reranker()

    print("Reranking candidates...")

    final_results = rerank_candidates(
        question=arguments.question,
        candidates=candidates,
        model=reranker,
        top_k=arguments.top_k,
    )

    print_results(
        question=arguments.question,
        results=final_results,
    )


if __name__ == "__main__":
    main()
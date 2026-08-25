from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import psycopg2
from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")


def required_env(name: str) -> str:
    value = os.getenv(name)

    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")

    return value


FOUNDRY_BASE_URL = required_env("FOUNDRY_BASE_URL").rstrip("/")
MODEL_ALIAS = required_env("EMBEDDING_MODEL_ALIAS")
MODEL_ID = required_env("EMBEDDING_MODEL_ID")
MODEL_VERSION = required_env("EMBEDDING_MODEL_VERSION")
EXPECTED_DIMENSION = int(required_env("EMBEDDING_DIMENSION"))
DEFAULT_BATCH_SIZE = int(required_env("EMBEDDING_BATCH_SIZE"))


def get_db_connection():
    return psycopg2.connect(
        host=required_env("DB_HOST"),
        port=required_env("DB_PORT"),
        dbname=required_env("DB_NAME"),
        user=required_env("DB_USER"),
        password=required_env("DB_PASSWORD"),
    )


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def fetch_silver_chunks(cursor, limit: int | None):
    query = """
        SELECT
            chunk_id,
            chunk_hash,
            chunk_text
        FROM silver.document_chunks
        ORDER BY chunk_id
    """

    parameters = ()

    if limit is not None:
        query += " LIMIT %s"
        parameters = (limit,)

    cursor.execute(query, parameters)
    rows = cursor.fetchall()

    chunks = []

    for chunk_id, chunk_hash, chunk_text in rows:
        if not chunk_hash:
            raise RuntimeError(
                f"Silver chunk {chunk_id} does not have a chunk_hash."
            )

        if not chunk_text or not chunk_text.strip():
            raise RuntimeError(
                f"Silver chunk {chunk_id} has empty chunk_text."
            )

        chunks.append(
            {
                "chunk_id": chunk_id,
                "chunk_hash": chunk_hash,
                "chunk_text": chunk_text,
                "embedding_input_hash": sha256_text(chunk_text),
            }
        )

    return chunks


def fetch_existing_embeddings(cursor):
    cursor.execute(
        """
        SELECT
            chunk_id,
            chunk_hash,
            embedding_input_hash
        FROM gold.chunk_embeddings
        WHERE model_id = %s
          AND model_version = %s;
        """,
        (MODEL_ID, MODEL_VERSION),
    )

    return {
        chunk_id: (chunk_hash, embedding_input_hash)
        for chunk_id, chunk_hash, embedding_input_hash in cursor.fetchall()
    }


def request_embeddings(texts: list[str]) -> list[list[float]]:
    payload = json.dumps(
        {
            "model": MODEL_ID,
            "input": texts,
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
        error_body = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"Foundry embedding request failed with HTTP "
            f"{error.code}: {error_body}"
        ) from error

    except URLError as error:
        raise RuntimeError(
            f"Could not reach Foundry Local at {FOUNDRY_BASE_URL}. "
            "Make sure the Foundry server is running."
        ) from error

    response_json = json.loads(response_body)
    embedding_items = response_json.get("data")

    if not isinstance(embedding_items, list):
        raise RuntimeError(
            f"Unexpected Foundry response: {response_json}"
        )

    embedding_items.sort(key=lambda item: item["index"])

    if len(embedding_items) != len(texts):
        raise RuntimeError(
            "Foundry returned a different number of embeddings. "
            f"Requested: {len(texts)}, received: {len(embedding_items)}"
        )

    embeddings = []

    for expected_index, item in enumerate(embedding_items):
        if item.get("index") != expected_index:
            raise RuntimeError(
                f"Unexpected embedding index: {item.get('index')}"
            )

        embedding = item.get("embedding")

        if not isinstance(embedding, list):
            raise RuntimeError(
                f"Embedding {expected_index} is not a list."
            )

        if len(embedding) != EXPECTED_DIMENSION:
            raise RuntimeError(
                f"Wrong embedding dimension at index {expected_index}. "
                f"Expected {EXPECTED_DIMENSION}, received {len(embedding)}."
            )

        if not all(
            isinstance(value, (int, float)) and math.isfinite(value)
            for value in embedding
        ):
            raise RuntimeError(
                f"Embedding {expected_index} contains invalid numbers."
            )

        embeddings.append([float(value) for value in embedding])

    return embeddings


def vector_literal(vector: list[float]) -> str:
    return json.dumps(vector, separators=(",", ":"))


def split_batches(items: list[dict], batch_size: int):
    for start in range(0, len(items), batch_size):
        yield items[start:start + batch_size]


def save_embedding_batch(cursor, connection, batch, embeddings):
    try:
        for chunk, embedding in zip(batch, embeddings, strict=True):
            # Aynı model ve sürüme ait eski kayıt varsa yenisiyle değiştir.
            cursor.execute(
                """
                DELETE FROM gold.chunk_embeddings
                WHERE chunk_id = %s
                  AND model_id = %s
                  AND model_version = %s;
                """,
                (
                    chunk["chunk_id"],
                    MODEL_ID,
                    MODEL_VERSION,
                ),
            )

            cursor.execute(
                """
                INSERT INTO gold.chunk_embeddings (
                    chunk_id,
                    chunk_hash,
                    embedding_input_hash,
                    model_alias,
                    model_id,
                    model_version,
                    embedding_dimension,
                    embedding,
                    created_at,
                    updated_at
                )
                VALUES (
                    %s, %s, %s, %s, %s,
                    %s, %s, %s::vector,
                    NOW(), NOW()
                );
                """,
                (
                    chunk["chunk_id"],
                    chunk["chunk_hash"],
                    chunk["embedding_input_hash"],
                    MODEL_ALIAS,
                    MODEL_ID,
                    MODEL_VERSION,
                    EXPECTED_DIMENSION,
                    vector_literal(embedding),
                ),
            )

        connection.commit()

    except Exception:
        connection.rollback()
        raise


def main():
    parser = argparse.ArgumentParser(
        description="Create Gold embeddings from Silver document chunks."
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of Silver chunks to inspect.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Number of chunks sent to Foundry in one request.",
    )

    arguments = parser.parse_args()

    if arguments.limit is not None and arguments.limit <= 0:
        parser.error("--limit must be greater than zero.")

    if arguments.batch_size <= 0:
        parser.error("--batch-size must be greater than zero.")

    print(f"Foundry URL:        {FOUNDRY_BASE_URL}")
    print(f"Model alias:        {MODEL_ALIAS}")
    print(f"Model ID:           {MODEL_ID}")
    print(f"Model version:      {MODEL_VERSION}")
    print(f"Expected dimension: {EXPECTED_DIMENSION}")
    print(f"Batch size:         {arguments.batch_size}")

    connection = get_db_connection()
    cursor = connection.cursor()

    try:
        silver_chunks = fetch_silver_chunks(
            cursor=cursor,
            limit=arguments.limit,
        )

        existing_embeddings = fetch_existing_embeddings(cursor)
        connection.commit()

        pending_chunks = []

        for chunk in silver_chunks:
            existing = existing_embeddings.get(chunk["chunk_id"])

            current_hashes = (
                chunk["chunk_hash"],
                chunk["embedding_input_hash"],
            )

            if existing == current_hashes:
                continue

            pending_chunks.append(chunk)

        skipped_count = len(silver_chunks) - len(pending_chunks)

        print(f"Silver chunks read: {len(silver_chunks)}")
        print(f"Unchanged/skipped:  {skipped_count}")
        print(f"Pending embeddings: {len(pending_chunks)}")

        if not pending_chunks:
            print("Gold embeddings are already up to date.")
            return

        saved_count = 0

        for batch_number, batch in enumerate(
            split_batches(pending_chunks, arguments.batch_size),
            start=1,
        ):
            print(
                f"Embedding batch {batch_number}: "
                f"{len(batch)} chunk(s)"
            )

            embeddings = request_embeddings(
                [chunk["chunk_text"] for chunk in batch]
            )

            save_embedding_batch(
                cursor=cursor,
                connection=connection,
                batch=batch,
                embeddings=embeddings,
            )

            saved_count += len(batch)

            print(
                f"Saved {saved_count}/{len(pending_chunks)} "
                "embedding(s)."
            )

        print("Gold embedding generation completed successfully.")

    except Exception:
        connection.rollback()
        print("Gold embedding generation failed.")
        raise

    finally:
        cursor.close()
        connection.close()


if __name__ == "__main__":
    main()
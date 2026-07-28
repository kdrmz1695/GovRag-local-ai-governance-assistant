from pathlib import Path
import hashlib
import os

import psycopg2
import tiktoken
from dotenv import load_dotenv

from legal_structure import (
    AnnotatedLine,
    parse_document_structure,
    summarize_structure,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENV_PATH = PROJECT_ROOT / ".env"

TOKENIZER_NAME = "cl100k_base"
CHUNK_SIZE = 700
CHUNK_OVERLAP = 100
CHUNKING_VERSION = "section-aware-v2-cl100k-700-100"

SKIPPED_SECTION_TYPES = {
    "table_of_contents",
}

EXPECTED_STRUCTURE_COUNTS = {
    "eu_ai_act_2024_1689": {
        "recital": 180,
        "article": 113,
        "annex": 13,
    },
    "gdpr_2016_679": {
        "recital": 173,
        "article": 99,
    },
    "edpb_opinion_202428_ai-models_en": {
        "opinion_section": 20,
    },
}

REQUIRED_ENV_VARIABLES = (
    "DB_HOST",
    "DB_PORT",
    "DB_NAME",
    "DB_USER",
    "DB_PASSWORD",
)

load_dotenv(ENV_PATH)


def get_db_connection():
    """
    Create and return a PostgreSQL connection.
    """
    missing_variables = [
        variable
        for variable in REQUIRED_ENV_VARIABLES
        if not os.getenv(variable)
    ]

    if missing_variables:
        missing_names = ", ".join(missing_variables)
        raise RuntimeError(
            f"Missing required environment variables: {missing_names}"
        )

    return psycopg2.connect(
        host=os.getenv("DB_HOST"),
        port=os.getenv("DB_PORT"),
        dbname=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
    )


def calculate_chunk_hash(chunk_text: str) -> str:
    """
    Calculate a deterministic SHA-256 hash for a chunk.
    """
    return hashlib.sha256(
        chunk_text.encode("utf-8")
    ).hexdigest()


def get_bronze_documents(cursor):
    """
    Read document-level metadata from Bronze.
    """
    cursor.execute(
        """
        SELECT
            document_id,
            document_name,
            page_count
        FROM bronze.raw_documents
        ORDER BY document_id;
        """
    )

    return cursor.fetchall()


def get_document_pages(cursor, document_id: int):
    """
    Read one document's pages in their original order.
    """
    cursor.execute(
        """
        SELECT
            page_number,
            page_text
        FROM bronze.document_pages
        WHERE document_id = %s
        ORDER BY page_number;
        """,
        (document_id,),
    )

    return cursor.fetchall()


def validate_detected_structure(
    document_name: str,
    annotated_lines: list[AnnotatedLine],
):
    """
    Stop processing when a known V1 document's legal structure
    does not match its expected reference counts.
    """
    expected_counts = EXPECTED_STRUCTURE_COUNTS.get(document_name)

    if not expected_counts:
        return

    structure_summary = summarize_structure(annotated_lines)

    for section_type, expected_count in expected_counts.items():
        actual_count = len(
            structure_summary.get(section_type, set())
        )

        if actual_count != expected_count:
            raise ValueError(
                f"Structure validation failed for {document_name}: "
                f"expected {expected_count} {section_type} references, "
                f"found {actual_count}"
            )


def print_structure_summary(
    document_name: str,
    annotated_lines: list[AnnotatedLine],
):
    """
    Print the number of unique references detected for each type.
    """
    structure_summary = summarize_structure(annotated_lines)

    printable_counts = {
        section_type: len(references)
        for section_type, references in structure_summary.items()
    }

    print(
        f"Detected structure for {document_name}: "
        f"{printable_counts}"
    )


def create_section_groups(
    annotated_lines: list[AnnotatedLine],
) -> list[dict]:
    """
    Group consecutive lines that belong to the same legal section.

    A group can represent an Article, Recital, Annex, EDPB section,
    chapter heading, or another document-level structural unit.
    """
    groups = []

    for line in annotated_lines:
        if line.section_type in SKIPPED_SECTION_TYPES:
            continue

        metadata_key = (
            line.section_type,
            line.section_reference,
            line.section_title,
        )

        if (
            not groups
            or groups[-1]["metadata_key"] != metadata_key
        ):
            groups.append(
                {
                    "metadata_key": metadata_key,
                    "section_type": line.section_type,
                    "section_reference": line.section_reference,
                    "section_title": line.section_title,
                    "parts": [],
                }
            )

        groups[-1]["parts"].append(
            {
                "page_number": line.page_number,
                "text": line.text,
            }
        )

    return groups


def tokenize_section_group(section_group: dict, encoding):
    """
    Tokenize one structural group while preserving token-to-page
    provenance.
    """
    tokens = []
    token_page_numbers = []

    for part in section_group["parts"]:
        text_with_separator = part["text"] + "\n"
        part_tokens = encoding.encode(text_with_separator)

        tokens.extend(part_tokens)
        token_page_numbers.extend(
            [part["page_number"]] * len(part_tokens)
        )

    return tokens, token_page_numbers


def create_chunks_for_section(
    section_group: dict,
    encoding,
    chunk_size: int,
    overlap: int,
) -> list[dict]:
    """
    Create chunks inside one legal structure boundary.

    Chunks can overlap inside one section but never cross from one
    Article, Recital, Annex, or opinion section into another.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero.")

    if overlap < 0:
        raise ValueError("overlap cannot be negative.")

    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size.")

    tokens, token_page_numbers = tokenize_section_group(
        section_group,
        encoding,
    )

    if not tokens:
        return []

    chunks = []
    start = 0

    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))

        chunk_tokens = tokens[start:end]
        chunk_pages = token_page_numbers[start:end]
        chunk_text = encoding.decode(chunk_tokens).strip()

        if chunk_text:
            chunks.append(
                {
                    "chunk_text": chunk_text,
                    "chunk_char_length": len(chunk_text),
                    "chunk_token_count": len(chunk_tokens),
                    "page_start": chunk_pages[0],
                    "page_end": chunk_pages[-1],
                    "section_type": section_group["section_type"],
                    "section_reference": (
                        section_group["section_reference"]
                    ),
                    "section_title": section_group["section_title"],
                    "chunk_hash": calculate_chunk_hash(chunk_text),
                }
            )

        if end == len(tokens):
            break

        start = end - overlap

    return chunks


def create_section_aware_chunks(
    annotated_lines: list[AnnotatedLine],
    encoding,
    chunk_size: int,
    overlap: int,
) -> list[dict]:
    """
    Create chunks for all detected structural groups in document order.
    """
    section_groups = create_section_groups(annotated_lines)
    chunks = []

    for section_group in section_groups:
        group_chunks = create_chunks_for_section(
            section_group=section_group,
            encoding=encoding,
            chunk_size=chunk_size,
            overlap=overlap,
        )

        chunks.extend(group_chunks)

    return chunks


def insert_document_chunks(
    cursor,
    document_id: int,
    document_name: str,
    chunks: list[dict],
):
    """
    Replace one document's previous chunks with section-aware chunks.
    """
    cursor.execute(
        """
        DELETE FROM silver.document_chunks
        WHERE document_id = %s;
        """,
        (document_id,),
    )

    for chunk_order, chunk in enumerate(chunks, start=1):
        cursor.execute(
            """
            INSERT INTO silver.document_chunks (
                document_id,
                document_name,
                chunk_order,
                chunk_text,
                chunk_char_length,
                chunk_token_count,
                page_start,
                page_end,
                section_type,
                section_reference,
                section_title,
                chunk_hash,
                chunking_version
            )
            VALUES (
                %s, %s, %s, %s, %s, %s, %s,
                %s, %s, %s, %s, %s, %s
            );
            """,
            (
                document_id,
                document_name,
                chunk_order,
                chunk["chunk_text"],
                chunk["chunk_char_length"],
                chunk["chunk_token_count"],
                chunk["page_start"],
                chunk["page_end"],
                chunk["section_type"],
                chunk["section_reference"],
                chunk["section_title"],
                chunk["chunk_hash"],
                CHUNKING_VERSION,
            ),
        )


def main():
    encoding = tiktoken.get_encoding(TOKENIZER_NAME)

    connection = get_db_connection()
    cursor = connection.cursor()

    try:
        documents = get_bronze_documents(cursor)

        print(f"Found {len(documents)} document(s) in Bronze.")

        for document_id, document_name, expected_page_count in documents:
            print("-" * 80)
            print(
                f"Creating section-aware chunks for: "
                f"{document_name}"
            )

            pages = get_document_pages(
                cursor=cursor,
                document_id=document_id,
            )

            if len(pages) != expected_page_count:
                raise ValueError(
                    f"Page count mismatch for {document_name}: "
                    f"expected {expected_page_count}, "
                    f"found {len(pages)}"
                )

            annotated_lines = parse_document_structure(
                document_name=document_name,
                pages=pages,
            )

            if not annotated_lines:
                raise ValueError(
                    f"No structured text was produced for: "
                    f"{document_name}"
                )

            validate_detected_structure(
                document_name=document_name,
                annotated_lines=annotated_lines,
            )

            print_structure_summary(
                document_name=document_name,
                annotated_lines=annotated_lines,
            )

            chunks = create_section_aware_chunks(
                annotated_lines=annotated_lines,
                encoding=encoding,
                chunk_size=CHUNK_SIZE,
                overlap=CHUNK_OVERLAP,
            )

            if not chunks:
                raise ValueError(
                    f"No chunks were created for: {document_name}"
                )

            insert_document_chunks(
                cursor=cursor,
                document_id=document_id,
                document_name=document_name,
                chunks=chunks,
            )

            print(
                f"Prepared {len(chunks)} section-aware chunks "
                f"for {document_name}"
            )

        connection.commit()

        print("=" * 80)
        print("Silver section-aware chunking completed.")

    except Exception:
        connection.rollback()
        print("Silver chunking failed. Transaction rolled back.")
        raise

    finally:
        cursor.close()
        connection.close()


if __name__ == "__main__":
    main()
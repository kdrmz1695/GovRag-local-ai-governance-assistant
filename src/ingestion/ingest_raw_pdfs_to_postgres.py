from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import sys

import fitz
import psycopg2
from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from corpus_manifest import (
    DEFAULT_MANIFEST_PATH,
    DEFAULT_RAW_DIR,
    CorpusDocument,
    ManifestValidationError,
    load_corpus_manifest,
    select_documents,
    validate_corpus_files,
    validate_selected_files,
)


RAW_DIR = DEFAULT_RAW_DIR
ENV_PATH = PROJECT_ROOT / ".env"

REQUIRED_ENV_VARIABLES = (
    "DB_HOST",
    "DB_PORT",
    "DB_NAME",
    "DB_USER",
    "DB_PASSWORD",
)

load_dotenv(ENV_PATH)


def get_db_connection():
    missing_variables = [
        variable
        for variable in REQUIRED_ENV_VARIABLES
        if not os.getenv(variable)
    ]
    if missing_variables:
        raise RuntimeError(
            "Missing required environment variables: "
            + ", ".join(missing_variables)
        )

    return psycopg2.connect(
        host=os.getenv("DB_HOST"),
        port=os.getenv("DB_PORT"),
        dbname=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
    )


def calculate_file_hash(file_path: Path) -> str:
    sha256 = hashlib.sha256()
    with file_path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            sha256.update(block)
    return sha256.hexdigest()


def extract_pdf_pages(pdf_path: Path):
    pages = []
    with fitz.open(pdf_path) as pdf:
        for page_index, page in enumerate(pdf):
            page_text = page.get_text("text")
            pages.append(
                {
                    "page_number": page_index + 1,
                    "page_text": page_text,
                    "page_char_length": len(page_text),
                }
            )

    full_text = "".join(page["page_text"] + "\n\n" for page in pages)
    return full_text, pages


def upsert_raw_document(
    cursor,
    document: CorpusDocument,
    file_hash: str,
    full_text: str,
    page_count: int,
) -> int:
    """Upsert one stable manifest identity without duplicating a revision."""
    cursor.execute(
        """
        SELECT document_id, file_hash
        FROM bronze.raw_documents
        WHERE document_name = %s
        ORDER BY document_id
        FOR UPDATE;
        """,
        (document.document_id,),
    )
    identity_rows = cursor.fetchall()
    if len(identity_rows) > 1:
        raise RuntimeError(
            f"Bronze contains duplicate logical document identity "
            f"{document.document_id!r}. Resolve it manually before ingestion."
        )

    cursor.execute(
        """
        SELECT document_id, document_name
        FROM bronze.raw_documents
        WHERE file_hash = %s
        FOR UPDATE;
        """,
        (file_hash,),
    )
    hash_row = cursor.fetchone()

    if identity_rows:
        database_document_id = identity_rows[0][0]
        if hash_row is not None and hash_row[0] != database_document_id:
            raise RuntimeError(
                f"PDF hash for {document.document_id!r} is already assigned to "
                f"Bronze document {hash_row[1]!r}."
            )

        cursor.execute(
            """
            UPDATE bronze.raw_documents
            SET source_name = %s,
                source_url = %s,
                file_name = %s,
                file_type = 'PDF',
                file_hash = %s,
                full_text = %s,
                page_count = %s,
                loaded_at = CURRENT_TIMESTAMP
            WHERE document_id = %s
            RETURNING document_id;
            """,
            (
                document.source_repository,
                document.source_url,
                document.file_name,
                file_hash,
                full_text,
                page_count,
                database_document_id,
            ),
        )
        return cursor.fetchone()[0]

    if hash_row is not None:
        raise RuntimeError(
            f"PDF hash for {document.document_id!r} is already assigned to "
            f"Bronze document {hash_row[1]!r}."
        )

    cursor.execute(
        """
        INSERT INTO bronze.raw_documents (
            document_name, source_name, source_url, file_name,
            file_type, file_hash, full_text, page_count
        )
        VALUES (%s, %s, %s, %s, 'PDF', %s, %s, %s)
        RETURNING document_id;
        """,
        (
            document.document_id,
            document.source_repository,
            document.source_url,
            document.file_name,
            file_hash,
            full_text,
            page_count,
        ),
    )
    return cursor.fetchone()[0]


def upsert_document_pages(cursor, database_document_id: int, pages: list):
    for page in pages:
        cursor.execute(
            """
            INSERT INTO bronze.document_pages (
                document_id, page_number, page_text, page_char_length
            )
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (document_id, page_number)
            DO UPDATE SET
                page_text = EXCLUDED.page_text,
                page_char_length = EXCLUDED.page_char_length,
                loaded_at = CURRENT_TIMESTAMP;
            """,
            (
                database_document_id,
                page["page_number"],
                page["page_text"],
                page["page_char_length"],
            ),
        )

    cursor.execute(
        """
        DELETE FROM bronze.document_pages
        WHERE document_id = %s AND page_number > %s;
        """,
        (database_document_id, len(pages)),
    )


def prepare_document(document: CorpusDocument, raw_dir: Path = RAW_DIR) -> dict:
    pdf_path = raw_dir / document.file_name
    file_hash = calculate_file_hash(pdf_path)
    full_text, pages = extract_pdf_pages(pdf_path)
    if not pages:
        raise ValueError(f"No pages were extracted from: {pdf_path}")
    if not full_text.strip():
        raise ValueError(f"No text was extracted from: {pdf_path}")
    return {
        "document": document,
        "pdf_path": pdf_path,
        "file_hash": file_hash,
        "full_text": full_text,
        "pages": pages,
    }


def run_ingestion(
    selected: list[CorpusDocument],
    *,
    raw_dir: Path = RAW_DIR,
    dry_run: bool = False,
    connection_factory=get_db_connection,
) -> list[dict]:
    prepared = []
    for document in selected:
        item = prepare_document(document, raw_dir)
        prepared.append(item)
        print(
            f"Prepared {document.document_id}: {document.file_name} | "
            f"SHA-256={item['file_hash']} | pages={len(item['pages'])}"
        )

    if dry_run:
        print("Dry run complete. No database connection or write was attempted.")
        return prepared

    connection = connection_factory()
    cursor = connection.cursor()
    try:
        for item in prepared:
            document = item["document"]
            database_document_id = upsert_raw_document(
                cursor,
                document,
                item["file_hash"],
                item["full_text"],
                len(item["pages"]),
            )
            upsert_document_pages(cursor, database_document_id, item["pages"])
            print(
                f"Staged {document.document_id} | "
                f"Bronze document_id={database_document_id}"
            )
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        cursor.close()
        connection.close()

    print(f"Bronze ingestion committed for {len(prepared)} selected document(s).")
    return prepared


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Ingest explicitly selected manifest PDFs into Bronze."
    )
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument(
        "--document-id",
        action="append",
        dest="document_ids",
        help="Stable manifest document_id. Repeat to select more than one.",
    )
    selection.add_argument(
        "--all",
        action="store_true",
        help="Process all enabled manifest documents.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and extract selected PDFs without connecting to PostgreSQL.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST_PATH,
        help="Corpus manifest path.",
    )
    return parser


def main() -> None:
    parser = build_argument_parser()
    arguments = parser.parse_args()
    try:
        manifest = load_corpus_manifest(arguments.manifest)
        selected = select_documents(
            manifest,
            arguments.document_ids,
            arguments.all,
        )
        validate_selected_files(selected, RAW_DIR)
        _, unknown_files = validate_corpus_files(manifest, RAW_DIR)
        if unknown_files:
            raise ManifestValidationError(
                "Unknown PDF file(s) are present in data/raw but absent from the "
                f"manifest: {', '.join(unknown_files)}"
            )
        run_ingestion(selected, dry_run=arguments.dry_run)
    except (ManifestValidationError, RuntimeError, ValueError) as error:
        parser.exit(status=2, message=f"Bronze ingestion validation failed: {error}\n")


if __name__ == "__main__":
    main()

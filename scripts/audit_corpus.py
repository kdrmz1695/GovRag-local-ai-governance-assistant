from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import sys

import fitz
from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from corpus_manifest import (
    DEFAULT_MANIFEST_PATH,
    DEFAULT_RAW_DIR,
    MANUAL_REVIEW_MARKER,
    CorpusDocument,
    ManifestValidationError,
    load_corpus_manifest,
    validate_corpus_files,
)
from processing.legal_structure import SUPPORTED_PARSER_TYPES


load_dotenv(PROJECT_ROOT / ".env")


def get_db_connection():
    import psycopg2

    required = ("DB_HOST", "DB_PORT", "DB_NAME", "DB_USER", "DB_PASSWORD")
    missing = [name for name in required if not os.getenv(name)]
    if missing:
        raise RuntimeError(
            "Missing database environment variables: " + ", ".join(missing)
        )
    return psycopg2.connect(
        host=os.getenv("DB_HOST"),
        port=os.getenv("DB_PORT"),
        dbname=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def inspect_pdf(document: CorpusDocument, raw_dir: Path) -> dict:
    path = raw_dir / document.file_name
    result = {
        "pdf_exists": path.is_file(),
        "sha256": None,
        "page_count": None,
        "total_extracted_characters": None,
        "empty_page_count": None,
    }
    if not result["pdf_exists"]:
        return result

    result["sha256"] = sha256_file(path)
    with fitz.open(path) as pdf:
        page_texts = [page.get_text("text") for page in pdf]
    result["page_count"] = len(page_texts)
    result["total_extracted_characters"] = sum(len(text) for text in page_texts)
    result["empty_page_count"] = sum(not text.strip() for text in page_texts)
    return result


def read_database_inventory(connection_factory, document_ids: list[str]) -> dict:
    connection = connection_factory()
    cursor = None
    try:
        connection.set_session(readonly=True, autocommit=False)
        cursor = connection.cursor()
        cursor.execute(
            """
            SELECT
                b.document_id,
                b.document_name,
                b.file_hash,
                b.page_count,
                (SELECT COUNT(*) FROM bronze.document_pages AS p
                 WHERE p.document_id = b.document_id) AS bronze_page_rows,
                (SELECT COUNT(*) FROM silver.document_chunks AS s
                 WHERE s.document_id = b.document_id) AS silver_chunk_count,
                (SELECT COUNT(*) FROM gold.chunk_embeddings AS g
                 JOIN silver.document_chunks AS s ON s.chunk_id = g.chunk_id
                 WHERE s.document_id = b.document_id) AS gold_embedding_count
            FROM bronze.raw_documents AS b
            WHERE b.document_name = ANY(%s)
            ORDER BY b.document_name, b.document_id;
            """,
            (document_ids,),
        )
        rows = cursor.fetchall()
    finally:
        if cursor is not None:
            cursor.close()
        connection.close()

    inventory: dict[str, list[dict]] = {}
    keys = (
        "database_document_id",
        "document_id",
        "bronze_sha256",
        "bronze_page_count",
        "bronze_page_rows",
        "silver_chunk_count",
        "gold_embedding_count",
    )
    for row in rows:
        record = dict(zip(keys, row, strict=True))
        inventory.setdefault(record["document_id"], []).append(record)
    return inventory


def build_audit_report(
    manifest,
    raw_dir: Path,
    database_inventory: dict[str, list[dict]] | None,
    database_error: str | None = None,
) -> dict:
    missing_files, unknown_files = validate_corpus_files(manifest, raw_dir)
    report = {
        "manifest": {
            "schema_version": manifest.schema_version,
            "corpus_id": manifest.corpus_id,
            "corpus_version": manifest.corpus_version,
            "path": str(DEFAULT_MANIFEST_PATH),
        },
        "raw_directory": str(raw_dir),
        "unknown_pdf_files": unknown_files,
        "database_checked": database_inventory is not None,
        "database_error": database_error,
        "documents": [],
        "warnings": [],
        "validation_failures": [],
    }

    if unknown_files:
        report["validation_failures"].append(
            "Unknown PDFs are present outside the corpus manifest: "
            + ", ".join(unknown_files)
        )
    if database_error:
        report["warnings"].append(
            "Database inventory is unavailable; file and manifest checks still ran."
        )

    for document in manifest.documents:
        pdf = inspect_pdf(document, raw_dir)
        database_checked = database_inventory is not None
        rows = (database_inventory or {}).get(document.document_id, [])
        database = rows[0] if len(rows) == 1 else None
        warnings = []
        failures = []

        if document.file_name in missing_files:
            failures.append("Manifest PDF is missing from data/raw.")
        if document.needs_manual_review:
            warnings.append(
                f"One or more metadata values contain {MANUAL_REVIEW_MARKER}."
            )
        if document.parser_type not in SUPPORTED_PARSER_TYPES:
            message = f"Parser is not implemented: {document.parser_type}."
            (failures if document.enabled else warnings).append(message)
        if database_checked and len(rows) > 1:
            failures.append("Duplicate stable document identity exists in Bronze.")
        if database_checked and database is None and not rows:
            warnings.append("Document is not present in Bronze.")
        if database is not None:
            if pdf["sha256"] and database["bronze_sha256"] != pdf["sha256"]:
                failures.append("Bronze SHA-256 does not match the manifest PDF.")
            if database["bronze_page_count"] != database["bronze_page_rows"]:
                failures.append(
                    "Bronze raw_documents.page_count does not match page rows."
                )
            if pdf["page_count"] is not None and (
                database["bronze_page_count"] != pdf["page_count"]
            ):
                failures.append("Bronze page count does not match the PDF.")
            if document.enabled and database["silver_chunk_count"] == 0:
                failures.append("Enabled Bronze document has no Silver chunks.")
            if (
                database["gold_embedding_count"]
                < database["silver_chunk_count"]
            ):
                warnings.append(
                    "Gold embedding count is lower than Silver chunk count."
                )
        if pdf["empty_page_count"]:
            warnings.append(
                f"PDF has {pdf['empty_page_count']} page(s) with no extracted text."
            )

        record = {
            "manifest_entry": asdict(document),
            **pdf,
            "selected_parser_type": document.parser_type,
            "present_in_bronze": bool(rows) if database_checked else None,
            "bronze_page_count": database["bronze_page_count"] if database else None,
            "silver_chunk_count": database["silver_chunk_count"] if database else None,
            "gold_embedding_count": database["gold_embedding_count"] if database else None,
            "warnings": warnings,
            "validation_failures": failures,
        }
        report["documents"].append(record)
        report["warnings"].extend(
            f"{document.document_id}: {message}" for message in warnings
        )
        report["validation_failures"].extend(
            f"{document.document_id}: {message}" for message in failures
        )

    return report


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Read-only audit of the GovRAG corpus manifest, PDFs, and DB layers."
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument(
        "--no-database",
        action="store_true",
        help="Run manifest and PDF checks without attempting a read-only DB connection.",
    )
    return parser


def main() -> int:
    arguments = build_argument_parser().parse_args()
    try:
        manifest = load_corpus_manifest(arguments.manifest)
    except ManifestValidationError as error:
        print(json.dumps({"validation_failures": [str(error)]}, indent=2))
        return 2

    database_inventory = None
    database_error = None
    if not arguments.no_database:
        try:
            database_inventory = read_database_inventory(
                get_db_connection,
                [document.document_id for document in manifest.documents],
            )
        except Exception as error:
            database_error = f"{type(error).__name__}: {error}"

    report = build_audit_report(
        manifest,
        arguments.raw_dir,
        database_inventory,
        database_error,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    return 1 if report["validation_failures"] else 0


if __name__ == "__main__":
    raise SystemExit(main())

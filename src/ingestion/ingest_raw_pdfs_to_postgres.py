from pathlib import Path
import hashlib
import os

import fitz
import psycopg2
from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
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
    """
    Create and return a PostgreSQL connection.

    Raises an error if a required environment variable is missing.
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


def calculate_file_hash(file_path: Path) -> str:
    """
    Calculate the SHA-256 hash without loading the whole file into memory.
    """
    sha256 = hashlib.sha256()

    with file_path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            sha256.update(block)

    return sha256.hexdigest()


def extract_pdf_pages(pdf_path: Path):
    """
    Extract text page by page while preserving the original page number.

    Returns:
        full_text: Complete document text.
        pages: Page-level text and metadata.
    """
    pages = []

    with fitz.open(pdf_path) as pdf:
        for page_index, page in enumerate(pdf):
            page_text = page.get_text("text")
            page_number = page_index + 1

            pages.append(
                {
                    "page_number": page_number,
                    "page_text": page_text,
                    "page_char_length": len(page_text),
                }
            )

    full_text = "".join(
        page["page_text"] + "\n\n"
        for page in pages
    )

    return full_text, pages


def upsert_raw_document(
    cursor,
    pdf_path: Path,
    file_hash: str,
    full_text: str,
    page_count: int,
) -> int:
    """
    Insert a new document or update an existing document.

    Existing source_name and source_url values are deliberately preserved.
    """
    cursor.execute(
        """
        INSERT INTO bronze.raw_documents (
            document_name,
            source_name,
            source_url,
            file_name,
            file_type,
            file_hash,
            full_text,
            page_count
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)

        ON CONFLICT (file_hash)
        DO UPDATE SET
            document_name = EXCLUDED.document_name,
            file_name = EXCLUDED.file_name,
            file_type = EXCLUDED.file_type,
            full_text = EXCLUDED.full_text,
            page_count = EXCLUDED.page_count

        RETURNING document_id;
        """,
        (
            pdf_path.stem,
            None,
            None,
            pdf_path.name,
            "PDF",
            file_hash,
            full_text,
            page_count,
        ),
    )

    return cursor.fetchone()[0]


def upsert_document_pages(cursor, document_id: int, pages: list):
    """
    Insert or update every page belonging to a document.
    """
    for page in pages:
        cursor.execute(
            """
            INSERT INTO bronze.document_pages (
                document_id,
                page_number,
                page_text,
                page_char_length
            )
            VALUES (%s, %s, %s, %s)

            ON CONFLICT (document_id, page_number)
            DO UPDATE SET
                page_text = EXCLUDED.page_text,
                page_char_length = EXCLUDED.page_char_length,
                loaded_at = CURRENT_TIMESTAMP;
            """,
            (
                document_id,
                page["page_number"],
                page["page_text"],
                page["page_char_length"],
            ),
        )

    cursor.execute(
        """
        DELETE FROM bronze.document_pages
        WHERE document_id = %s
          AND page_number > %s;
        """,
        (document_id, len(pages)),
    )


def main():
    pdf_paths = sorted(RAW_DIR.glob("*.pdf"))

    if not pdf_paths:
        print(f"No PDF files found in: {RAW_DIR}")
        return

    connection = get_db_connection()
    cursor = connection.cursor()

    try:
        for pdf_path in pdf_paths:
            print("-" * 80)
            print(f"Processing: {pdf_path.name}")

            file_hash = calculate_file_hash(pdf_path)

            full_text, pages = extract_pdf_pages(pdf_path)

            if not pages:
                raise ValueError(
                    f"No pages were extracted from: {pdf_path.name}"
                )

            document_id = upsert_raw_document(
                cursor=cursor,
                pdf_path=pdf_path,
                file_hash=file_hash,
                full_text=full_text,
                page_count=len(pages),
            )

            upsert_document_pages(
                cursor=cursor,
                document_id=document_id,
                pages=pages,
            )

            connection.commit()

            print(
                f"Done: {pdf_path.name} | "
                f"Document ID: {document_id} | "
                f"Pages: {len(pages)}"
            )

        print("=" * 80)
        print("Bronze document and page ingestion finished.")

    except Exception:
        connection.rollback()
        print("Bronze ingestion failed. Transaction rolled back.")
        raise

    finally:
        cursor.close()
        connection.close()


if __name__ == "__main__":
    main()
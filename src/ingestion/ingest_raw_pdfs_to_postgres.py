from pathlib import Path
import os
import hashlib

import fitz
import psycopg2
from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"

load_dotenv(PROJECT_ROOT / ".env")


conn = psycopg2.connect(
    host=os.getenv("DB_HOST"),
    port=os.getenv("DB_PORT"),
    dbname=os.getenv("DB_NAME"),
    user=os.getenv("DB_USER"),
    password=os.getenv("DB_PASSWORD"),
)

cursor = conn.cursor()


for pdf_path in RAW_DIR.glob("*.pdf"):
    print(f"Processing: {pdf_path.name}")

    with open(pdf_path, "rb") as f:
        file_hash = hashlib.sha256(f.read()).hexdigest()

    pdf = fitz.open(pdf_path)
    full_text = ""

    for page in pdf:
        full_text += page.get_text("text") + "\n\n"

    page_count = pdf.page_count
    pdf.close()

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
        ON CONFLICT (file_hash) DO NOTHING;
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

    conn.commit()
    print(f"Done: {pdf_path.name} | Pages: {page_count}")


cursor.close()
conn.close()

print("Bronze ingestion finished.")
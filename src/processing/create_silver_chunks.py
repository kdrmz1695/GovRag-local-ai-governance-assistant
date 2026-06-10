from pathlib import Path
import os
import psycopg2
import tiktoken
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENV_PATH = PROJECT_ROOT / ".env"

load_dotenv(ENV_PATH)

CHUNK_SIZE = 700
CHUNK_OVERLAP = 100

def get_db_connection():
    return psycopg2.connect(
        host=os.getenv("DB_HOST"),
        port=os.getenv("DB_PORT"),
        dbname=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
    )


def create_token_chunks(text: str, encoding, chunk_size: int, overlap: int):
    tokens = encoding.encode(text)

    chunks = []
    step = chunk_size - overlap

    for start in range(0, len(tokens), step):
        end = start + chunk_size
        chunk_tokens = tokens[start:end]

        if not chunk_tokens:
            continue

        chunk_text = encoding.decode(chunk_tokens)

        chunks.append({
            "chunk_text": chunk_text,
            "chunk_token_count": len(chunk_tokens),
            "chunk_char_length": len(chunk_text),
        })

    return chunks


def main():
    encoding = tiktoken.get_encoding("cl100k_base")

    connection = get_db_connection()
    cursor = connection.cursor()

    try:
        cursor.execute("""
            SELECT document_id, document_name, full_text
            FROM bronze.raw_documents
            ORDER BY document_id;
        """)

        documents = cursor.fetchall()

        print(f"Found {len(documents)} document(s) in Bronze.")

        for document_id, document_name, full_text in documents:
            print("-" * 80)
            print(f"Creating chunks for: {document_name}")

            chunks = create_token_chunks(
                text=full_text,
                encoding=encoding,
                chunk_size=CHUNK_SIZE,
                overlap=CHUNK_OVERLAP,
            )

            cursor.execute(
                """
                DELETE FROM silver.document_chunks
                WHERE document_id = %s;
                """,
                (document_id,)
            )

            for index, chunk in enumerate(chunks, start=1):
                cursor.execute(
                    """
                    INSERT INTO silver.document_chunks (
                        document_id,
                        document_name,
                        chunk_order,
                        chunk_text,
                        chunk_char_length,
                        chunk_token_count,
                        section_type,
                        section_reference
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s);
                    """,
                    (
                        document_id,
                        document_name,
                        index,
                        chunk["chunk_text"],
                        chunk["chunk_char_length"],
                        chunk["chunk_token_count"],
                        None,
                        None,
                    )
                )

            connection.commit()

            print(f"Inserted {len(chunks)} chunks for {document_name}")

        print("=" * 80)
        print("Silver chunking completed.")

    except Exception as error:
        connection.rollback()
        print("Error occurred during Silver chunking.")
        print(error)

    finally:
        cursor.close()
        connection.close()


if __name__ == "__main__":
    main()
-- ============================================================
-- Silver Layer Table Creation
-- Table: silver.document_chunks
-- Project: GovRAG - Local AI Governance Assistant
-- Purpose:
--   Store page-aware, token-based chunks with citation and
--   processing lineage metadata.
--
-- Grain:
--   1 row = 1 citation-ready text chunk
--
-- Notes:
--   - Source documents are stored in bronze.raw_documents.
--   - Page text is stored in bronze.document_pages.
--   - Embeddings are created later in the Gold layer.
-- ============================================================


CREATE TABLE IF NOT EXISTS silver.document_chunks (
    chunk_id SERIAL PRIMARY KEY,

    document_id INTEGER NOT NULL
        REFERENCES bronze.raw_documents(document_id),

    document_name TEXT NOT NULL,

    chunk_order INTEGER NOT NULL,

    chunk_text TEXT NOT NULL,

    chunk_char_length INTEGER,

    chunk_token_count INTEGER,

    page_start INTEGER,

    page_end INTEGER,

    section_type TEXT,

    section_reference TEXT,

    section_title TEXT,

    chunk_hash TEXT,

    chunking_version TEXT,

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,


    CONSTRAINT document_chunks_document_order_key
        UNIQUE (document_id, chunk_order),


    CONSTRAINT document_chunks_page_range_check
        CHECK (
            (
                page_start IS NULL
                AND page_end IS NULL
            )
            OR
            (
                page_start IS NOT NULL
                AND page_end IS NOT NULL
                AND page_start > 0
                AND page_end >= page_start
            )
        )
);


CREATE INDEX IF NOT EXISTS idx_document_chunks_section_reference
    ON silver.document_chunks (section_reference);
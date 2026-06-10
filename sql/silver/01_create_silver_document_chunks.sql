-- ============================================================
-- Silver Layer Table Creation
-- Table: silver.document_chunks
-- Project: GovRAG - Local AI Governance Assistant
-- Purpose:
--   Store token-based text chunks created from Bronze raw documents.
--
-- Grain:
--   1 row = 1 text chunk
--
-- Notes:
--   - Source documents are stored in bronze.raw_documents.
--   - This table stores retrieval-ready text chunks.
--   - Embeddings are created later in the Gold layer.
-- ============================================================

CREATE TABLE IF NOT EXISTS silver.document_chunks (
    chunk_id SERIAL PRIMARY KEY,

    document_id INTEGER NOT NULL REFERENCES bronze.raw_documents(document_id),

    document_name TEXT NOT NULL,

    chunk_order INTEGER NOT NULL,

    chunk_text TEXT NOT NULL,

    chunk_char_length INTEGER,

    chunk_token_count INTEGER,

    section_type TEXT,

    section_reference TEXT,

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
-- ============================================================
-- Bronze Layer Table Creation
-- Table: bronze.raw_documents
-- Project: GovRAG - Local AI Governance Assistant
-- Purpose:
--   Store raw extracted governance/legal documents.
--
-- Grain:
--   1 row = 1 source document
--
-- Notes:
--   - Each PDF/DOCX/TXT document is stored as full text.
--   - Chunking, article parsing, and embeddings are handled
--     in later Silver and Gold layers.
--   - file_hash is used to prevent duplicate ingestion.
-- ============================================================

CREATE TABLE IF NOT EXISTS bronze.raw_documents (
    document_id SERIAL PRIMARY KEY,

    document_name TEXT NOT NULL,

    source_name TEXT,

    source_url TEXT,

    file_name TEXT NOT NULL,

    file_type TEXT NOT NULL,

    file_hash TEXT UNIQUE,

    full_text TEXT NOT NULL,

    page_count INTEGER,

    loaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
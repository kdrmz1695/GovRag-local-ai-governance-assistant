-- ============================================================
-- Bronze Layer Page Table Creation
-- Table: bronze.document_pages
-- Project: GovRAG - Local AI Governance Assistant
-- Purpose:
--   Preserve the extracted text and source page number for every
--   PDF page so Silver chunks can produce verifiable citations.
--
-- Grain:
--   1 row = 1 page of 1 source document
-- ============================================================


CREATE TABLE IF NOT EXISTS bronze.document_pages (
    document_id INTEGER NOT NULL,

    page_number INTEGER NOT NULL,

    page_text TEXT NOT NULL,

    page_char_length INTEGER NOT NULL,

    loaded_at TIMESTAMP WITHOUT TIME ZONE
        NOT NULL DEFAULT CURRENT_TIMESTAMP,


    CONSTRAINT document_pages_pkey
        PRIMARY KEY (document_id, page_number),


    CONSTRAINT document_pages_document_id_fkey
        FOREIGN KEY (document_id)
        REFERENCES bronze.raw_documents (document_id)
        ON DELETE CASCADE,


    CONSTRAINT document_pages_page_number_check
        CHECK (page_number > 0),


    CONSTRAINT document_pages_char_length_check
        CHECK (page_char_length >= 0)
);
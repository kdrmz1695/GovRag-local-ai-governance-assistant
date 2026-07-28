-- ============================================================
-- Silver Citation Provenance Migration
-- Table: silver.document_chunks
-- Project: GovRAG - Local AI Governance Assistant
-- Purpose:
--   Add page and section provenance required for verifiable
--   citations without deleting existing Silver chunks.
-- ============================================================


BEGIN;


-- ============================================================
-- Add citation and lineage columns
-- ============================================================

ALTER TABLE silver.document_chunks
    ADD COLUMN IF NOT EXISTS page_start INTEGER,
    ADD COLUMN IF NOT EXISTS page_end INTEGER,
    ADD COLUMN IF NOT EXISTS section_title TEXT,
    ADD COLUMN IF NOT EXISTS chunk_hash TEXT,
    ADD COLUMN IF NOT EXISTS chunking_version TEXT;


-- ============================================================
-- Ensure chunk order is unique inside each document
-- ============================================================

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1
        FROM pg_constraint
        WHERE conname = 'document_chunks_document_order_key'
          AND conrelid = 'silver.document_chunks'::regclass
    ) THEN
        ALTER TABLE silver.document_chunks
            ADD CONSTRAINT document_chunks_document_order_key
            UNIQUE (document_id, chunk_order);
    END IF;
END
$$;


-- ============================================================
-- Validate citation page ranges
--
-- Valid:
--   page_start = NULL and page_end = NULL
--   page_start = 42 and page_end = 43
--
-- Invalid:
--   page_start = 43 and page_end = 42
--   only one of the page columns is populated
-- ============================================================

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1
        FROM pg_constraint
        WHERE conname = 'document_chunks_page_range_check'
          AND conrelid = 'silver.document_chunks'::regclass
    ) THEN
        ALTER TABLE silver.document_chunks
            ADD CONSTRAINT document_chunks_page_range_check
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
            );
    END IF;
END
$$;


-- ============================================================
-- Index section references for filtered retrieval
-- ============================================================

CREATE INDEX IF NOT EXISTS idx_document_chunks_section_reference
    ON silver.document_chunks (section_reference);


COMMIT;
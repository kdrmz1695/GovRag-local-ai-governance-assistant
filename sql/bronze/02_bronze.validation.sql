-- ============================================================
-- Bronze Layer Validation
-- Table: bronze.raw_documents
-- Project: GovRAG - Local AI Governance Assistant
-- Purpose:
--   Validate document ingestion, source metadata, extracted text,
--   page counts, and duplicate prevention.
-- ============================================================


-- 1. Total number of Bronze documents
SELECT
    COUNT(*) AS total_documents
FROM bronze.raw_documents;


-- 2. Document-level ingestion summary
SELECT
    document_id,
    document_name,
    source_name,
    file_name,
    file_type,
    page_count,
    LENGTH(full_text) AS full_text_char_length,
    loaded_at
FROM bronze.raw_documents
ORDER BY document_id;


-- 3. Check required fields
-- Expected result: 0 rows
SELECT
    document_id,
    document_name,
    file_name,
    file_type,
    page_count
FROM bronze.raw_documents
WHERE document_name IS NULL
   OR TRIM(document_name) = ''
   OR file_name IS NULL
   OR TRIM(file_name) = ''
   OR file_type IS NULL
   OR TRIM(file_type) = ''
   OR full_text IS NULL
   OR TRIM(full_text) = ''
   OR page_count IS NULL
   OR page_count <= 0;


-- 4. Check duplicate file hashes
-- Expected result: 0 rows
SELECT
    file_hash,
    COUNT(*) AS duplicate_count
FROM bronze.raw_documents
WHERE file_hash IS NOT NULL
GROUP BY file_hash
HAVING COUNT(*) > 1;


-- 5. Check missing source metadata
-- Expected result for the current three documents: 0 rows
SELECT
    document_id,
    document_name,
    source_name,
    source_url
FROM bronze.raw_documents
WHERE source_name IS NULL
   OR TRIM(source_name) = ''
   OR source_url IS NULL
   OR TRIM(source_url) = '';


-- 6. Check suspiciously short extracted documents
-- Expected result: 0 rows
SELECT
    document_id,
    document_name,
    page_count,
    LENGTH(full_text) AS full_text_char_length
FROM bronze.raw_documents
WHERE LENGTH(TRIM(full_text)) < 1000;


-- 7. Check file hash format
-- SHA-256 hashes must contain exactly 64 hexadecimal characters.
-- Expected result: 0 rows
SELECT
    document_id,
    document_name,
    file_hash
FROM bronze.raw_documents
WHERE file_hash IS NULL
   OR file_hash !~ '^[0-9a-fA-F]{64}$';
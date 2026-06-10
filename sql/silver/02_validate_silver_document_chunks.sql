-- ============================================================
-- Silver Layer Validation
-- Table: silver.document_chunks
-- Purpose:
--   Validate token-based chunk generation from Bronze documents.
-- ============================================================


-- 1. Check total number of chunks
SELECT 
    COUNT(*) AS total_chunks
FROM silver.document_chunks;


-- 2. Check chunk count and chunk size distribution per document
SELECT
    document_name,
    COUNT(*) AS chunk_count,
    MIN(chunk_token_count) AS min_tokens,
    MAX(chunk_token_count) AS max_tokens,
    AVG(chunk_token_count)::INTEGER AS avg_tokens,
    MIN(chunk_char_length) AS min_chars,
    MAX(chunk_char_length) AS max_chars,
    AVG(chunk_char_length)::INTEGER AS avg_chars
FROM silver.document_chunks
GROUP BY document_name
ORDER BY document_name;


-- 3. Preview first chunks from each document
SELECT
    document_name,
    chunk_order,
    chunk_token_count,
    LEFT(chunk_text, 500) AS chunk_preview
FROM silver.document_chunks
WHERE chunk_order <= 3
ORDER BY document_name, chunk_order;


-- 4. Check empty or very short chunks
SELECT
    chunk_id,
    document_name,
    chunk_order,
    chunk_token_count,
    chunk_char_length
FROM silver.document_chunks
WHERE chunk_text IS NULL
   OR LENGTH(TRIM(chunk_text)) < 100;


-- 5. Check whether every Bronze document produced chunks
SELECT
    b.document_id,
    b.document_name,
    COUNT(s.chunk_id) AS chunk_count
FROM bronze.raw_documents b
LEFT JOIN silver.document_chunks s
    ON b.document_id = s.document_id
GROUP BY b.document_id, b.document_name
ORDER BY b.document_id;
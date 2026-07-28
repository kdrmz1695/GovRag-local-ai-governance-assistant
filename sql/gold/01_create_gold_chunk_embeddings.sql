-- ============================================================
-- Create Gold Chunk Embeddings
-- Project: GovRAG - Local AI Governance Assistant
-- Grain:
--   One row per Silver chunk and concrete embedding model ID.
--
-- Notes:
--   - Silver remains the source of chunk text and citation metadata.
--   - Gold stores model-dependent vector representations.
--   - No approximate-nearest-neighbor index is created for V1.
--     Exact cosine search is the baseline for the current corpus.
-- ============================================================

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1
        FROM pg_extension
        WHERE extname = 'vector'
    ) THEN
        RAISE EXCEPTION
            'pgvector extension is required before creating Gold embeddings';
    END IF;
END;
$$;

CREATE TABLE IF NOT EXISTS gold.chunk_embeddings (
    embedding_id BIGSERIAL PRIMARY KEY,
    chunk_id INTEGER NOT NULL,
    chunk_hash TEXT NOT NULL,
    embedding_input_hash TEXT NOT NULL,
    model_alias TEXT NOT NULL,
    model_id TEXT NOT NULL,
    model_version TEXT NOT NULL,
    embedding_dimension INTEGER NOT NULL,
    embedding public.vector(1024) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT chunk_embeddings_chunk_id_fkey
        FOREIGN KEY (chunk_id)
        REFERENCES silver.document_chunks (chunk_id)
        ON DELETE CASCADE,

    CONSTRAINT chunk_embeddings_chunk_hash_format_check
        CHECK (chunk_hash ~ '^[0-9a-f]{64}$'),

    CONSTRAINT chunk_embeddings_input_hash_format_check
        CHECK (embedding_input_hash ~ '^[0-9a-f]{64}$'),

    CONSTRAINT chunk_embeddings_dimension_check
        CHECK (
            embedding_dimension = 1024
            AND embedding_dimension = public.vector_dims(embedding)
        ),

    CONSTRAINT chunk_embeddings_chunk_model_key
        UNIQUE (chunk_id, model_id)
);

COMMENT ON TABLE gold.chunk_embeddings IS
    'Model-dependent embeddings for citation-ready Silver chunks.';

COMMENT ON COLUMN gold.chunk_embeddings.chunk_id IS
    'Foreign key to the citation-ready Silver chunk.';

COMMENT ON COLUMN gold.chunk_embeddings.chunk_hash IS
    'SHA-256 hash copied from Silver for stale-data validation.';

COMMENT ON COLUMN gold.chunk_embeddings.embedding_input_hash IS
    'SHA-256 hash of the exact formatted text sent to the embedding model.';

COMMENT ON COLUMN gold.chunk_embeddings.model_alias IS
    'Stable Foundry Local catalog alias requested by the application.';

COMMENT ON COLUMN gold.chunk_embeddings.model_id IS
    'Concrete hardware-specific and versioned model ID selected at runtime.';

COMMENT ON COLUMN gold.chunk_embeddings.model_version IS
    'Version parsed from the concrete Foundry Local model ID.';

COMMENT ON COLUMN gold.chunk_embeddings.embedding IS
    '1024-dimensional Qwen3 embedding stored with pgvector.';
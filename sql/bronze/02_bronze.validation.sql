-- ============================================================
-- Create PostgreSQL Schemas
-- Project: GovRAG - Local AI Governance Assistant
-- Purpose:
--   Create medallion-inspired database schemas for the
--   document-to-vector RAG pipeline.
-- ============================================================

CREATE SCHEMA IF NOT EXISTS bronze;
CREATE SCHEMA IF NOT EXISTS silver;
CREATE SCHEMA IF NOT EXISTS gold;
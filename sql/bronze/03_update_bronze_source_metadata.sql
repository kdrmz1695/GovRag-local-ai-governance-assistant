-- ============================================================
-- Bronze Layer Metadata Update
-- Table: bronze.raw_documents
-- Purpose:
--   Add official source metadata for ingested legal/governance
--   documents after raw PDF ingestion.
-- ============================================================


UPDATE bronze.raw_documents
SET
    source_name = 'EDPB',
    source_url = 'https://www.edpb.europa.eu/our-work-tools/our-documents/opinion-board-art-64/opinion-282024-certain-data-protection-aspects_en'
WHERE document_name = 'edpb_opinion_202428_ai-models_en';


UPDATE bronze.raw_documents
SET
    source_name = 'EUR-Lex',
    source_url = 'https://eur-lex.europa.eu/eli/reg/2024/1689/oj/eng'
WHERE document_name = 'eu_ai_act_2024_1689';


UPDATE bronze.raw_documents
SET
    source_name = 'EUR-Lex',
    source_url = 'https://eur-lex.europa.eu/eli/reg/2016/679/oj/eng'
WHERE document_name = 'gdpr_2016_679';
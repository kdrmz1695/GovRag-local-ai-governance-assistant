from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from corpus_manifest import (
    CorpusDocument,
    ManifestValidationError,
    load_corpus_manifest,
    select_documents,
    validate_corpus_files,
    validate_manifest_data,
)
from ingestion import ingest_raw_pdfs_to_postgres as ingestion
from embeddings import create_gold_embeddings as gold
from processing import create_silver_chunks as silver
from processing.legal_structure import (
    UnsupportedParserError,
    parse_document_structure,
)


def valid_document(**overrides) -> dict:
    document = {
        "document_id": "doc_one",
        "file_name": "doc_one.pdf",
        "official_title": "Document One",
        "short_title": "Doc One",
        "issuing_body": "Issuer",
        "source_repository": "Repository",
        "source_url": "https://example.invalid/doc-one",
        "document_type": "guidelines",
        "legal_status": "test_fixture",
        "language": "en",
        "version_or_date": "test-version",
        "parser_type": "generic",
        "enabled": True,
    }
    document.update(overrides)
    return document


def valid_manifest(documents=None) -> dict:
    return {
        "schema_version": "1.0",
        "corpus_id": "test_corpus",
        "corpus_version": "test-version",
        "documents": documents or [valid_document()],
    }


class ManifestValidationTests(unittest.TestCase):
    def test_repository_manifest_has_ten_unique_entries(self):
        manifest = load_corpus_manifest()
        self.assertEqual(len(manifest.documents), 10)
        self.assertEqual(sum(document.enabled for document in manifest.documents), 3)

    def test_missing_required_field_is_rejected(self):
        data = valid_manifest()
        del data["documents"][0]["official_title"]
        with self.assertRaisesRegex(ManifestValidationError, "official_title"):
            validate_manifest_data(data)

    def test_duplicate_stable_document_ids_are_rejected(self):
        data = valid_manifest(
            [valid_document(), valid_document(file_name="second.pdf")]
        )
        with self.assertRaisesRegex(ManifestValidationError, "Duplicate document_id"):
            validate_manifest_data(data)

    def test_duplicate_filenames_are_case_insensitive(self):
        data = valid_manifest(
            [
                valid_document(),
                valid_document(document_id="doc_two", file_name="DOC_ONE.PDF"),
            ]
        )
        with self.assertRaisesRegex(ManifestValidationError, "Duplicate file_name"):
            validate_manifest_data(data)

    def test_filtering_by_document_id(self):
        manifest = validate_manifest_data(
            valid_manifest(
                [
                    valid_document(),
                    valid_document(document_id="doc_two", file_name="doc_two.pdf"),
                ]
            )
        )
        selected = select_documents(manifest, ["doc_two"], False)
        self.assertEqual([document.document_id for document in selected], ["doc_two"])


class CorpusFileValidationTests(unittest.TestCase):
    def test_missing_pdf_is_reported(self):
        manifest = validate_manifest_data(valid_manifest())
        with tempfile.TemporaryDirectory() as directory:
            missing, unknown = validate_corpus_files(manifest, Path(directory))
        self.assertEqual(missing, ["doc_one.pdf"])
        self.assertEqual(unknown, [])

    def test_unknown_pdf_is_reported(self):
        manifest = validate_manifest_data(valid_manifest())
        with tempfile.TemporaryDirectory() as directory:
            raw_dir = Path(directory)
            (raw_dir / "doc_one.pdf").touch()
            (raw_dir / "unknown.pdf").touch()
            missing, unknown = validate_corpus_files(manifest, raw_dir)
        self.assertEqual(missing, [])
        self.assertEqual(unknown, ["unknown.pdf"])


class PipelineSafetyTests(unittest.TestCase):
    def setUp(self):
        self.document = CorpusDocument(**valid_document())

    def test_bronze_dry_run_never_connects_to_database(self):
        prepared = {
            "document": self.document,
            "pdf_path": Path("doc_one.pdf"),
            "file_hash": "a" * 64,
            "full_text": "text",
            "pages": [{"page_number": 1, "page_text": "text"}],
        }
        connection_factory = Mock(side_effect=AssertionError("must not connect"))
        with patch.object(ingestion, "prepare_document", return_value=prepared):
            result = ingestion.run_ingestion(
                [self.document],
                dry_run=True,
                connection_factory=connection_factory,
            )
        self.assertEqual(result, [prepared])
        connection_factory.assert_not_called()

    def test_changed_hash_updates_existing_stable_identity(self):
        cursor = Mock()
        cursor.fetchall.return_value = [(7, "old-hash")]
        cursor.fetchone.side_effect = [None, (7,)]

        database_document_id = ingestion.upsert_raw_document(
            cursor,
            self.document,
            "b" * 64,
            "updated text",
            2,
        )

        self.assertEqual(database_document_id, 7)
        statements = [call.args[0] for call in cursor.execute.call_args_list]
        self.assertTrue(any("UPDATE bronze.raw_documents" in sql for sql in statements))
        self.assertFalse(any("INSERT INTO bronze.raw_documents" in sql for sql in statements))

    def test_unchanged_selected_document_is_not_rebuilt(self):
        chunks = [{"chunk_text": "unchanged"}]
        with (
            patch.object(silver, "get_bronze_document", return_value=(1, "doc_one", 1)),
            patch.object(silver, "get_document_pages", return_value=[(1, "text")]),
            patch.object(silver, "validate_detected_structure"),
            patch.object(silver, "print_structure_summary"),
            patch.object(silver, "create_section_aware_chunks", return_value=chunks),
            patch.object(silver, "document_chunks_are_current", return_value=True),
            patch.object(silver, "insert_document_chunks") as insert,
        ):
            status = silver.process_document(
                Mock(), self.document, Mock(), dry_run=False
            )
        self.assertEqual(status, "unchanged")
        insert.assert_not_called()

    def test_silver_dry_run_does_not_rebuild_changed_document(self):
        chunks = [{"chunk_text": "changed"}]
        with (
            patch.object(silver, "get_bronze_document", return_value=(1, "doc_one", 1)),
            patch.object(silver, "get_document_pages", return_value=[(1, "text")]),
            patch.object(silver, "validate_detected_structure"),
            patch.object(silver, "print_structure_summary"),
            patch.object(silver, "create_section_aware_chunks", return_value=chunks),
            patch.object(silver, "document_chunks_are_current", return_value=False),
            patch.object(silver, "insert_document_chunks") as insert,
        ):
            status = silver.process_document(
                Mock(), self.document, Mock(), dry_run=True
            )
        self.assertEqual(status, "would_rebuild")
        insert.assert_not_called()

    def test_unsupported_parser_is_explicit(self):
        with self.assertRaisesRegex(UnsupportedParserError, "Unsupported parser_type"):
            parse_document_structure(
                "future_guideline",
                [(1, "Heading\nBody")],
                parser_type="future_guideline_parser",
            )

    def test_gold_document_filter_is_applied_to_silver_query(self):
        cursor = Mock()
        cursor.fetchall.return_value = []
        self.assertEqual(
            gold.fetch_silver_chunks(
                cursor,
                limit=None,
                document_ids=["doc_one"],
            ),
            [],
        )
        query, parameters = cursor.execute.call_args.args
        self.assertIn("document_name = ANY(%s)", query)
        self.assertEqual(parameters, (["doc_one"],))


if __name__ == "__main__":
    unittest.main()

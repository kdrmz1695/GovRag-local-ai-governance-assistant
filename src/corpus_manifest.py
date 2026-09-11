from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST_PATH = PROJECT_ROOT / "config" / "corpus_manifest.json"
DEFAULT_RAW_DIR = PROJECT_ROOT / "data" / "raw"
MANUAL_REVIEW_MARKER = "MANUAL_REVIEW_REQUIRED"

REQUIRED_DOCUMENT_FIELDS = (
    "document_id",
    "file_name",
    "official_title",
    "short_title",
    "issuing_body",
    "source_repository",
    "source_url",
    "document_type",
    "legal_status",
    "language",
    "version_or_date",
    "parser_type",
    "enabled",
)

DOCUMENT_ID_PATTERN = re.compile(r"^[a-z0-9]+(?:[._-][a-z0-9]+)*$")


class ManifestValidationError(ValueError):
    pass


@dataclass(frozen=True)
class CorpusDocument:
    document_id: str
    file_name: str
    official_title: str
    short_title: str
    issuing_body: str
    source_repository: str
    source_url: str
    document_type: str
    legal_status: str
    language: str
    version_or_date: str
    parser_type: str
    enabled: bool

    @property
    def needs_manual_review(self) -> bool:
        return any(
            MANUAL_REVIEW_MARKER in value
            for value in (
                self.source_repository,
                self.source_url,
                self.legal_status,
                self.version_or_date,
            )
        )


@dataclass(frozen=True)
class CorpusManifest:
    schema_version: str
    corpus_id: str
    corpus_version: str
    documents: tuple[CorpusDocument, ...]

    def by_id(self) -> dict[str, CorpusDocument]:
        return {document.document_id: document for document in self.documents}

    def by_file_name(self) -> dict[str, CorpusDocument]:
        return {document.file_name.casefold(): document for document in self.documents}


def _required_string(document: dict, field: str, index: int) -> str:
    value = document.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ManifestValidationError(
            f"documents[{index}].{field} must be a non-empty string."
        )
    return value.strip()


def validate_manifest_data(data: dict) -> CorpusManifest:
    if not isinstance(data, dict):
        raise ManifestValidationError("Corpus manifest must be a JSON object.")

    for field in ("schema_version", "corpus_id", "corpus_version"):
        if not isinstance(data.get(field), str) or not data[field].strip():
            raise ManifestValidationError(f"{field} must be a non-empty string.")

    raw_documents = data.get("documents")
    if not isinstance(raw_documents, list) or not raw_documents:
        raise ManifestValidationError("documents must be a non-empty array.")

    documents = []
    seen_ids: dict[str, int] = {}
    seen_file_names: dict[str, int] = {}

    for index, raw_document in enumerate(raw_documents):
        if not isinstance(raw_document, dict):
            raise ManifestValidationError(f"documents[{index}] must be an object.")

        missing = [
            field for field in REQUIRED_DOCUMENT_FIELDS
            if field not in raw_document
        ]
        if missing:
            raise ManifestValidationError(
                f"documents[{index}] is missing required fields: {', '.join(missing)}"
            )

        values = {
            field: _required_string(raw_document, field, index)
            for field in REQUIRED_DOCUMENT_FIELDS
            if field != "enabled"
        }

        enabled = raw_document.get("enabled")
        if type(enabled) is not bool:
            raise ManifestValidationError(
                f"documents[{index}].enabled must be true or false."
            )

        document_id = values["document_id"]
        if not DOCUMENT_ID_PATTERN.fullmatch(document_id):
            raise ManifestValidationError(
                f"documents[{index}].document_id is not a stable slug: {document_id!r}"
            )

        file_name = values["file_name"]
        if Path(file_name).name != file_name or Path(file_name).suffix.casefold() != ".pdf":
            raise ManifestValidationError(
                f"documents[{index}].file_name must be a PDF basename: {file_name!r}"
            )

        normalized_id = document_id.casefold()
        if normalized_id in seen_ids:
            raise ManifestValidationError(
                f"Duplicate document_id {document_id!r} at documents[{seen_ids[normalized_id]}] "
                f"and documents[{index}]."
            )
        seen_ids[normalized_id] = index

        normalized_file_name = file_name.casefold()
        if normalized_file_name in seen_file_names:
            raise ManifestValidationError(
                f"Duplicate file_name {file_name!r} at "
                f"documents[{seen_file_names[normalized_file_name]}] and documents[{index}]."
            )
        seen_file_names[normalized_file_name] = index

        documents.append(CorpusDocument(enabled=enabled, **values))

    return CorpusManifest(
        schema_version=data["schema_version"].strip(),
        corpus_id=data["corpus_id"].strip(),
        corpus_version=data["corpus_version"].strip(),
        documents=tuple(documents),
    )


def load_corpus_manifest(path: Path = DEFAULT_MANIFEST_PATH) -> CorpusManifest:
    try:
        raw_data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise ManifestValidationError(f"Corpus manifest not found: {path}") from error
    except json.JSONDecodeError as error:
        raise ManifestValidationError(
            f"Corpus manifest is not valid JSON: {path}: {error}"
        ) from error

    return validate_manifest_data(raw_data)


def select_documents(
    manifest: CorpusManifest,
    document_ids: list[str] | None,
    select_all: bool,
) -> list[CorpusDocument]:
    if bool(document_ids) == bool(select_all):
        raise ManifestValidationError(
            "Select one or more --document-id values, or use --all."
        )

    if select_all:
        selected = [document for document in manifest.documents if document.enabled]
        if not selected:
            raise ManifestValidationError("The manifest has no enabled documents.")
        return selected

    by_id = manifest.by_id()
    selected = []
    seen = set()
    for document_id in document_ids or []:
        if document_id in seen:
            continue
        seen.add(document_id)
        document = by_id.get(document_id)
        if document is None:
            raise ManifestValidationError(
                f"Unknown manifest document_id: {document_id}"
            )
        if not document.enabled:
            raise ManifestValidationError(
                f"Manifest document is disabled: {document_id}"
            )
        selected.append(document)
    return selected


def validate_corpus_files(
    manifest: CorpusManifest,
    raw_dir: Path = DEFAULT_RAW_DIR,
) -> tuple[list[str], list[str]]:
    expected = manifest.by_file_name()
    actual_files = sorted(
        (
            path for path in raw_dir.iterdir()
            if path.is_file() and path.suffix.casefold() == ".pdf"
        ),
        key=lambda path: path.name.casefold(),
    ) if raw_dir.is_dir() else []

    missing = [
        document.file_name
        for document in manifest.documents
        if not (raw_dir / document.file_name).is_file()
    ]
    unknown = [
        path.name
        for path in actual_files
        if path.name.casefold() not in expected
    ]
    return missing, unknown


def validate_selected_files(
    selected: list[CorpusDocument],
    raw_dir: Path = DEFAULT_RAW_DIR,
) -> None:
    missing = [
        document.file_name
        for document in selected
        if not (raw_dir / document.file_name).is_file()
    ]
    if missing:
        raise ManifestValidationError(
            "Selected manifest PDF file(s) are missing from "
            f"{raw_dir}: {', '.join(missing)}"
        )

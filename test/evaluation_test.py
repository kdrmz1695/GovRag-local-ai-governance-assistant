"""Read-only GovRAG retrieval evaluation. Production modules are NOT modified.

Place this file and eval_questions.json in <gov_rag>/tests/.
Run from the project root:
    python tests/evaluation_test.py
    python tests/evaluation_test.py --only GDPR-01
    python tests/evaluation_test.py --self-test

Uses the existing retrieval.semantic_search functions and .env configuration.
No chat generation, training, ingestion, or database writes are performed.
Each run creates a new tests/evaluation_results/<run-id>/ directory containing
results.json (passages + diagnostics) and summary.csv (one row per question).

Exit codes: 0 = evaluation completed (not necessarily perfect retrieval),
1 = --fail-on-miss was requested and at least one reranked source was missed,
2 = invalid configuration, unresolved ground truth, or runtime error,
130 = interrupted. --self-test needs only Python's standard library.

This is an explicitly invoked integration evaluation, not an automatic pytest
test. Importing this file does not connect to services or load models.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from copy import deepcopy
import csv
from datetime import datetime, timezone
import gc
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import platform
import sys
import time
import unicodedata
import uuid

__test__ = False
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
INVENTORY_SQL = """
SELECT s.chunk_id, s.document_name, s.section_type, s.section_reference,
       s.page_start, s.page_end, s.chunking_version,
       EXISTS (
           SELECT 1 FROM gold.chunk_embeddings AS g
           WHERE g.chunk_id = s.chunk_id
             AND g.model_id = %s AND g.model_version = %s
       ) AS searchable
FROM silver.document_chunks AS s
ORDER BY s.chunk_id;
"""


def normalize(value) -> str:
    return " ".join(unicodedata.normalize("NFKC", str(value or "")).split()).casefold()


def document_name(value) -> str:
    return normalize(value).removesuffix(".pdf").strip()


def matches_source(chunk: dict, expected_document: str, expected: dict) -> bool:
    if document_name(chunk.get("document_name")) != document_name(expected_document):
        return False
    for field in ("section_type", "section_reference"):
        if normalize(chunk.get(field)) != normalize(expected[field]):
            return False
    if "pdf_pages" in expected:
        start, end = chunk.get("page_start"), chunk.get("page_end")
        if type(start) is not int or type(end) is not int or start < 1 or end < start:
            return False
        if not any(start <= page <= end for page in expected["pdf_pages"]):
            return False
    return True


def first_hit_rank(chunks: list[dict], expected_document: str, expected: list[dict]):
    for rank, chunk in enumerate(chunks, start=1):
        if any(matches_source(chunk, expected_document, source) for source in expected):
            return rank
    return None


def validate_dataset(data: dict) -> None:
    if not isinstance(data, dict) or data.get("schema_version") != "1.0":
        raise ValueError("Expected evaluation JSON schema_version 1.0.")
    documents, questions = data.get("documents"), data.get("questions")
    if not isinstance(documents, dict) or not isinstance(questions, list) or not questions:
        raise ValueError("Dataset must contain documents and a nonempty questions list.")
    seen = set()
    for question in questions:
        if not isinstance(question, dict):
            raise ValueError("Each question must be an object.")
        identifier = question.get("id")
        if not isinstance(identifier, str) or not identifier.strip() or identifier in seen:
            raise ValueError("Question IDs must be unique, nonempty strings.")
        seen.add(identifier)
        if not isinstance(question.get("question"), str) or not question["question"].strip():
            raise ValueError(f"{identifier}: question text is missing.")
        document = documents.get(question.get("document_key"), {})
        if not isinstance(document, dict) or not isinstance(document.get("document_name"), str) or not document["document_name"].strip():
            raise ValueError(f"{identifier}: document_name is missing.")
        sources = question.get("expected_sources")
        if not isinstance(sources, list) or not sources:
            raise ValueError(f"{identifier}: expected_sources must be a nonempty list.")
        for source in sources:
            if not isinstance(source, dict) or any(
                not isinstance(source.get(key), str) or not source[key].strip()
                for key in ("section_type", "section_reference")
            ):
                raise ValueError(f"{identifier}: invalid source section metadata.")
            pages = source.get("pdf_pages")
            if "pdf_pages" in source and (
                not isinstance(pages, list) or not pages
                or any(type(page) is not int or page < 1 for page in pages)
            ):
                raise ValueError(f"{identifier}: pdf_pages must contain positive integers.")


def error_text(error: Exception) -> str:
    message = f"{type(error).__name__}: {error}"
    password = os.getenv("DB_PASSWORD")
    if password:
        message = message.replace(password, "<redacted>")
    return message


@contextmanager
def readonly_cursor(backend):
    connection = backend.get_db_connection()
    cursor = None
    try:
        # Enforced by PostgreSQL, not just a promise to issue SELECT statements.
        connection.set_session(readonly=True, autocommit=False)
        cursor = connection.cursor()
        yield cursor
    finally:
        try:
            if cursor is not None:
                cursor.close()
        finally:
            connection.close()  # No commit: all operations are read-only.


def get_inventory(backend) -> list[dict]:
    with readonly_cursor(backend) as cursor:
        cursor.execute(INVENTORY_SQL, (backend.EMBEDDING_MODEL_ID, backend.EMBEDDING_MODEL_VERSION))
        rows = cursor.fetchall()
    keys = ("chunk_id", "document_name", "section_type", "section_reference",
            "page_start", "page_end", "chunking_version", "searchable")
    return [dict(zip(keys, row, strict=True)) for row in rows]


def validate_chunks(chunks, limit: int, score_field: str) -> None:
    if not isinstance(chunks, list) or len(chunks) > limit:
        raise ValueError("Pipeline returned an invalid candidate list or more than the requested K.")
    for chunk in chunks:
        if not isinstance(chunk, dict) or chunk.get("chunk_id") is None:
            raise ValueError("A returned candidate is missing chunk_id.")
        if not isinstance(chunk.get("chunk_text"), str) or not chunk["chunk_text"].strip():
            raise ValueError("A returned candidate has no usable chunk_text.")
        if not isinstance(chunk.get("document_name"), str) or not chunk["document_name"].strip():
            raise ValueError("A returned candidate has no document_name.")
        if not math.isfinite(float(chunk[score_field])):
            raise ValueError(f"A returned candidate has a non-finite {score_field}.")


def summary(report: dict) -> dict:
    records = report["questions"]
    result = {"total_questions": len(records), "status_counts": {}}
    for record in records:
        status = record["status"]
        result["status_counts"][status] = result["status_counts"].get(status, 0) + 1
    for field in ("vector_hit", "vector_top_k_hit", "reranked_hit"):
        measured = [r[field] for r in records if type(r[field]) is bool]
        result[field] = {
            "hits": sum(measured), "measured_questions": len(measured),
            "rate": sum(measured) / len(measured) if measured else None,
        }
    paired = [r for r in records if type(r["reranked_hit"]) is bool]
    result["same_k_comparison"] = {
        "paired_questions": len(paired),
        "gained_hits": sum(r["reranked_hit"] and not r["vector_top_k_hit"] for r in paired),
        "lost_hits": sum(r["vector_top_k_hit"] and not r["reranked_hit"] for r in paired),
    }
    return result


def make_report(data: dict, candidate_k: int, top_k: int) -> dict:
    records = []
    for question in data["questions"]:
        records.append({
            "id": question["id"], "question": question["question"],
            "document_key": question["document_key"],
            "expected_document": data["documents"][question["document_key"]]["document_name"],
            "expected_sources": deepcopy(question["expected_sources"]),
            "review_reference": deepcopy(question),
            "status": "pending", "error": None, "error_stage": None,
            "vector_hit": None, "vector_top_k_hit": None, "reranked_hit": None,
            "first_vector_hit_rank": None, "first_reranked_hit_rank": None,
            "vector_candidates": [], "reranked_top_k": [], "seconds": {},
        })
    return {
        "run_status": "running", "started_at": datetime.now(timezone.utc).isoformat(),
        "dataset_id": data.get("dataset_id"), "dataset_version": data.get("version"),
        "candidate_k": candidate_k, "top_k": top_k,
        "evaluation_rules": deepcopy(data.get("evaluation_rules", {})),
        "scope": "Source-anchor retrieval evaluation only. No answer generation or citation verification.",
        "metric_labels": {
            "vector_hit": f"source_anchor_hit_at_{candidate_k}",
            "vector_top_k_hit": f"vector_source_anchor_hit_at_{top_k}",
            "reranked_hit": f"reranked_source_anchor_hit_at_{top_k}",
        },
        "denominators": "Each rate uses only measured questions. Errors and unresolved ground truth remain visible in status_counts; they are not counted as successes or retrieval misses.",
        "questions": records,
    }


def evaluate(report: dict, backend, checkpoint=lambda report: None) -> None:
    candidate_k, top_k = report["candidate_k"], report["top_k"]
    print("[Preflight] Checking expected source anchors in Silver and active Gold embeddings...", flush=True)
    inventory = get_inventory(backend)
    report["corpus"] = {
        "silver_chunk_count": len(inventory),
        "searchable_chunk_count": sum(bool(row["searchable"]) for row in inventory),
        "document_names": sorted({row["document_name"] for row in inventory}),
        "chunking_versions": sorted({str(row["chunking_version"]) for row in inventory}),
        "note": "Do not modify/re-ingest the corpus during a run. Queries use separate read-only transactions.",
    }
    for record in report["questions"]:
        matching = [row for row in inventory if any(
            matches_source(row, record["expected_document"], expected)
            for expected in record["expected_sources"]
        )]
        record["matching_silver_chunks"] = len(matching)
        record["matching_searchable_chunks"] = sum(bool(row["searchable"]) for row in matching)
        if not record["matching_searchable_chunks"]:
            record["status"] = "unresolved_ground_truth"
            record["error_stage"] = "preflight"
            record["error"] = (
                "Expected section exists in Silver but has no Gold embedding for the configured model/version."
                if matching else "Expected document/section/page metadata was not found in Silver. Check dataset names and ingestion metadata."
            )
            print(f"  {record['id']}: UNRESOLVED - {record['error']}", flush=True)
    checkpoint(report)

    # Finish ALL query embeddings before loading the Python GPU reranker.
    # This avoids alternating GPU workloads and loads the reranker only once.
    print(f"[1/2] Query embeddings and pgvector top-{candidate_k}...", flush=True)
    for index, record in enumerate(report["questions"], start=1):
        if record["status"] != "pending":
            continue
        stage = "embedding"
        try:
            print(f"  [{index}/{len(report['questions'])}] {record['id']}", flush=True)
            started = time.perf_counter()
            embedding = backend.request_query_embedding(record["question"])
            record["seconds"]["embedding"] = round(time.perf_counter() - started, 4)
            stage = "vector_retrieval"
            started = time.perf_counter()
            with readonly_cursor(backend) as cursor:
                candidates = backend.retrieve_vector_candidates(
                    cursor=cursor, query_embedding=embedding, candidate_k=candidate_k,
                )
            validate_chunks(candidates, candidate_k, "vector_similarity")
            record["seconds"]["vector_retrieval"] = round(time.perf_counter() - started, 4)
            record["vector_candidates"] = deepcopy(candidates)
            record["duplicate_vector_chunk_count"] = len(candidates) - len({str(c["chunk_id"]) for c in candidates})
            rank = first_hit_rank(candidates, record["expected_document"], record["expected_sources"])
            record["first_vector_hit_rank"] = rank
            record["vector_hit"] = rank is not None
            record["vector_top_k_hit"] = rank is not None and rank <= top_k
            record["status"] = "retrieved" if candidates else "completed"
            if not candidates:
                record["reranked_hit"] = False
                record["note"] = "No candidates returned; reranking skipped."
        except Exception as error:
            record.update(status="error", error_stage=stage, error=error_text(error))
            print(f"  {record['id']}: ERROR ({stage}) - {record['error']}", flush=True)
        checkpoint(report)

    pending = [r for r in report["questions"] if r["status"] == "retrieved"]
    if not pending:
        return
    print(f"[2/2] Loading reranker once; selecting top-{top_k} for each question...", flush=True)
    model = None
    try:
        started = time.perf_counter()
        try:
            model = backend.load_reranker()
        except Exception as error:
            for record in pending:
                record.update(status="error", error_stage="reranker_load", error=error_text(error))
            print(f"  Reranker could not be loaded: {error_text(error)}", flush=True)
            checkpoint(report)
            return
        report["reranker_load_seconds"] = round(time.perf_counter() - started, 4)
        for record in pending:
            try:
                started = time.perf_counter()
                # The production reranker mutates candidate dicts. Preserve the baseline.
                rerank_input = deepcopy(record["vector_candidates"])
                reranked = backend.rerank_candidates(
                    question=record["question"], candidates=rerank_input,
                    model=model, top_k=top_k,
                )
                validate_chunks(rerank_input, candidate_k, "reranker_score")
                validate_chunks(reranked, top_k, "reranker_score")
                if len(reranked) != min(top_k, len(record["vector_candidates"])):
                    raise ValueError("Reranker returned an unexpected number of passages.")
                original_ids = {str(c["chunk_id"]) for c in record["vector_candidates"]}
                if any(str(c["chunk_id"]) not in original_ids for c in reranked):
                    raise ValueError("Reranker returned a chunk outside the vector candidate set.")
                record["reranked_top_k"] = deepcopy(reranked)
                rank = first_hit_rank(reranked, record["expected_document"], record["expected_sources"])
                if rank is not None and not record["vector_hit"]:
                    raise ValueError("Invalid result: reranked source hit was absent from the candidate pool.")
                record["first_reranked_hit_rank"] = rank
                record["reranked_hit"] = rank is not None
                record["seconds"]["reranking"] = round(time.perf_counter() - started, 4)
                record["status"] = "completed"
                print(f"  {record['id']}: vector={record['vector_hit']} | vector top-{top_k}={record['vector_top_k_hit']} | reranked={record['reranked_hit']}", flush=True)
            except Exception as error:
                record.update(status="error", error_stage="reranking", error=error_text(error))
                print(f"  {record['id']}: ERROR (reranking) - {record['error']}", flush=True)
            checkpoint(report)
    finally:
        del model
        gc.collect()
        try:
            if backend.torch.cuda.is_available():
                backend.torch.cuda.empty_cache()
        except Exception:
            pass  # Cleanup must not hide evaluation results or the original error.


def save_reports(report: dict, directory: Path) -> None:
    report["summary"] = summary(report)
    temporary = directory / "results.json.tmp"
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(directory / "results.json")
    temporary = directory / "summary.csv.tmp"
    with temporary.open("w", encoding="utf-8-sig", newline="") as stream:
        fields = ["id", "status", "question", "expected_document", "vector_hit",
                  "vector_top_k_hit", "reranked_hit", "first_vector_hit_rank",
                  "first_reranked_hit_rank", "error_stage", "error"]
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for record in report["questions"]:
            # Neutralize spreadsheet formulas in arbitrary future question/error text.
            row = {field: record.get(field) for field in fields}
            for field, value in row.items():
                if isinstance(value, str) and value.lstrip().startswith(("=", "+", "-", "@")):
                    row[field] = "'" + value
            writer.writerow(row)
    temporary.replace(directory / "summary.csv")


def print_summary(report: dict, directory: Path) -> None:
    values = summary(report)
    print("\nGOVRAG SOURCE-ANCHOR EVALUATION")
    for field, label in report["metric_labels"].items():
        metric = values[field]
        rate = f"{metric['rate']:.1%}" if metric["rate"] is not None else "not measured"
        print(f"{label}: {metric['hits']}/{metric['measured_questions']} ({rate})")
    print(f"Question statuses (total {values['total_questions']}): {values['status_counts']}")
    changes = values["same_k_comparison"]
    print(f"Reranker vs vector at same K: gained={changes['gained_hits']}, lost={changes['lost_hits']} ({changes['paired_questions']} paired questions)")
    print("These are source-section matches, NOT legal-answer accuracy or citation correctness.")
    print(f"Reports: {directory}")


def self_test() -> int:
    import unittest

    class ScoringTests(unittest.TestCase):
        def setUp(self):
            self.source = {"section_type": "article", "section_reference": "Article 6"}
            self.chunk = {"document_name": "gdpr_2016_679", **self.source}

        def test_exact(self):
            self.assertTrue(matches_source(self.chunk, "gdpr_2016_679", self.source))

        def test_case_whitespace_extension(self):
            chunk = {**self.chunk, "section_reference": " ARTICLE   6 "}
            self.assertTrue(matches_source(chunk, " GDPR_2016_679.PDF ", self.source))

        def test_wrong_document(self):
            self.assertFalse(matches_source(self.chunk, "eu_ai_act_2024_1689", self.source))

        def test_article_60_is_not_6(self):
            self.assertFalse(matches_source({**self.chunk, "section_reference": "Article 60"}, "gdpr_2016_679", self.source))

        def test_text_mention_is_not_metadata(self):
            chunk = {**self.chunk, "section_reference": "Article 7", "chunk_text": "Article 6"}
            self.assertFalse(matches_source(chunk, "gdpr_2016_679", self.source))

        def test_wrong_type(self):
            self.assertFalse(matches_source({**self.chunk, "section_type": "recital"}, "gdpr_2016_679", self.source))

        def test_page_overlap(self):
            self.assertTrue(matches_source({**self.chunk, "page_start": 15, "page_end": 17}, "gdpr_2016_679", {**self.source, "pdf_pages": [16]}))

        def test_wrong_page(self):
            self.assertFalse(matches_source({**self.chunk, "page_start": 18, "page_end": 19}, "gdpr_2016_679", {**self.source, "pdf_pages": [16]}))

        def test_missing_page(self):
            self.assertFalse(matches_source(self.chunk, "gdpr_2016_679", {**self.source, "pdf_pages": [16]}))

        def test_invalid_page_interval(self):
            self.assertFalse(matches_source({**self.chunk, "page_start": 17, "page_end": 15}, "gdpr_2016_679", {**self.source, "pdf_pages": [16]}))

        def test_first_rank_and_duplicates(self):
            wrong = {**self.chunk, "section_reference": "Article 60"}
            self.assertEqual(first_hit_rank([wrong, self.chunk, self.chunk], "gdpr_2016_679", [self.source]), 2)

        def test_alternative_anchor(self):
            wrong = {**self.source, "section_reference": "Article 60"}
            self.assertEqual(first_hit_rank([self.chunk], "gdpr_2016_679", [wrong, self.source]), 1)

        def test_empty(self):
            self.assertIsNone(first_hit_rank([], "gdpr_2016_679", [self.source]))

        def test_unmeasured_is_not_success(self):
            records = [
                {"status": "completed", "vector_hit": True, "vector_top_k_hit": False, "reranked_hit": True},
                {"status": "error", "vector_hit": None, "vector_top_k_hit": None, "reranked_hit": None},
            ]
            result = summary({"questions": records})
            self.assertEqual(result["total_questions"], 2)
            self.assertEqual(result["vector_hit"]["measured_questions"], 1)
            self.assertEqual(result["status_counts"]["error"], 1)
            self.assertEqual(result["same_k_comparison"]["gained_hits"], 1)

    suite = unittest.defaultTestLoader.loadTestsFromTestCase(ScoringTests)
    return 0 if unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful() else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", type=Path, help="Default: adjacent eval_questions.json, or evaluation/eval_questions.json.")
    parser.add_argument("--candidate-k", type=int)
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--only", nargs="+", help="Evaluate only these question IDs.")
    parser.add_argument("--fail-on-miss", action="store_true")
    parser.add_argument("--self-test", action="store_true", help="Test scoring without Foundry, PostgreSQL, or third-party packages.")
    args = parser.parse_args()
    if args.self_test:
        return self_test()
    try:
        path = args.dataset
        if path is None:
            path = SCRIPT_DIR / "eval_questions.json"
            if not path.is_file():
                path = PROJECT_ROOT / "evaluation" / "eval_questions.json"
        raw_data = path.read_bytes()
        data = json.loads(raw_data.decode("utf-8-sig"))
        validate_dataset(data)
        if args.only:
            unknown = set(args.only) - {q["id"] for q in data["questions"]}
            if unknown:
                raise ValueError(f"Unknown question IDs: {sorted(unknown)}")
            data["questions"] = [q for q in data["questions"] if q["id"] in args.only]
        settings = data.get("retrieval_settings", {})
        candidate_k = args.candidate_k if args.candidate_k is not None else settings.get("candidate_k", 20)
        top_k = args.top_k if args.top_k is not None else settings.get("rerank_top_k", 4)
        if type(candidate_k) is not int or type(top_k) is not int or not 0 < top_k <= candidate_k:
            raise ValueError("K settings must satisfy 0 < top_k <= candidate_k.")
        source = PROJECT_ROOT / "src" / "retrieval" / "semantic_search.py"
        if not source.is_file():
            raise FileNotFoundError("Place evaluation_test.py in your GovRAG tests folder; src/retrieval/semantic_search.py was not found.")
    except (ValueError, TypeError, OSError) as error:
        print(f"Evaluation setup failed: {error_text(error)}", file=sys.stderr)
        return 2

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8]
    directory = SCRIPT_DIR / "evaluation_results" / run_id
    directory.mkdir(parents=True, exist_ok=False)
    report = make_report(data, candidate_k, top_k)
    report.update(dataset_file=str(path.resolve()), dataset_sha256=hashlib.sha256(raw_data).hexdigest(),
                  semantic_search_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                  runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), python_version=platform.python_version())
    checkpoint = lambda current: save_reports(current, directory)
    checkpoint(report)
    exit_code = 0
    started = time.perf_counter()
    try:
        sys.path.insert(0, str(PROJECT_ROOT / "src"))
        backend = importlib.import_module("retrieval.semantic_search")
        if Path(backend.__file__).resolve() != source.resolve():
            raise RuntimeError("A different retrieval.semantic_search module was imported; check your Python environment.")
        report["models"] = {name: getattr(backend, name) for name in (
            "EMBEDDING_MODEL_ID", "EMBEDDING_MODEL_VERSION", "EXPECTED_DIMENSION",
            "RERANKER_MODEL_ID", "RERANKER_DEVICE",
        )}
        print(f"Dataset: {path.resolve()} | Questions: {len(data['questions'])}", flush=True)
        print(f"Using existing semantic_search.py; top-{candidate_k} -> top-{top_k}. Database access is read-only.", flush=True)
        evaluate(report, backend, checkpoint)
        incomplete = any(r["status"] != "completed" for r in report["questions"])
        report["run_status"] = "incomplete" if incomplete else "completed"
        if incomplete:
            exit_code = 2
        elif args.fail_on_miss and any(not r["reranked_hit"] for r in report["questions"]):
            exit_code = 1
    except KeyboardInterrupt:
        report["run_status"] = "interrupted"
        exit_code = 130
    except Exception as error:
        report.update(run_status="error", fatal_error=error_text(error))
        print(f"Evaluation stopped: {error_text(error)}", file=sys.stderr)
        exit_code = 2
    finally:
        for record in report["questions"]:
            if record["status"] in ("pending", "retrieved"):
                record["status"] = "not_completed"
        report["finished_at"] = datetime.now(timezone.utc).isoformat()
        report["total_seconds"] = round(time.perf_counter() - started, 4)
        checkpoint(report)
        print_summary(report, directory)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())

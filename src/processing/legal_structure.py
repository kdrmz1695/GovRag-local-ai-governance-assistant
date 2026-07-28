from dataclasses import dataclass
import re


@dataclass(frozen=True)
class LineRecord:
    page_number: int
    text: str


@dataclass(frozen=True)
class AnnotatedLine:
    page_number: int
    text: str
    section_type: str
    section_reference: str
    section_title: str | None


@dataclass(frozen=True)
class SectionMetadata:
    section_type: str
    section_reference: str
    section_title: str | None


LEGAL_DOCUMENT_NAMES = {
    "eu_ai_act_2024_1689",
    "gdpr_2016_679",
}

EDPB_DOCUMENT_NAME = "edpb_opinion_202428_ai-models_en"

SECTION_NUMBER_PATTERN = re.compile(r"\d+(?:\.\d+){0,2}")
COMBINED_SECTION_PATTERN = re.compile(
    r"^(\d+(?:\.\d+){0,2})\s+(.+)$"
)


def normalize_line(line: str) -> str:
    return " ".join(line.split())


def create_line_records(
    pages: list[tuple[int, str]],
) -> list[LineRecord]:
    records = []

    for page_number, page_text in pages:
        for raw_line in page_text.splitlines():
            normalized_line = normalize_line(raw_line)

            if not normalized_line:
                continue

            records.append(
                LineRecord(
                    page_number=page_number,
                    text=normalized_line,
                )
            )

    return records


def get_next_line_title(
    records: list[LineRecord],
    current_index: int,
) -> str | None:
    if current_index + 1 >= len(records):
        return None

    return records[current_index + 1].text.rstrip("`").strip()


def annotate_line(
    record: LineRecord,
    metadata: SectionMetadata,
) -> AnnotatedLine:
    return AnnotatedLine(
        page_number=record.page_number,
        text=record.text,
        section_type=metadata.section_type,
        section_reference=metadata.section_reference,
        section_title=metadata.section_title,
    )


def parse_legal_document(
    records: list[LineRecord],
) -> list[AnnotatedLine]:
    current_metadata = SectionMetadata(
        section_type="front_matter",
        section_reference="Front matter",
        section_title=None,
    )

    expected_recital = 1
    expected_article = 1

    in_recitals = False
    articles_enabled = False
    annex_mode = False

    annotated_lines = []

    for index, record in enumerate(records):
        line = record.text
        upper_line = line.upper()
        next_title = get_next_line_title(records, index)

        if upper_line == "WHEREAS:":
            in_recitals = True

        if upper_line == "HAVE ADOPTED THIS REGULATION:":
            in_recitals = False
            articles_enabled = True

            current_metadata = SectionMetadata(
                section_type="enacting_formula",
                section_reference="Enacting formula",
                section_title=None,
            )

        recital_match = re.fullmatch(
            r"\((\d+)\)(?:\s+.*)?",
            line,
        )

        if (
            in_recitals
            and recital_match
            and int(recital_match.group(1)) == expected_recital
        ):
            current_metadata = SectionMetadata(
                section_type="recital",
                section_reference=f"Recital {expected_recital}",
                section_title=None,
            )
            expected_recital += 1

        annex_match = re.fullmatch(
            r"ANNEX\s+([IVXLCDM]+)",
            line,
            flags=re.IGNORECASE,
        )

        if annex_match:
            annex_mode = True

            annex_number = annex_match.group(1).upper()

            current_metadata = SectionMetadata(
                section_type="annex",
                section_reference=f"Annex {annex_number}",
                section_title=next_title,
            )

        if articles_enabled and not annex_mode:
            chapter_match = re.fullmatch(
                r"CHAPTER\s+([IVXLCDM]+)",
                line,
                flags=re.IGNORECASE,
            )

            legal_section_match = re.fullmatch(
                r"SECTION\s+(\d+)",
                line,
                flags=re.IGNORECASE,
            )

            article_match = re.fullmatch(
                r"Article\s+(\d+)",
                line,
                flags=re.IGNORECASE,
            )

            if chapter_match:
                chapter_number = chapter_match.group(1).upper()

                current_metadata = SectionMetadata(
                    section_type="chapter",
                    section_reference=f"Chapter {chapter_number}",
                    section_title=next_title,
                )

            elif legal_section_match:
                section_number = legal_section_match.group(1)

                current_metadata = SectionMetadata(
                    section_type="legal_section",
                    section_reference=f"Section {section_number}",
                    section_title=next_title,
                )

            elif (
                article_match
                and int(article_match.group(1)) == expected_article
            ):
                current_metadata = SectionMetadata(
                    section_type="article",
                    section_reference=f"Article {expected_article}",
                    section_title=next_title,
                )
                expected_article += 1

        annotated_lines.append(
            annotate_line(record, current_metadata)
        )

    return annotated_lines


def clean_toc_title(title_parts: list[str]) -> str:
    joined_title = " ".join(title_parts)

    return re.sub(
        r"\s*\.{3,}\s*\d+\s*$",
        "",
        joined_title,
    ).strip()


def extract_edpb_toc(
    records: list[LineRecord],
) -> tuple[int, list[str], dict[str, str]]:
    try:
        toc_start = next(
            index
            for index, record in enumerate(records)
            if record.text.lower() == "table of contents"
        )
    except StopIteration as error:
        raise ValueError(
            "EDPB table of contents could not be found."
        ) from error

    first_reference_index = next(
        index
        for index in range(toc_start + 1, len(records))
        if records[index].text == "1"
    )

    first_title_line = records[first_reference_index + 1].text
    first_title = clean_toc_title([first_title_line])

    body_heading_pattern = re.compile(
        rf"^1\s+{re.escape(first_title)}$",
        flags=re.IGNORECASE,
    )

    try:
        body_start = next(
            index
            for index in range(first_reference_index + 1, len(records))
            if body_heading_pattern.fullmatch(records[index].text)
        )
    except StopIteration as error:
        raise ValueError(
            "EDPB body start could not be found."
        ) from error

    section_order = []
    section_titles = {}

    index = first_reference_index

    while index < body_start:
        reference = records[index].text

        if not SECTION_NUMBER_PATTERN.fullmatch(reference):
            index += 1
            continue

        title_parts = []
        lookahead = index + 1
        found_page_leader = False

        while lookahead < body_start:
            candidate_line = records[lookahead].text

            if SECTION_NUMBER_PATTERN.fullmatch(candidate_line):
                break

            title_parts.append(candidate_line)

            if re.search(
                r"\.{3,}\s*\d+\s*$",
                candidate_line,
            ):
                found_page_leader = True
                break

            if len(title_parts) >= 6:
                break

            lookahead += 1

        if found_page_leader and reference not in section_titles:
            section_order.append(reference)
            section_titles[reference] = clean_toc_title(title_parts)

        index += 1

    if not section_order:
        raise ValueError(
            "No EDPB section entries were extracted from the TOC."
        )

    return body_start, section_order, section_titles


def parse_edpb_document(
    records: list[LineRecord],
) -> list[AnnotatedLine]:
    body_start, section_order, section_titles = extract_edpb_toc(
        records
    )

    current_metadata = SectionMetadata(
        section_type="front_matter",
        section_reference="Front matter",
        section_title=None,
    )

    expected_section_index = 0
    annotated_lines = []

    for index, record in enumerate(records):
        line = record.text

        if line.lower() == "executive summary":
            current_metadata = SectionMetadata(
                section_type="executive_summary",
                section_reference="Executive summary",
                section_title="Executive summary",
            )

        if line.lower() == "table of contents":
            current_metadata = SectionMetadata(
                section_type="table_of_contents",
                section_reference="Table of contents",
                section_title="Table of contents",
            )

        if index >= body_start:
            section_match = COMBINED_SECTION_PATTERN.fullmatch(line)

            if (
                section_match
                and expected_section_index < len(section_order)
                and section_match.group(1)
                == section_order[expected_section_index]
            ):
                section_number = section_match.group(1)

                current_metadata = SectionMetadata(
                    section_type="opinion_section",
                    section_reference=f"Section {section_number}",
                    section_title=section_titles[section_number],
                )

                expected_section_index += 1

        annotated_lines.append(
            annotate_line(record, current_metadata)
        )

    return annotated_lines


def parse_generic_document(
    records: list[LineRecord],
) -> list[AnnotatedLine]:
    metadata = SectionMetadata(
        section_type="document",
        section_reference="Full document",
        section_title=None,
    )

    return [
        annotate_line(record, metadata)
        for record in records
    ]


def parse_document_structure(
    document_name: str,
    pages: list[tuple[int, str]],
) -> list[AnnotatedLine]:
    records = create_line_records(pages)

    if not records:
        return []

    if document_name in LEGAL_DOCUMENT_NAMES:
        return parse_legal_document(records)

    if document_name == EDPB_DOCUMENT_NAME:
        return parse_edpb_document(records)

    return parse_generic_document(records)


def summarize_structure(
    annotated_lines: list[AnnotatedLine],
) -> dict[str, set[str]]:
    summary: dict[str, set[str]] = {}

    for line in annotated_lines:
        summary.setdefault(line.section_type, set()).add(
            line.section_reference
        )

    return summary
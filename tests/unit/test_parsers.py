"""Tests for format-specific parsers."""

from pathlib import Path

from rag_pipeline.data.parsers import (
    CSVParser,
    DOCXParser,
    HTMLParser,
    MarkdownParser,
    PDFParser,
    TXTParser,
    get_parser,
)

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"


def test_pdf_parser():
    parsed = PDFParser().parse(FIXTURES / "sample.pdf")
    assert "Sample PDF" in parsed.content
    assert parsed.metadata["format"] == "pdf"


def test_docx_parser():
    parsed = DOCXParser().parse(FIXTURES / "sample.docx")
    assert "Sample DOCX" in parsed.content
    assert len(parsed.tables) == 1
    assert parsed.metadata["format"] == "docx"


def test_txt_parser():
    parsed = TXTParser().parse(FIXTURES / "sample.txt")
    assert "sample text file" in parsed.content
    assert parsed.metadata["format"] == "txt"


def test_markdown_parser():
    parsed = MarkdownParser().parse(FIXTURES / "sample.md")
    assert "# Sample Document" in parsed.content
    assert len(parsed.sections) >= 3
    assert parsed.sections[0].title == "Sample Document"
    assert parsed.sections[0].level == 1


def test_html_parser():
    parsed = HTMLParser().parse(FIXTURES / "sample.html")
    assert "Sample HTML" in parsed.content
    assert "skip this" not in parsed.content.lower()  # script removed
    assert "Skip this footer" not in parsed.content  # footer removed
    assert parsed.metadata["title"] == "Sample HTML Page"


def test_csv_parser():
    parsed = CSVParser().parse(FIXTURES / "sample.csv")
    assert "FastAPI" in parsed.content
    assert parsed.metadata["row_count"] == 5


def test_get_parser():
    assert isinstance(get_parser(FIXTURES / "sample.pdf"), PDFParser)
    assert isinstance(get_parser(FIXTURES / "sample.docx"), DOCXParser)
    assert isinstance(get_parser(FIXTURES / "sample.txt"), TXTParser)
    assert isinstance(get_parser(FIXTURES / "sample.md"), MarkdownParser)
    assert isinstance(get_parser(FIXTURES / "sample.html"), HTMLParser)
    assert isinstance(get_parser(FIXTURES / "sample.csv"), CSVParser)


def test_get_parser_unknown():
    import pytest

    with pytest.raises(ValueError, match="No parser"):
        get_parser(FIXTURES / "sample.xyz")

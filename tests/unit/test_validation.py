"""Tests for file validation."""

from pathlib import Path

from rag_pipeline.data.validation import compute_hash, detect_mime, validate_file

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"


def test_validate_txt():
    result = validate_file(FIXTURES / "sample.txt")
    assert result.valid
    assert result.file_hash is not None


def test_validate_pdf():
    result = validate_file(FIXTURES / "sample.pdf")
    assert result.valid


def test_validate_docx():
    result = validate_file(FIXTURES / "sample.docx")
    assert result.valid


def test_validate_md():
    result = validate_file(FIXTURES / "sample.md")
    assert result.valid


def test_validate_html():
    result = validate_file(FIXTURES / "sample.html")
    assert result.valid


def test_validate_csv():
    result = validate_file(FIXTURES / "sample.csv")
    assert result.valid


def test_validate_not_found():
    result = validate_file(Path("/nonexistent/file.txt"))
    assert not result.valid
    assert "not found" in result.error.lower()


def test_validate_empty_file(tmp_path: Path):
    empty = tmp_path / "empty.txt"
    empty.write_text("")
    result = validate_file(empty)
    assert not result.valid
    assert "empty" in result.error.lower()


def test_validate_unsupported_format(tmp_path: Path):
    exe = tmp_path / "script.exe"
    exe.write_text("binary")
    result = validate_file(exe)
    assert not result.valid
    assert "unsupported" in result.error.lower()


def test_compute_hash_deterministic():
    h1 = compute_hash(FIXTURES / "sample.txt")
    h2 = compute_hash(FIXTURES / "sample.txt")
    assert h1 == h2
    assert len(h1) == 64  # SHA-256 hex


def test_detect_mime():
    assert detect_mime(FIXTURES / "sample.pdf") == "application/pdf"
    assert detect_mime(FIXTURES / "sample.docx") == "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    assert detect_mime(FIXTURES / "sample.txt") == "text/plain"
    assert detect_mime(FIXTURES / "sample.md") == "text/markdown"
    assert detect_mime(FIXTURES / "sample.html") == "text/html"
    assert detect_mime(FIXTURES / "sample.csv") == "text/csv"

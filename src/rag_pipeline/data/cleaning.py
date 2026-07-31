"""Text cleaning pipeline — normalize text before chunking."""

from __future__ import annotations

import html as html_module
import re
import unicodedata
from dataclasses import dataclass

import ftfy
from bs4 import BeautifulSoup

# ---- Individual cleaners -----------------------------------------------------


def fix_encoding(text: str) -> str:
    """Fix mojibake and encoding artifacts using ftfy."""
    return ftfy.fix_text(text)


def normalize_unicode(text: str) -> str:
    """Normalize unicode to NFKC form (fullwidth→ASCII, ligatures→letters)."""
    return unicodedata.normalize("NFKC", text)


def decode_html_entities(text: str) -> str:
    """Decode HTML entities: &amp; → &, &lt; → <, &#123; → {."""
    return html_module.unescape(text)


def remove_control_chars(text: str) -> str:
    """Remove non-printable control characters (keep newline, tab, carriage return)."""
    return "".join(
        ch for ch in text
        if ch in ("\n", "\t", "\r") or not unicodedata.category(ch).startswith("C")
    )


def normalize_whitespace(text: str) -> str:
    """Collapse multiple spaces to one, strip trailing whitespace per line."""
    lines = text.split("\n")
    lines = [re.sub(r"[^\S\n]+", " ", line).rstrip() for line in lines]
    return "\n".join(lines)


def collapse_blank_lines(text: str, max_blank: int = 1) -> str:
    """Replace 2+ consecutive blank lines with max_blank blank lines."""
    pattern = re.compile(rf"(\n{{{max_blank + 1},}})")
    replacement = "\n" * (max_blank + 1)
    return pattern.sub(replacement, text)


def clean_pdf_artifacts(text: str) -> str:
    """Fix common PDF extraction artifacts."""
    # Fix hyphenated line breaks: "multi-\nline" → "multiline"
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    # Fix broken words across lines (word at end, lowercase start next line)
    return re.sub(r"(\w)\n([a-z])", r"\1\2", text)


def strip_residual_html(text: str) -> str:
    """Strip leftover HTML tags that parsers didn't remove."""
    soup = BeautifulSoup(text, "html.parser")
    return soup.get_text(separator=" ")


# ---- Cleaning pipeline -------------------------------------------------------


@dataclass
class CleaningStats:
    """Metrics from a cleaning pass."""

    chars_before: int = 0
    chars_after: int = 0
    lines_before: int = 0
    lines_after: int = 0

    @property
    def chars_removed(self) -> int:
        return self.chars_before - self.chars_after

    @property
    def lines_removed(self) -> int:
        return self.lines_before - self.lines_after


@dataclass
class CleaningConfig:
    """Which cleaners to apply."""

    fix_encoding: bool = True
    normalize_unicode: bool = True
    decode_html_entities: bool = True
    remove_control_chars: bool = True
    normalize_whitespace: bool = True
    collapse_blank_lines: bool = True
    max_blank_lines: int = 1
    clean_pdf_artifacts: bool = True
    strip_residual_html: bool = True


class TextCleaner:
    """Configurable text cleaning pipeline."""

    def __init__(self, config: CleaningConfig | None = None) -> None:
        self.config = config or CleaningConfig()

    def clean(self, text: str, text_format: str = "generic") -> tuple[str, CleaningStats]:
        """Apply cleaning pipeline to text.

        Returns cleaned text and cleaning statistics.
        """
        stats = CleaningStats(
            chars_before=len(text),
            lines_before=text.count("\n") + 1,
        )

        # 1. Always apply (format-agnostic)
        if self.config.fix_encoding:
            text = fix_encoding(text)
        if self.config.decode_html_entities:
            text = decode_html_entities(text)
        if self.config.normalize_unicode:
            text = normalize_unicode(text)
        if self.config.remove_control_chars:
            text = remove_control_chars(text)
        if self.config.normalize_whitespace:
            text = normalize_whitespace(text)
        if self.config.collapse_blank_lines:
            text = collapse_blank_lines(text, self.config.max_blank_lines)

        # 2. Format-specific
        if text_format == "pdf" and self.config.clean_pdf_artifacts:
            text = clean_pdf_artifacts(text)
        if text_format in ("html", "markdown") and self.config.strip_residual_html:
            text = strip_residual_html(text)

        stats.chars_after = len(text)
        stats.lines_after = text.count("\n") + 1

        return text, stats

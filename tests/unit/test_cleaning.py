"""Tests for text cleaning pipeline."""

from rag_pipeline.data.cleaning import (
    CleaningConfig,
    TextCleaner,
    clean_pdf_artifacts,
    collapse_blank_lines,
    decode_html_entities,
    fix_encoding,
    normalize_unicode,
    normalize_whitespace,
    remove_control_chars,
    strip_residual_html,
)


class TestIndividualCleaners:
    def test_fix_encoding_mojibake(self):
        assert fix_encoding("\u00e2\u0080\u0099") == '\''

    def test_fix_encoding_already_clean(self):
        assert fix_encoding("hello world") == "hello world"

    def test_normalize_unicode_fullwidth(self):
        assert normalize_unicode("\uff28\uff45\uff4c\uff4c\uff4f") == "Hello"

    def test_normalize_unicode_ligature(self):
        assert normalize_unicode("file") == "file"

    def test_normalize_unicode_already_nfkc(self):
        assert normalize_unicode("hello") == "hello"

    def test_decode_html_entities_amp(self):
        assert decode_html_entities("&amp;") == "&"

    def test_decode_html_entities_lt(self):
        assert decode_html_entities("&lt;div&gt;") == "<div>"

    def test_decode_html_entities_numeric(self):
        assert decode_html_entities("&#65;") == "A"

    def test_decode_html_entities_mixed(self):
        assert decode_html_entities("a &amp; b &lt; c") == "a & b < c"

    def test_remove_control_chars(self):
        assert remove_control_chars("hello\x00\x01\x02world") == "helloworld"

    def test_remove_control_chars_keeps_newline_tab(self):
        assert remove_control_chars("line1\nline2\ttab") == "line1\nline2\ttab"

    def test_remove_control_chars_keeps_regular_text(self):
        assert remove_control_chars("Hello, World! 123") == "Hello, World! 123"

    def test_normalize_whitespace(self):
        assert normalize_whitespace("hello  world") == "hello world"

    def test_normalize_whitespace_trailing(self):
        assert normalize_whitespace("hello   ") == "hello"

    def test_normalize_whitespace_preserves_newlines(self):
        assert normalize_whitespace("line1\nline2") == "line1\nline2"

    def test_collapse_blank_lines(self):
        text = "a\n\n\n\n\nb"
        assert collapse_blank_lines(text) == "a\n\nb"

    def test_collapse_blank_lines_single(self):
        text = "a\n\nb"
        assert collapse_blank_lines(text) == "a\n\nb"

    def test_collapse_blank_lines_custom_max(self):
        text = "a\n\n\n\nb"
        assert collapse_blank_lines(text, max_blank=2) == "a\n\n\nb"

    def test_clean_pdf_artifacts_hyphenated(self):
        assert clean_pdf_artifacts("multi-\nline") == "multiline"

    def test_clean_pdf_artifacts_broken_word(self):
        assert clean_pdf_artifacts("hel\nlo") == "hello"

    def test_clean_pdf_artifacts_no_change(self):
        assert clean_pdf_artifacts("hello world") == "hello world"

    def test_strip_residual_html(self):
        assert strip_residual_html("<p>hello</p>") == "hello"

    def test_strip_residual_html_nested(self):
        assert strip_residual_html("<div><span>text</span></div>") == "text"

    def test_strip_residual_html_entities(self):
        result = strip_residual_html("<p>a &amp; b</p>")
        assert "a" in result
        assert "b" in result


class TestTextCleaner:
    def test_clean_generic(self):
        cleaner = TextCleaner()
        text = "hello  world"
        cleaned, stats = cleaner.clean(text)
        assert cleaned == "hello world"
        assert stats.chars_before == 12
        assert stats.chars_after == 11

    def test_clean_with_encoding_issues(self):
        cleaner = TextCleaner()
        text = "hello  \x00  world"
        cleaned, _ = cleaner.clean(text)
        assert "\x00" not in cleaned
        assert "hello" in cleaned
        assert "world" in cleaned

    def test_clean_pdf_format(self):
        cleaner = TextCleaner()
        text = "multi-\nline"
        cleaned, _ = cleaner.clean(text, text_format="pdf")
        assert "multiline" in cleaned

    def test_clean_html_format(self):
        cleaner = TextCleaner()
        text = "<p>hello</p>"
        cleaned, _ = cleaner.clean(text, text_format="html")
        assert "hello" in cleaned

    def test_clean_markdown_format(self):
        cleaner = TextCleaner()
        text = "<div>hello</div>"
        cleaned, _ = cleaner.clean(text, text_format="markdown")
        assert "hello" in cleaned

    def test_clean_empty_string(self):
        cleaner = TextCleaner()
        cleaned, stats = cleaner.clean("")
        assert cleaned == ""
        assert stats.chars_before == 0
        assert stats.chars_after == 0

    def test_clean_whitespace_only(self):
        cleaner = TextCleaner()
        cleaned, _ = cleaner.clean("   \n\n   ")
        assert cleaned.strip() == ""

    def test_clean_preserves_structure(self):
        cleaner = TextCleaner()
        text = "## Heading\n\nParagraph 1.\n\nParagraph 2."
        cleaned, _ = cleaner.clean(text)
        assert "## Heading" in cleaned
        assert "Paragraph 1." in cleaned
        assert "Paragraph 2." in cleaned

    def test_clean_disabled_steps(self):
        config = CleaningConfig(
            fix_encoding=False,
            normalize_unicode=False,
            remove_control_chars=False,
        )
        cleaner = TextCleaner(config)
        text = "hello\x00world"
        cleaned, _ = cleaner.clean(text)
        assert "\x00" in cleaned

    def test_clean_stats(self):
        cleaner = TextCleaner()
        text = "hello  \x00  world\n\n\n\nend"
        _cleaned, stats = cleaner.clean(text)
        assert stats.chars_before > 0
        assert stats.lines_before >= 1
        assert stats.chars_removed >= 0

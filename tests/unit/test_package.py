"""Smoke test — verify package imports and version."""


def test_package_imports():
    import rag_pipeline

    assert hasattr(rag_pipeline, "__version__")


def test_version_is_string():
    from rag_pipeline import __version__

    assert isinstance(__version__, str)
    assert __version__ == "0.1.0"

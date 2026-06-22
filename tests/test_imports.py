"""Verify every package is importable."""


def test_import_core() -> None:
    import core  # noqa: F401


def test_import_ingest() -> None:
    import ingest  # noqa: F401


def test_import_retrieval() -> None:
    import retrieval  # noqa: F401


def test_import_qa() -> None:
    import qa  # noqa: F401


def test_import_eval() -> None:
    import eval  # noqa: F401


def test_import_api() -> None:
    import api  # noqa: F401


def test_all_packages_import() -> None:
    import api  # noqa: F401
    import core  # noqa: F401
    import eval  # noqa: F401
    import ingest  # noqa: F401
    import qa  # noqa: F401
    import retrieval  # noqa: F401

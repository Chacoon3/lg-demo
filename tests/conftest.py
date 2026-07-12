import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--manual",
        action="store_true",
        default=False,
        help="Run tests marked as manual.",
    )


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "manual: marks a test as manual")


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if config.getoption("--manual"):
        return

    skip_manual = pytest.mark.skip(reason="need --manual option to run")
    for item in items:
        if "manual" in item.keywords:
            item.add_marker(skip_manual)

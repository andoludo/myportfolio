from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def bear_db() -> Path:
    return Path(__file__).parent / "data" / "bear_portfolio.db"

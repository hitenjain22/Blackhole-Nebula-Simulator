"""Shared test fixtures. Slow tests are marked ``@pytest.mark.slow`` and skipped by default
(pyproject addopts "-m 'not slow'"); run them with ``uv run pytest -s -m slow``."""

import numpy as np
import pytest

SEED = 20261014


@pytest.fixture
def rng() -> np.random.Generator:
    """A fixed-seed generator, fresh for each test, so every test run is identical."""
    return np.random.default_rng(SEED)


@pytest.fixture(autouse=True)
def _table_row_on_new_line():
    """With -s, pytest prints the test file name without a newline; start our rows below it."""
    print()
    yield

# Tests

- Run: `uv run pytest -s` (prints the comparison table). Slow tests: `uv run pytest -s -m slow`.
- Every numeric comparison goes through `tests/helpers.py::check`.
- Expected values come from [`reference/expected-values.md`](../reference/expected-values.md) and are cited by section (e.g. "§2 L"). They are requirements: never change one or loosen a tolerance to make a test pass (CLAUDE.md rule 3).
- Layout mirrors `src/nebula/` as the code grows (e.g. `tests/physics/test_gravity.py`).

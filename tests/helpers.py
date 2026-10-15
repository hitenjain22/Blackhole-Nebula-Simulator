"""The one way tests compare numbers (Phase 0, Step 0.6).

Every test calls ``check``, so ``uv run pytest -s`` prints a table of
name | expected | actual | tolerance | PASS/FAIL, which is what step reports quote
(CLAUDE.md, workflow 3). Expected values come from reference/expected-values.md; cite the
section in the check name or the test docstring (e.g. "§2 L").
"""

import numpy as np

_header_printed = False
NAME_WIDTH = 44


def _fmt(value) -> str:
    if isinstance(value, np.ndarray) and value.size > 1:
        return f"array{value.shape}"
    value = np.asarray(value).item() if isinstance(value, np.ndarray) else value
    if isinstance(value, bool | np.bool_):
        return str(bool(value))
    if isinstance(value, int | np.integer):
        return f"{value:,}"
    if isinstance(value, float | np.floating):
        return f"{value:.12g}"
    return str(value)


def _tol_text(rtol, atol) -> str:
    parts = []
    if rtol is not None:
        parts.append(f"rtol {rtol:g}")
    if atol is not None:
        parts.append(f"atol {atol:g}")
    return ", ".join(parts) if parts else "exact"


def _print_header() -> None:
    global _header_printed
    if not _header_printed:
        print(
            f"\n{'check':{NAME_WIDTH}s} | {'expected':>18s} | {'actual':>18s} | tolerance  | result"
        )
        print("-" * (NAME_WIDTH + 70))
        _header_printed = True


def check(name: str, expected, actual, rtol: float | None = None, atol: float | None = None):
    """Print one table row and assert that ``actual`` matches ``expected``.

    Passes when |actual - expected| <= atol + rtol * |expected| for every element (missing
    tolerances count as 0). With neither tolerance given, values must be exactly equal; that
    is also how non-numeric values (strings, booleans) are compared. Arrays report their
    worst element.
    """
    _print_header()
    numeric = rtol is not None or atol is not None
    extra = ""
    if numeric:
        e = np.asarray(expected, dtype=np.float64)
        a = np.asarray(actual, dtype=np.float64)
        if e.shape != a.shape and e.size != 1:
            raise ValueError(f"{name}: shape mismatch {e.shape} vs {a.shape}")
        err = np.abs(a - e)
        limit = (atol or 0.0) + (rtol or 0.0) * np.abs(e)
        ok = bool(np.all(err <= limit)) and not np.any(np.isnan(a))
        if a.size > 1:
            worst = int(np.argmax(err - limit))
            with np.errstate(divide="ignore", invalid="ignore"):
                rel = np.max(err / np.abs(e)) if np.all(e != 0) else np.nan
            extra = f"  (max abs err {err.max():.3g}, max rel err {rel:.3g}, worst index {worst})"
    else:
        ok = bool(np.all(np.asarray(expected == actual)))
    verdict = "PASS" if ok else "FAIL"
    print(
        f"{name:{NAME_WIDTH}s} | {_fmt(expected):>18s} | {_fmt(actual):>18s} | "
        f"{_tol_text(rtol, atol):10s} | {verdict}{extra}"
    )
    assert ok, f"{name}: expected {expected!r}, got {actual!r} ({_tol_text(rtol, atol)})"

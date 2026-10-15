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


def _num(x: float) -> str:
    return f"{x:.0e}" if x < 0.01 else f"{x:g}"


def _tol_text(rtol, atol) -> str:
    parts = []
    if rtol is not None:
        parts.append(f"rtol {_num(rtol)}")
    if atol is not None:
        parts.append(f"atol {_num(atol)}")
    return ", ".join(parts) if parts else "exact"


def _print_header() -> None:
    global _header_printed
    if not _header_printed:
        print(
            f"\n{'check':{NAME_WIDTH}s} | {'expected':>18s} | {'actual':>18s} | tolerance  | result"
        )
        print("-" * (NAME_WIDTH + 70))
        _header_printed = True


def check(
    name: str,
    expected,
    actual,
    rtol: float | None = None,
    atol: float | None = None,
    per_vector: bool = False,
):
    """Print one table row and assert that ``actual`` matches ``expected``.

    Passes when |actual - expected| <= atol + rtol * |expected| for every element (missing
    tolerances count as 0). With neither tolerance given, values must be exactly equal; that
    is also how non-numeric values (strings, booleans) are compared. Arrays report their
    worst element.

    ``per_vector=True`` is for arrays of 3-vectors (last axis): |.| is then the vector length,
    so the error of each vector is measured against that vector's size, not against each of
    its components (a component that is nearly zero only because of how the axes happen to
    point would otherwise dominate; see docs/DECISIONS.md, Step 1.2).
    """
    _print_header()
    numeric = rtol is not None or atol is not None
    extra = ""
    if numeric:
        e = np.asarray(expected, dtype=np.float64)
        a = np.asarray(actual, dtype=np.float64)
        if e.shape != a.shape and e.size != 1:
            raise ValueError(f"{name}: shape mismatch {e.shape} vs {a.shape}")
        if per_vector:
            err = np.linalg.norm(a - e, axis=-1)
            size = np.linalg.norm(np.broadcast_to(e, a.shape), axis=-1)
        else:
            err = np.abs(a - e)
            size = np.abs(np.broadcast_to(e, a.shape))
        limit = (atol or 0.0) + (rtol or 0.0) * size
        ok = bool(np.all(err <= limit)) and not np.any(np.isnan(a))
        if a.size > 1:
            worst = int(np.argmax(err - limit))
            with np.errstate(divide="ignore", invalid="ignore"):
                rel = np.max(err / size) if np.all(size != 0) else np.nan
            extra = f"  (max abs err {err.max():.3g}, max rel err {rel:.3g}, worst index {worst})"
    else:
        ok = bool(np.all(np.asarray(expected == actual)))
    verdict = "PASS" if ok else "FAIL"
    if per_vector:
        extra = extra.replace("worst index", "per vector, worst vector")
    print(
        f"{name:{NAME_WIDTH}s} | {_fmt(expected):>18s} | {_fmt(actual):>18s} | "
        f"{_tol_text(rtol, atol):10s} | {verdict}{extra}"
    )
    assert ok, f"{name}: expected {expected!r}, got {actual!r} ({_tol_text(rtol, atol)})"

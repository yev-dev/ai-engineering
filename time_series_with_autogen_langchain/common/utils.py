"""
Date and string utility functions for the time-series pipeline.

Provides robust date-string parsing, validation, and normalisation so that
agents can handle any reasonable date format a user may supply without
crashing or producing ambiguous results.
"""

from __future__ import annotations

import re
from datetime import date, datetime, timedelta
from typing import Any, Callable

import pandas as pd

# ---------------------------------------------------------------------------
# Public domain constants
# ---------------------------------------------------------------------------

# The canonical format used internally by PipelineRequest
CANONICAL_DATE_FMT = "%Y-%m-%d"
CANONICAL_DATETIME_FMT = "%Y-%m-%dT%H:%M:%S"

# Default lower / upper bounds for sanity checks
DEFAULT_MIN_DATE = date(1900, 1, 1)
DEFAULT_MAX_DATE = date(2100, 12, 31)

# ---------------------------------------------------------------------------
# Tolerated input formats – ordered by (unambiguity, prevalence).
# The parser tries these in order and returns the first match.
# ---------------------------------------------------------------------------

_DATE_FORMATS: list[tuple[str, str, bool]] = [
    # (strptime_format, description, is_iso_like)
    ("%Y-%m-%d", "ISO 8601 (2024-01-15)", True),
    ("%Y/%m/%d", "ISO slash (2024/01/15)", True),
    ("%Y%m%d", "Compact ISO (20240115)", True),
    ("%Y-%m-%dT%H:%M:%S", "ISO datetime (2024-01-15T10:30:00)", True),
    ("%Y-%m-%d %H:%M:%S", "ISO datetime with space (2024-01-15 10:30:00)", True),
    ("%Y-%m-%dT%H:%M:%S.%f", "ISO with microseconds (2024-01-15T10:30:00.123456)", True),
    # US-style (month/day)
    ("%m/%d/%Y", "US slash (01/15/2024)", False),
    ("%m-%d-%Y", "US dash (01-15-2024)", False),
    ("%m.%d.%Y", "US dot (01.15.2024)", False),
    # European-style (day/month)
    ("%d/%m/%Y", "EU slash (15/01/2024)", False),
    ("%d-%m-%Y", "EU dash (15-01-2024)", False),
    ("%d.%m.%Y", "EU dot (15.01.2024)", False),
    # Human-readable with full month names
    ("%B %d, %Y", "Long US (January 15, 2024)", False),
    ("%B %d %Y", "Long US no comma (January 15 2024)", False),
    ("%d %B %Y", "Long EU (15 January 2024)", False),
    ("%b %d, %Y", "Abbr US (Jan 15, 2024)", False),
    ("%b %d %Y", "Abbr US no comma (Jan 15 2024)", False),
    ("%d %b %Y", "Abbr EU (15 Jan 2024)", False),
    # Short year variants (for Y2K assume 2000+)
    ("%m/%d/%y", "US short year (01/15/24)", False),
    ("%d/%m/%y", "EU short year (15/01/24)", False),
    ("%Y.%m.%d", "Dotted ISO (2024.01.15)", True),
]

# ---------------------------------------------------------------------------
# Relative date patterns
# ---------------------------------------------------------------------------

_RELATIVE_PATTERNS: list[tuple[re.Pattern, Callable[[re.Match], date]]] = [
    (re.compile(r"^today$", re.IGNORECASE), lambda _: date.today()),
    (re.compile(r"^now$", re.IGNORECASE), lambda _: date.today()),
    (re.compile(r"^yesterday$", re.IGNORECASE), lambda _: date.today() - timedelta(days=1)),
    (
        re.compile(r"^last\s+week$", re.IGNORECASE),
        lambda _: date.today() - timedelta(weeks=1),
    ),
    (
        re.compile(r"^last\s+month$", re.IGNORECASE),
        lambda _: _shift_months(date.today(), -1),
    ),
    (
        re.compile(r"^last\s+quarter$", re.IGNORECASE),
        lambda _: _shift_months(date.today(), -3),
    ),
    (
        re.compile(r"^last\s+year$", re.IGNORECASE),
        lambda _: _shift_months(date.today(), -12),
    ),
    (
        re.compile(r"^(\d+)\s+(day|days?)\s+ago$", re.IGNORECASE),
        lambda m: date.today() - timedelta(days=int(m.group(1))),
    ),
    (
        re.compile(r"^(\d+)\s+(week|weeks?)\s+ago$", re.IGNORECASE),
        lambda m: date.today() - timedelta(weeks=int(m.group(1))),
    ),
    (
        re.compile(r"^(\d+)\s+(month|months?)\s+ago$", re.IGNORECASE),
        lambda m: _shift_months(date.today(), -int(m.group(1))),
    ),
    (
        re.compile(r"^(\d+)\s+(year|years?)\s+ago$", re.IGNORECASE),
        lambda m: _shift_months(date.today(), -int(m.group(1)) * 12),
    ),
]

# ---------------------------------------------------------------------------
# Quarter / semi-annual patterns
# ---------------------------------------------------------------------------

_QUARTER_PATTERN = re.compile(
    r"^(?P<year>\d{4})\s*[-/]?\s*[Qq](?P<q>[1-4])$"
)
_QUARTER_REVERSE = re.compile(
    r"^[Qq](?P<q>[1-4])\s*[-/]?\s*(?P<year>\d{4})$"
)
_YEAR_PATTERN = re.compile(r"^\d{4}$")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _shift_months(d: date, months: int) -> date:
    """Add *months* (possibly negative) to *d*, clamping day to month max."""
    total = d.year * 12 + (d.month - 1) + months
    year = total // 12
    month = (total % 12) + 1
    import calendar
    max_day = calendar.monthrange(year, month)[1]
    return date(year, month, min(d.day, max_day))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def parse_date_str(
    raw: str,
    *,
    allow_relative: bool = True,
    allow_quarter: bool = True,
    allow_year_only: bool = False,
    prefer_eu: bool | None = None,
    min_date: date = DEFAULT_MIN_DATE,
    max_date: date = DEFAULT_MAX_DATE,
) -> date:
    """Parse *raw* into a ``datetime.date``, raising ``ValueError`` on failure.

    Parameters
    ----------
    raw : str
        The input string to parse.
    allow_relative : bool
        Accept relative expressions (``“today”``, ``“last month”``, …).
    allow_quarter : bool
        Accept quarter designators (``“2024-Q1”``, ``“Q1 2024”``).
    allow_year_only : bool
        Accept bare 4-digit years (return January 1 of that year).
    prefer_eu : bool or None
        - ``True``  → try EU (DD/MM) before US (MM/DD) for ambiguous formats.
        - ``False`` → try US before EU.
        - ``None``  → auto-detect: if the first component > 12 assume EU,
          otherwise try US first.
    min_date, max_date : date
        Bounds for sanity checking.

    Returns
    -------
    date

    Raises
    ------
    ValueError
        If *raw* cannot be interpreted or the result is out of bounds.
    """
    raw = raw.strip()
    if not raw:
        raise ValueError("Empty date string.")

    # 1) Relative expressions
    if allow_relative:
        for pattern, func in _RELATIVE_PATTERNS:
            m = pattern.match(raw)
            if m:
                result = func(m)
                _check_bounds(result, min_date, max_date)
                return result

    # 2) Quarter designators
    if allow_quarter:
        result = _try_parse_quarter(raw)
        if result is not None:
            _check_bounds(result, min_date, max_date)
            return result

    # 3) Bare year
    if allow_year_only and _YEAR_PATTERN.match(raw):
        result = date(int(raw), 1, 1)
        _check_bounds(result, min_date, max_date)
        return result

    # 4) strptime-based formats
    result = _try_strptime_formats(raw, prefer_eu)
    if result is not None:
        _check_bounds(result, min_date, max_date)
        return result

    raise ValueError(
        f"Cannot parse date string {raw!r}. "
        f"Try ISO (2024-01-15), US (01/15/2024), EU (15/01/2024), "
        f"human-readable (Jan 15, 2024), or relative (today, last month)."
    )


def _try_strptime_formats(raw: str, prefer_eu: bool | None) -> date | None:
    """Iterate *raw* against known formats.

    When *prefer_eu* is ``None`` the function attempts auto-detection for
    slash/dash/dot separated formats that could be either US or EU.
    """
    candidates: list[tuple[str, str, bool]] = list(_DATE_FORMATS)

    # Re-order ambiguous formats based on preference
    ambiguous_keys = ["US slash", "US dash", "US dot", "EU slash", "EU dash", "EU dot"]
    ambiguous = [(fmt, desc, iso) for fmt, desc, iso in candidates if any(k in desc for k in ambiguous_keys)]
    fixed = [(fmt, desc, iso) for fmt, desc, iso in candidates if fmt not in {p[0] for p in ambiguous}]

    if prefer_eu is True:
        # EU first, US second for each pair
        ordered: list[tuple[str, str, bool]] = []
        for eu_key, us_key in zip(
            ["EU slash", "EU dash", "EU dot"],
            ["US slash", "US dash", "US dot"],
        ):
            eu = next((f, d, i) for f, d, i in ambiguous if us_key in d)  # match EU variants
            us = next((f, d, i) for f, d, i in ambiguous if us_key in d.replace("EU", "US"))
            ordered.append(eu)
            ordered.append(us)
        candidates = fixed + ordered
    elif prefer_eu is False:
        # US first
        ordered = []
        for us_key, eu_key in zip(
            ["US slash", "US dash", "US dot"],
            ["EU slash", "EU dash", "EU dot"],
        ):
            us = next((f, d, i) for f, d, i in ambiguous if us_key in d)
            ordered.append(us)
            ordered.append(eu)
        candidates = fixed + ordered
    else:
        # Auto-detect: check first numeric component
        first_num = _first_numeric_component(raw)
        if first_num is not None and first_num > 12:
            # Day > 12 must be EU
            candidates = fixed + sorted(ambiguous, key=lambda x: (0 if "EU" in x[1] else 1))
        else:
            # Conservative: try ISO-likes first, then US, then EU
            candidates = fixed + sorted(ambiguous, key=lambda x: (0 if x[2] else (1 if "US" in x[1] else 2)))

    for fmt, _desc, _iso in candidates:
        try:
            parsed = datetime.strptime(raw, fmt)
            # Reject year 1900 artefacts from strptime fallback
            if parsed.year <= 1900:
                continue
            return parsed.date()
        except ValueError:
            continue
    return None


def _first_numeric_component(raw: str) -> int | None:
    """Return the first integer in a delimited date string, or None."""
    parts = re.split(r"[-/. ]", raw)
    for p in parts:
        p = p.strip()
        if p.isdigit():
            return int(p)
    return None


def _try_parse_quarter(raw: str) -> date | None:
    """Try to interpret *raw* as a quarter designator.

    Returns the first day of the quarter, or None.
    """
    m = _QUARTER_PATTERN.match(raw)
    if not m:
        m = _QUARTER_REVERSE.match(raw)
    if m:
        year = int(m.group("year"))
        q = int(m.group("q"))
        month = (q - 1) * 3 + 1
        return date(year, month, 1)
    return None


def _check_bounds(
    d: date,
    min_date: date = DEFAULT_MIN_DATE,
    max_date: date = DEFAULT_MAX_DATE,
) -> None:
    """Raise ``ValueError`` if *d* is outside [min_date, max_date]."""
    if d < min_date:
        raise ValueError(
            f"Date {d.isoformat()} is before the minimum allowed ({min_date.isoformat()})."
        )
    if d > max_date:
        raise ValueError(
            f"Date {d.isoformat()} is after the maximum allowed ({max_date.isoformat()})."
        )


# ---------------------------------------------------------------------------
# Batched / convenience helpers
# ---------------------------------------------------------------------------


def parse_date_pair(
    start_raw: str,
    end_raw: str,
    *,
    allow_relative: bool = True,
    allow_quarter: bool = True,
    allow_year_only: bool = False,
    prefer_eu: bool | None = None,
    min_date: date = DEFAULT_MIN_DATE,
    max_date: date = DEFAULT_MAX_DATE,
    require_ordered: bool = True,
    max_range_days: int | None = None,
) -> tuple[date, date]:
    """Parse and validate a start/end date pair.

    Parameters
    ----------
    start_raw, end_raw : str
        Date strings as supplied by the user.
    require_ordered : bool
        Raise if start > end.
    max_range_days : int or None
        Maximum allowed span in days.

    Returns
    -------
    (start_date, end_date)

    Raises
    ------
    ValueError
        If either date is invalid or the pair fails validation.
    """
    start = parse_date_str(
        start_raw,
        allow_relative=allow_relative,
        allow_quarter=allow_quarter,
        allow_year_only=allow_year_only,
        prefer_eu=prefer_eu,
        min_date=min_date,
        max_date=max_date,
    )
    end = parse_date_str(
        end_raw,
        allow_relative=allow_relative,
        allow_quarter=allow_quarter,
        allow_year_only=allow_year_only,
        prefer_eu=prefer_eu,
        min_date=min_date,
        max_date=max_date,
    )

    if require_ordered and start > end:
        raise ValueError(
            f"Start date ({start.isoformat()}) must not be after end date ({end.isoformat()})."
        )

    if max_range_days is not None:
        span = (end - start).days
        if span > max_range_days:
            raise ValueError(
                f"Date range of {span} days exceeds maximum allowed ({max_range_days})."
            )

    return (start, end)


def format_date(d: date | datetime, fmt: str = CANONICAL_DATE_FMT) -> str:
    """Format a date/datetime to the canonical string representation."""
    return d.strftime(fmt)


def to_pandas_timestamp(raw: str) -> pd.Timestamp:
    """Convert a parsed date string directly to a pandas Timestamp.

    Uses ``parse_date_str`` internally, so it supports all the same formats.
    """
    d = parse_date_str(raw)
    return pd.Timestamp(d)


def is_valid_date_str(
    raw: str,
    *,
    allow_relative: bool = True,
    allow_quarter: bool = True,
    allow_year_only: bool = False,
    prefer_eu: bool | None = None,
    min_date: date = DEFAULT_MIN_DATE,
    max_date: date = DEFAULT_MAX_DATE,
) -> bool:
    """Return True if *raw* can be parsed as a valid date (no exception)."""
    try:
        parse_date_str(
            raw,
            allow_relative=allow_relative,
            allow_quarter=allow_quarter,
            allow_year_only=allow_year_only,
            prefer_eu=prefer_eu,
            min_date=min_date,
            max_date=max_date,
        )
        return True
    except ValueError:
        return False
"""Optional per-column anonymizers.

Every anonymizer maps each *unique* original value to exactly one replacement
(consistent mapping), so repeated values stay repeated and the joint
distribution of the column — the statistical variety this project exists to
preserve — survives anonymization. Missing values (NaN) are left untouched.

Available kinds (see :data:`KINDS` for aliases):

- ``names`` — replace values with realistic full names drawn from a bundled
  library of first/middle/last names (:class:`NameAnonymizer`).
- ``sequential_id`` — replace values with ``start, start+interval, ...`` in
  order of first appearance (:class:`SequentialIdAnonymizer`).
- ``numeric_jitter`` — replace numbers with a random value within ±20 % of
  the original by default (:class:`NumericJitterAnonymizer`).
- ``random_string`` / ``hex`` — replace values with random character
  sequences or hexadecimal strings (:class:`RandomStringAnonymizer`).
"""

from __future__ import annotations

import random
import string
from abc import ABC, abstractmethod
from typing import Any, Hashable, Mapping, Sequence

import pandas as pd

from . import _names
from ._logging import get_logger

log = get_logger(__name__)

AnonymizerSpec = "ColumnAnonymizer | str | tuple[str, dict] | dict"


class ColumnAnonymizer(ABC):
    """Base class: builds a unique-value → replacement mapping per column."""

    def transform(self, series: pd.Series, rng: random.Random | None = None) -> pd.Series:
        """Return a copy of ``series`` with all non-null values replaced."""
        rng = rng if rng is not None else random.Random()
        uniques = pd.unique(series.dropna())
        mapping = self.build_mapping(list(uniques), rng)
        log.debug(
            "%s: column %r — %d unique values mapped",
            type(self).__name__, series.name, len(mapping),
        )
        result = series.map(mapping)
        return self._restore_dtype(series, result)

    def _restore_dtype(self, original: pd.Series, result: pd.Series) -> pd.Series:
        return result

    @abstractmethod
    def build_mapping(
        self, uniques: Sequence[Hashable], rng: random.Random
    ) -> dict[Hashable, Any]:
        """Map each unique original value to its replacement."""


def _fresh(seen: set, generate, rng: random.Random, tries: int = 100) -> str:
    """Draw from ``generate`` until unseen; fall back to a numbered suffix."""
    for _ in range(tries):
        candidate = generate(rng)
        if candidate not in seen:
            return candidate
    base = generate(rng)
    i = 2
    while f"{base} {i}" in seen:
        i += 1
    return f"{base} {i}"


class NameAnonymizer(ColumnAnonymizer):
    """Replace values with realistic names from the bundled name library.

    ``style`` is one of ``first``, ``last``, ``first_last`` (default),
    ``first_middle_last``, ``last_first``. Replacements are unique within the
    column; if the requested style cannot supply enough distinct
    combinations, middle names are added automatically.
    """

    STYLES = ("first", "last", "first_last", "first_middle_last", "last_first")

    def __init__(self, style: str = "first_last"):
        if style not in self.STYLES:
            raise ValueError(
                f"Unknown name style {style!r}. Choose from {self.STYLES}."
            )
        self.style = style

    @staticmethod
    def _capacity(style: str) -> int:
        f, m, last = (
            len(_names.FIRST_NAMES),
            len(_names.MIDDLE_NAMES),
            len(_names.LAST_NAMES),
        )
        return {
            "first": f,
            "last": last,
            "first_last": f * last,
            "last_first": f * last,
            "first_middle_last": f * m * last,
        }[style]

    def _generate(self, style: str, rng: random.Random) -> str:
        first = rng.choice(_names.FIRST_NAMES)
        middle = rng.choice(_names.MIDDLE_NAMES)
        last = rng.choice(_names.LAST_NAMES)
        return {
            "first": first,
            "last": last,
            "first_last": f"{first} {last}",
            "first_middle_last": f"{first} {middle} {last}",
            "last_first": f"{last}, {first}",
        }[style]

    def build_mapping(self, uniques, rng):
        style = self.style
        # keep the collision rate low: escalate once past half capacity
        if len(uniques) > self._capacity(style) // 2:
            style = "first_middle_last"
            log.debug(
                "NameAnonymizer: %d values exceed half capacity of %r; "
                "using first_middle_last", len(uniques), self.style,
            )
        seen: set[str] = set()
        mapping: dict[Hashable, Any] = {}
        for value in uniques:
            name = _fresh(seen, lambda r: self._generate(style, r), rng)
            seen.add(name)
            mapping[value] = name
        return mapping


class SequentialIdAnonymizer(ColumnAnonymizer):
    """Replace values with sequential IDs: ``start, start+interval, ...``.

    IDs are assigned in order of first appearance. With a ``prefix`` or
    nonzero ``width`` the output is a zero-padded string (e.g. ``CUST-00120``);
    otherwise it stays numeric.
    """

    def __init__(self, start: int = 1, interval: int = 1, prefix: str = "", width: int = 0):
        if interval == 0:
            raise ValueError("interval must be nonzero")
        self.start = int(start)
        self.interval = int(interval)
        self.prefix = str(prefix)
        self.width = int(width)

    def build_mapping(self, uniques, rng):
        mapping = {}
        current = self.start
        for value in uniques:
            if self.prefix or self.width:
                mapping[value] = f"{self.prefix}{current:0{self.width}d}"
            else:
                mapping[value] = current
            current += self.interval
        return mapping

    def _restore_dtype(self, original, result):
        if not self.prefix and not self.width:
            if result.isna().any():
                return result.astype("Int64")
            return result.astype("int64")
        return result


class NumericJitterAnonymizer(ColumnAnonymizer):
    """Replace numbers with a random value within ±``pct`` of the original.

    Defaults to ±20 %. Integer columns stay integers; ``round_to`` rounds
    floats to that many decimal places. Note: a value of exactly 0 has no
    magnitude to jitter and is returned unchanged.
    """

    def __init__(self, pct: float = 0.20, round_to: int | None = None):
        if not 0 < pct < 1:
            raise ValueError("pct must be between 0 and 1 (e.g. 0.2 for ±20%)")
        self.pct = float(pct)
        self.round_to = round_to

    def transform(self, series, rng=None):
        if not pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series):
            raise TypeError(
                f"numeric_jitter needs a numeric column; {series.name!r} is "
                f"{series.dtype}"
            )
        return super().transform(series, rng)

    def build_mapping(self, uniques, rng):
        integer = all(float(v) == int(v) for v in uniques) if uniques else False
        mapping = {}
        for value in uniques:
            jittered = float(value) * rng.uniform(1 - self.pct, 1 + self.pct)
            if integer:
                mapping[value] = int(round(jittered))
            elif self.round_to is not None:
                mapping[value] = round(jittered, self.round_to)
            else:
                mapping[value] = jittered
        return mapping


class RandomStringAnonymizer(ColumnAnonymizer):
    """Replace values with random strings (unique within the column).

    ``charset`` is one of ``alphanumeric`` (default), ``letters``,
    ``digits``, ``hex``. ``prefix`` is prepended to every generated string.
    """

    CHARSETS = {
        "alphanumeric": string.ascii_lowercase + string.digits,
        "letters": string.ascii_lowercase,
        "digits": string.digits,
        "hex": "0123456789abcdef",
    }

    def __init__(self, length: int = 8, charset: str = "alphanumeric", prefix: str = ""):
        if charset not in self.CHARSETS:
            raise ValueError(
                f"Unknown charset {charset!r}. Choose from {tuple(self.CHARSETS)}."
            )
        if length < 1:
            raise ValueError("length must be >= 1")
        self.length = int(length)
        self.charset = charset
        self.prefix = str(prefix)

    def _generate(self, rng: random.Random) -> str:
        chars = self.CHARSETS[self.charset]
        return self.prefix + "".join(rng.choice(chars) for _ in range(self.length))

    def build_mapping(self, uniques, rng):
        seen: set[str] = set()
        mapping: dict[Hashable, Any] = {}
        for value in uniques:
            s = _fresh(seen, self._generate, rng)
            seen.add(s)
            mapping[value] = s
        return mapping


KINDS: dict[str, type[ColumnAnonymizer]] = {
    "names": NameAnonymizer,
    "name": NameAnonymizer,
    "sequential_id": SequentialIdAnonymizer,
    "sequential": SequentialIdAnonymizer,
    "seq": SequentialIdAnonymizer,
    "numeric_jitter": NumericJitterAnonymizer,
    "numbers": NumericJitterAnonymizer,
    "jitter": NumericJitterAnonymizer,
    "random_string": RandomStringAnonymizer,
    "string": RandomStringAnonymizer,
    "hex": RandomStringAnonymizer,
}


def make_anonymizer(kind: str, **options) -> ColumnAnonymizer:
    """Build an anonymizer by kind name, e.g. ``make_anonymizer("seq", start=100)``.

    The ``hex`` kind is shorthand for ``random_string`` with
    ``charset="hex"``.
    """
    key = kind.strip().lower()
    if key not in KINDS:
        raise ValueError(f"Unknown anonymizer kind {kind!r}. Choose from {sorted(set(KINDS))}.")
    if key == "hex":
        options.setdefault("charset", "hex")
    return KINDS[key](**options)


def _coerce(value) -> ColumnAnonymizer:
    if isinstance(value, ColumnAnonymizer):
        return value
    if isinstance(value, str):
        return make_anonymizer(value)
    if isinstance(value, tuple) and len(value) == 2 and isinstance(value[1], dict):
        return make_anonymizer(value[0], **value[1])
    if isinstance(value, dict):
        opts = dict(value)
        kind = opts.pop("kind")
        return make_anonymizer(kind, **opts)
    raise TypeError(
        "Anonymizer spec must be a ColumnAnonymizer, a kind string, a "
        "(kind, options) tuple, or a dict with a 'kind' key; "
        f"got {value!r}"
    )


def anonymize(
    df: pd.DataFrame,
    spec: Mapping[str, "ColumnAnonymizer | str | tuple | dict"],
    seed: int | None = None,
) -> pd.DataFrame:
    """Return a copy of ``df`` with the columns named in ``spec`` anonymized.

    ``spec`` maps column name → anonymizer, where the anonymizer may be a
    :class:`ColumnAnonymizer` instance, a kind string (``"names"``), a
    ``(kind, options)`` tuple (``("sequential_id", {"start": 1000})``), or a
    dict (``{"kind": "hex", "length": 12}``). ``seed`` makes the run
    reproducible. Columns not named in ``spec`` are returned unchanged.
    """
    rng = random.Random(seed)
    out = df.copy()
    for col, raw in spec.items():
        if col not in df.columns:
            raise KeyError(f"Column {col!r} not found in DataFrame")
        anonymizer = _coerce(raw)
        log.info("anonymizing column %r with %s", col, type(anonymizer).__name__)
        out[col] = anonymizer.transform(out[col], rng)
    return out

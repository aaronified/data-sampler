"""Optional per-column anonymizers.

Every anonymizer maps each *unique* original value to exactly one replacement
(consistent mapping), so repeated values stay repeated — the statistical
variety this project exists to preserve survives anonymization. Missing values
(NaN/NaT) are left untouched.

Two families differ in how completely the distribution survives:

- **Relabelling** anonymizers (``names``, ``sequential_id``, ``random_string``,
  ``hex``) are *bijective*: distinct originals map to distinct replacements, so
  the exact value-count distribution is preserved (only the labels change).
- **Jitter** anonymizers (``numeric_jitter``, ``datetime_jitter``) add bounded
  random noise: each replacement stays within ±the configured bound of its
  original, so the range and rough shape survive, but two *nearby* distinct
  values can round/shift onto the same replacement (jitter is noise, not a
  bijection). Widen the bound, add ``round_to``/finer ``unit`` resolution, or
  use a relabelling anonymizer if you need strict distinct-to-distinct output.

Available kinds (see :data:`KINDS` for aliases):

- ``names`` — replace values with realistic full names drawn from a bundled
  library of first/middle/last names (:class:`NameAnonymizer`).
- ``sequential_id`` — replace values with ``start, start+interval, ...`` in
  order of first appearance (:class:`SequentialIdAnonymizer`).
- ``numeric_jitter`` — replace numbers with a random value within ±20 % of
  the original by default (:class:`NumericJitterAnonymizer`).
- ``datetime_jitter`` — shift each date/time by a random offset within a
  ±window (±7 days by default) (:class:`DatetimeJitterAnonymizer`).
- ``random_string`` / ``hex`` — replace values with random character
  sequences or hexadecimal strings (:class:`RandomStringAnonymizer`).
"""

from __future__ import annotations

import random
import string
from abc import ABC, abstractmethod
from typing import Any, Hashable, Mapping, Sequence

import numpy as np
import pandas as pd

from . import _names
from ._logging import get_logger

log = get_logger(__name__)

AnonymizerSpec = "ColumnAnonymizer | str | tuple[str, dict] | dict"


def _np_rng(rng: random.Random) -> np.random.Generator:
    """Derive a numpy Generator from a :class:`random.Random`.

    Draws 64 bits from ``rng`` so vectorized anonymizers stay deterministic
    for a given seed and column order (``anonymize`` passes one ``Random`` to
    every column in turn), without coupling to numpy's global state.
    """
    return np.random.default_rng(rng.getrandbits(64))


class ColumnAnonymizer(ABC):
    """Base class: replace each unique original value with one replacement.

    The pipeline is vectorized. The column is dictionary-encoded once with
    :func:`pandas.factorize` — yielding integer ``codes`` per row plus the
    distinct ``uniques`` in order of first appearance — each subclass produces
    one replacement per unique value via :meth:`build_replacements`, and the
    result is assembled by fancy-indexing (gathering) ``replacements`` with
    ``codes``. That is the in-process equivalent of a native join against a
    mapping table: no Python loop over rows and no second hash pass, so cost
    scales with the number of *unique* values plus a vectorized gather over the
    rows. The consistent-mapping guarantee (equal values → equal replacements)
    is inherent to the encoding, and missing values (``NaN``/``NaT``) are
    preserved.
    """

    def transform(self, series: pd.Series, rng: random.Random | None = None) -> pd.Series:
        """Return a copy of ``series`` with all non-null values replaced."""
        rng = rng if rng is not None else random.Random()
        codes, uniques = pd.factorize(series, use_na_sentinel=True)
        if len(uniques) == 0:  # all-null (or empty) column: nothing to map
            return series.copy()
        replacements = self.build_replacements(uniques, rng)
        log.debug(
            "%s: column %r — %d unique values mapped over %d rows",
            type(self).__name__, series.name, len(uniques), len(series),
        )
        result = self._gather(series, codes, replacements)
        return self._restore_dtype(series, result)

    @staticmethod
    def _gather(
        series: pd.Series, codes: np.ndarray, replacements: Sequence[Any]
    ) -> pd.Series:
        """Gather ``replacements`` by ``codes``, restoring NaN where code == -1."""
        repl = np.asarray(replacements)
        # a code of -1 marks a missing value; clip it to 0 for the fancy-index
        # (so we never read out of bounds) then mask those positions back to NaN
        safe = np.where(codes >= 0, codes, 0)
        gathered = repl[safe]
        result = pd.Series(gathered, index=series.index, name=series.name)
        if (codes < 0).any():
            result = result.where(pd.Series(codes >= 0, index=series.index))
        return result

    def build_mapping(
        self, uniques: Sequence[Hashable], rng: random.Random
    ) -> dict[Hashable, Any]:
        """Map each unique original value to its replacement (a dict).

        Kept for direct callers; :meth:`transform` uses the vectorized
        :meth:`build_replacements` path instead.
        """
        uniques = list(uniques)
        return dict(zip(uniques, self.build_replacements(uniques, rng)))

    def _restore_dtype(self, original: pd.Series, result: pd.Series) -> pd.Series:
        # round-trip a pandas nullable string dtype (else its pd.NA marker
        # degrades to a float nan); plain object columns are left as-is
        dt = original.dtype
        if pd.api.types.is_string_dtype(dt) and not pd.api.types.is_object_dtype(dt):
            try:
                return result.astype(dt)
            except (TypeError, ValueError):
                return result
        return result

    @abstractmethod
    def build_replacements(
        self, uniques: Sequence[Hashable], rng: random.Random
    ) -> Sequence[Any]:
        """Return one replacement per unique value, in the order given.

        ``uniques`` are the column's distinct non-null values in order of first
        appearance (the ``uniques`` array from :func:`pandas.factorize`).
        """


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

    def build_replacements(self, uniques, rng):
        n = len(uniques)
        style = self.style
        # keep the collision rate low: escalate once past half capacity
        if n > self._capacity(style) // 2:
            style = "first_middle_last"
            log.debug(
                "NameAnonymizer: %d values exceed half capacity of %r; "
                "using first_middle_last", n, self.style,
            )
        seen: set[str] = set()
        out: list[str] = []
        for _ in range(n):
            name = _fresh(seen, lambda r: self._generate(style, r), rng)
            seen.add(name)
            out.append(name)
        return out


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

    def build_replacements(self, uniques, rng):
        n = len(uniques)
        # vectorized: start, start+interval, … in order of first appearance
        values = self.start + np.arange(n, dtype=np.int64) * self.interval
        if self.prefix or self.width:
            # string formatting is per-unique (∝ uniques, not rows)
            return np.array(
                [f"{self.prefix}{int(v):0{self.width}d}" for v in values],
                dtype=object,
            )
        return values

    def _restore_dtype(self, original, result):
        if not self.prefix and not self.width:
            if result.isna().any():
                return result.astype("Int64")
            return result.astype("int64")
        return result


class NumericJitterAnonymizer(ColumnAnonymizer):
    """Replace numbers with a random value within ±``pct`` of the original.

    Defaults to ±20 %. Integer columns stay integers (the in-bound draw is
    rounded to the nearest int, which for small magnitudes can nudge the result
    a fraction beyond ±``pct``); ``round_to`` rounds floats to that many decimal
    places. Note: a value of exactly 0 has no magnitude to jitter and is
    returned unchanged.

    This is additive noise, not a bijection: distinct originals that sit closer
    together than the jitter can move them (e.g. adjacent integers under a small
    ``pct``) may land on the same replacement. The consistent mapping and the
    ±``pct`` bound always hold; strict distinct-to-distinct output does not.
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

    def build_replacements(self, uniques, rng):
        vals = np.asarray(uniques, dtype=float)
        integer = bool(np.all(np.mod(vals, 1) == 0)) if len(vals) else False
        factors = _np_rng(rng).uniform(1 - self.pct, 1 + self.pct, size=len(vals))
        jittered = vals * factors
        if integer:
            return np.rint(jittered).astype(np.int64)
        if self.round_to is not None:
            return np.round(jittered, self.round_to)
        return jittered


class DatetimeJitterAnonymizer(ColumnAnonymizer):
    """Shift each date/time by a random offset within ±``max_delta``.

    ``max_delta`` is anything :class:`pandas.Timedelta` accepts (``"7D"``,
    ``"12h"``, a number of ``unit`` s); it defaults to ±7 days. The offset is
    drawn uniformly in whole ``unit`` steps (``"s"`` by default), so pick a
    ``unit`` no coarser than the resolution you want to keep — e.g.
    ``unit="D"`` jitters only by whole days. As with every anonymizer the
    mapping is consistent: equal timestamps receive equal shifts, so the
    column's shape survives. ``NaT`` is left untouched.

    Non-datetime columns are coerced with :func:`pandas.to_datetime` first (so
    date strings loaded from CSV work); a column that cannot be parsed as
    dates raises :class:`TypeError`. Timezone-aware inputs keep their zone.

    Like :class:`NumericJitterAnonymizer` this is bounded noise, not a
    bijection: distinct timestamps closer together than the jitter window (and
    at the chosen ``unit`` resolution) can shift onto the same replacement. Use
    a wider ``max_delta`` or finer ``unit`` if that matters.
    """

    def __init__(self, max_delta: str | int | float = "7D", unit: str = "s"):
        self.max_delta = pd.Timedelta(max_delta)
        if self.max_delta <= pd.Timedelta(0):
            raise ValueError("max_delta must be positive (e.g. '7D' or '12h')")
        self.unit = unit
        # number of whole `unit` steps that fit in max_delta (the jitter span)
        self._span = int(self.max_delta / pd.Timedelta(1, unit))
        if self._span < 1:
            raise ValueError(
                f"max_delta {max_delta!r} is smaller than one {unit!r} step; "
                "use a finer unit"
            )

    def transform(self, series, rng=None):
        if not pd.api.types.is_datetime64_any_dtype(series):
            try:
                series = pd.to_datetime(series, errors="raise")
            except (ValueError, TypeError) as exc:
                raise TypeError(
                    f"datetime_jitter needs a datetime column; {series.name!r} "
                    f"is {series.dtype} and could not be parsed as dates ({exc})"
                ) from exc
        return super().transform(series, rng)

    def build_replacements(self, uniques, rng):
        # vectorized: one uniform integer offset per unique, in whole units
        offsets = _np_rng(rng).integers(
            -self._span, self._span + 1, size=len(uniques)
        )
        base = pd.DatetimeIndex(uniques)
        return base + pd.to_timedelta(offsets, unit=self.unit)

    def _restore_dtype(self, original, result):
        # the gather yields object/datetime64 values; normalize back to
        # datetime64 (preserving tz-awareness of the coerced input)
        return pd.to_datetime(result)


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

    def build_replacements(self, uniques, rng):
        seen: set[str] = set()
        out: list[str] = []
        for _ in range(len(uniques)):
            s = _fresh(seen, self._generate, rng)
            seen.add(s)
            out.append(s)
        return out


KINDS: dict[str, type[ColumnAnonymizer]] = {
    "names": NameAnonymizer,
    "name": NameAnonymizer,
    "sequential_id": SequentialIdAnonymizer,
    "sequential": SequentialIdAnonymizer,
    "seq": SequentialIdAnonymizer,
    "numeric_jitter": NumericJitterAnonymizer,
    "numbers": NumericJitterAnonymizer,
    "jitter": NumericJitterAnonymizer,
    "datetime_jitter": DatetimeJitterAnonymizer,
    "datetime": DatetimeJitterAnonymizer,
    "dates": DatetimeJitterAnonymizer,
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

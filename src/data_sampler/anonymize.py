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


# ── gender / ethnicity resolution (used by NameAnonymizer) ──────────────────────

GENDERS = ("male", "female", "third", "undisclosed")

# ISO/IEC 5218 numeric codes plus common textual encodings in several languages
_MALE_TOKENS = {
    "m", "male", "man", "men", "boy", "1", "männlich", "mannlich", "homme",
    "hombre", "masculino", "maschio", "erkek", "мужской", "男", "남",
}
_FEMALE_TOKENS = {
    "f", "female", "woman", "women", "girl", "2", "weiblich", "femme", "mujer",
    "feminino", "femmina", "kadin", "kadın", "женский", "女", "여",
}
_THIRD_TOKENS = {
    "x", "o", "other", "nonbinary", "non-binary", "non binary", "nb", "enby",
    "third", "third gender", "genderqueer", "diverse", "divers", "d", "3",
}
_UNDISCLOSED_TOKENS = {
    "", "u", "unknown", "undisclosed", "not disclosed", "prefer not to say",
    "prefer not", "n/a", "na", "none", "null", "nan", "0", "9", "?",
}


def _auto_gender(value) -> str | None:
    """Best-effort map of a raw gender-column value to a canonical gender.

    Returns ``"male"`` / ``"female"`` / ``"third"`` / ``"undisclosed"``, or
    ``None`` when the value is unrecognized (so the caller can fall back to a
    fixed gender, or a manual mapping can override it).
    """
    s = str(value).strip().lower()
    if s in _MALE_TOKENS:
        return "male"
    if s in _FEMALE_TOKENS:
        return "female"
    if s in _THIRD_TOKENS:
        return "third"
    if s in _UNDISCLOSED_TOKENS:
        return "undisclosed"
    # conservative textual fallback for values we didn't enumerate
    if s.startswith("male") or s.startswith("man"):
        return "male"
    if s.startswith("female") or s.startswith("woman") or s.startswith("fem"):
        return "female"
    return None


def _auto_ethnicity(value) -> str | None:
    """Best-effort map of a raw ethnicity-column value to a known group/prefix.

    Matches an exact group (``"chinese"``, ``"indian_bengali_hindu"``) or a
    family prefix (``"indian"`` → all ``indian_*`` groups). Returns ``None`` if
    nothing matches.
    """
    s = str(value).strip().lower().replace(" ", "_").replace("-", "_")
    if not s:
        return None
    if s in _names.FIRST_ETHNICITIES:
        return s
    if any(e == s or e.startswith(s + "_") for e in _names.FIRST_ETHNICITIES):
        return s  # usable as a family prefix
    return None


def suggest_gender_mapping(values) -> dict:
    """Auto-detect a ``{raw value: gender}`` map for the distinct values of a
    gender column (``None`` for values needing a manual choice). For the TUI's
    "auto-detect + manual override" mapping step and for scripted use."""
    seen: dict = {}
    for v in values:
        if v not in seen:
            seen[v] = _auto_gender(v)
    return seen


def suggest_ethnicity_mapping(values) -> dict:
    """Auto-detect a ``{raw value: ethnicity}`` map for the distinct values of
    an ethnicity column (``None`` where no group matched)."""
    seen: dict = {}
    for v in values:
        if v not in seen:
            seen[v] = _auto_ethnicity(v)
    return seen


class NameAnonymizer(ColumnAnonymizer):
    """Replace values with realistic names from the bundled name library.

    ``style`` is one of ``first``, ``last``, ``first_last`` (default),
    ``first_middle_last``, ``last_first``.

    Gender and ethnicity shape which names are drawn:

    - ``gender`` fixes one gender for the whole column — ``"male"``,
      ``"female"``, ``"third"`` (names mixed across genders but kept within an
      ethnicity), ``"undisclosed"`` (any gender, any ethnicity), or ``None``
      (a global mix).
    - ``ethnicity`` fixes an ethnic group/prefix (e.g. ``"chinese"``,
      ``"indian"``, ``"indian_bengali_hindu"``); ``None`` draws from all.
    - ``gender_column`` / ``ethnicity_column`` instead read per-row values from
      *another* column and map them per value. ``gender_map`` / ``ethnicity_map``
      override the auto-detection for specific raw values.
    - ``randomize_gender`` (only with ``gender_column``) reassigns each name a
      random gender and returns a rewritten gender column too, so both the names
      and the gender field are anonymized consistently.

    Replacements are unique within the column; middle names are added
    automatically if a style can't supply enough distinct combinations.
    """

    STYLES = ("first", "last", "first_last", "first_middle_last", "last_first")

    def __init__(
        self,
        style: str = "first_last",
        *,
        gender: str | None = None,
        ethnicity: str | None = None,
        gender_column: str | None = None,
        ethnicity_column: str | None = None,
        gender_map: dict | None = None,
        ethnicity_map: dict | None = None,
        randomize_gender: bool = False,
    ):
        if style not in self.STYLES:
            raise ValueError(
                f"Unknown name style {style!r}. Choose from {self.STYLES}."
            )
        if gender is not None and gender not in GENDERS:
            raise ValueError(f"Unknown gender {gender!r}. Choose from {GENDERS} or None.")
        self.style = style
        self.gender = gender
        self.ethnicity = ethnicity
        self.gender_column = gender_column
        self.ethnicity_column = ethnicity_column
        self.gender_map = dict(gender_map) if gender_map else None
        self.ethnicity_map = dict(ethnicity_map) if ethnicity_map else None
        self.randomize_gender = bool(randomize_gender)

    @property
    def is_linked(self) -> bool:
        """True when the anonymizer needs another column (gender/ethnicity)."""
        return bool(
            self.gender_column or self.ethnicity_column or self.randomize_gender
        )

    def _pools(self, gender: str | None, ethnicity: str | None):
        if gender == "undisclosed":  # reveal nothing: any gender, any ethnicity
            return _names.first_names(None, None), _names.last_names(None, None)
        if gender == "third":  # mixed across genders, kept within the ethnicity
            return _names.first_names(None, ethnicity), _names.last_names(None, ethnicity)
        return _names.first_names(gender, ethnicity), _names.last_names(gender, ethnicity)

    def _generate(self, style, rng, gender=None, ethnicity=None) -> str:
        first_pool, last_pool = self._pools(gender, ethnicity)
        first = rng.choice(first_pool)
        last = rng.choice(last_pool)
        middle = rng.choice(_names.MIDDLE_NAMES)
        return {
            "first": first,
            "last": last,
            "first_last": f"{first} {last}",
            "first_middle_last": f"{first} {middle} {last}",
            "last_first": f"{last}, {first}",
        }[style]

    def _capacity(self, style, gender=None, ethnicity=None) -> int:
        f = len(_names.first_names(None if gender == "third" else gender,
                                   None if gender == "undisclosed" else ethnicity))
        last = len(_names.last_names(None if gender in ("third", "undisclosed") else gender,
                                     None if gender == "undisclosed" else ethnicity))
        m = len(_names.MIDDLE_NAMES)
        return {
            "first": f, "last": last, "first_last": f * last,
            "last_first": f * last, "first_middle_last": f * m * last,
        }[style]

    def _make_unique(self, per_unique, rng) -> list[str]:
        """Generate one unique replacement per (gender, ethnicity) pair."""
        seen: set[str] = set()
        out: list[str] = []
        for gender, ethnicity in per_unique:
            style = self.style
            # escalate to add a middle name if this pool is getting crowded
            if style != "first_middle_last" and len(out) > self._capacity(
                style, gender, ethnicity
            ) // 2:
                style = "first_middle_last"
            name = _fresh(seen, lambda r: self._generate(style, r, gender, ethnicity), rng)
            seen.add(name)
            out.append(name)
        return out

    def build_replacements(self, uniques, rng):
        # fixed (or global-mix) mode: one (gender, ethnicity) for every unique
        per_unique = [(self.gender, self.ethnicity)] * len(uniques)
        return self._make_unique(per_unique, rng)

    @staticmethod
    def _first_value_per_unique(codes, source: pd.Series, n: int) -> list:
        """First non-null ``source`` value seen for each factorized code."""
        vals = source.to_numpy()
        out: list = [None] * n
        filled = 0
        for row, code in enumerate(codes):
            if code >= 0 and out[code] is None:
                v = vals[row]
                if not (isinstance(v, float) and np.isnan(v)) and v is not None:
                    out[code] = v
                    filled += 1
                    if filled == n:
                        break
        return out

    def transform_linked(self, series, df, rng=None):
        """Transform ``series`` using gender/ethnicity read from other columns.

        Returns ``(names, gender_column_or_None)``. The second element is a
        rewritten gender column (only when ``randomize_gender`` is set with a
        ``gender_column``), otherwise ``None``.
        """
        rng = rng if rng is not None else random.Random()
        codes, uniques = pd.factorize(series, use_na_sentinel=True)
        n = len(uniques)
        if n == 0:
            return series.copy(), None

        genders: list = [self.gender] * n
        ethnicities: list = [self.ethnicity] * n
        if self.gender_column:
            raw = self._first_value_per_unique(codes, df[self.gender_column], n)
            genders = [self._map_gender(v) for v in raw]
        if self.ethnicity_column:
            raw = self._first_value_per_unique(codes, df[self.ethnicity_column], n)
            ethnicities = [self._map_ethnicity(v) for v in raw]
        if self.randomize_gender:
            genders = [rng.choice(("male", "female")) for _ in range(n)]

        repl = self._make_unique(list(zip(genders, ethnicities)), rng)
        names = self._restore_dtype(series, self._gather(series, codes, repl))

        new_gender = None
        if self.randomize_gender and self.gender_column:
            labels = np.asarray(genders, dtype=object)
            safe = np.where(codes >= 0, codes, 0)
            new_gender = pd.Series(
                labels[safe], index=series.index, name=self.gender_column
            )
            if (codes < 0).any():
                new_gender = new_gender.where(pd.Series(codes >= 0, index=series.index))
        return names, new_gender

    def _map_gender(self, value) -> str | None:
        if self.gender_map and value in self.gender_map:
            return self.gender_map[value]
        detected = _auto_gender(value)
        return detected if detected is not None else self.gender

    def _map_ethnicity(self, value) -> str | None:
        if self.ethnicity_map and value in self.ethnicity_map:
            return self.ethnicity_map[value]
        detected = _auto_ethnicity(value)
        return detected if detected is not None else self.ethnicity


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
        # a NameAnonymizer that reads gender/ethnicity from other columns needs
        # the whole frame, and may rewrite the gender column alongside the names
        if isinstance(anonymizer, NameAnonymizer) and anonymizer.is_linked:
            for linked in (anonymizer.gender_column, anonymizer.ethnicity_column):
                if linked and linked not in df.columns:
                    raise KeyError(f"Linked column {linked!r} not found in DataFrame")
            names, new_gender = anonymizer.transform_linked(out[col], out, rng)
            out[col] = names
            if new_gender is not None:
                out[anonymizer.gender_column] = new_gender
        else:
            out[col] = anonymizer.transform(out[col], rng)
    return out

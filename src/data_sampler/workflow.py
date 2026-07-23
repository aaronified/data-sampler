"""Guided anonymization workflow: pick a *column type* per column, three ways.

The one shared abstraction is :class:`AnonymizationPlan` — a mapping of each
column to a "type" (an anonymizer kind plus its options). A plan can be built:

- **programmatically (pre-specify through a function):**
  ``plan.assign("salary", "numeric_jitter", pct=0.1)``, or
  :meth:`AnonymizationPlan.suggest` to auto-infer a type for every column from
  its :class:`~data_sampler.stats.ColumnStats`;
- **interactively (choose from options):**
  :meth:`AnonymizationPlan.choose_interactively` walks the columns and offers a
  numbered menu of type options for each, defaulting to the suggested type;
- **from the TUI (click):** the columns screen writes the user's clicks into a
  plan (the TUI's per-column config *is* a plan).

Then :meth:`AnonymizationPlan.apply` runs :func:`data_sampler.anonymize`.

The available types are the anonymizer kinds (see
:data:`data_sampler.anonymize.KINDS`) plus ``"none"`` (leave unchanged);
:data:`TYPE_OPTIONS` is the canonical, human-labelled menu shared by the CLI
wizard and the TUI.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable, Iterable

import pandas as pd

from ._logging import get_logger
from .anonymize import anonymize, make_anonymizer
from .stats import ColumnStats, compute_column_stats

log = get_logger(__name__)

# Canonical column-type menu: (kind, label, help). ``kind`` is a key into
# ``anonymize.KINDS`` (or "none"). Shared by the CLI wizard and the TUI so the
# option list stays in one place.
TYPE_OPTIONS: list[tuple[str, str, str]] = [
    ("none", "leave unchanged", "write the column through as-is"),
    ("names", "names", "replace with realistic names from the bundled library"),
    ("sequential_id", "sequential id", "renumber as start, start+interval, …"),
    ("numeric_jitter", "numeric jitter", "perturb numbers within ± a percentage"),
    ("datetime_jitter", "datetime jitter", "shift dates/times within a ± window"),
    ("random_string", "random string", "replace with random character strings"),
    ("hex", "hex string", "replace with random hexadecimal strings"),
]

TYPE_LABELS: dict[str, str] = {kind: label for kind, label, _ in TYPE_OPTIONS}
TYPE_KINDS: tuple[str, ...] = tuple(kind for kind, _, _ in TYPE_OPTIONS)

# alias → canonical menu kind (anonymize.KINDS accepts e.g. "seq"/"jitter";
# the plan stores the canonical name so the wizard/TUI menus recognize it)
_CANONICAL_KINDS: dict[str, str] = {
    "name": "names",
    "sequential": "sequential_id",
    "seq": "sequential_id",
    "numbers": "numeric_jitter",
    "jitter": "numeric_jitter",
    "datetime": "datetime_jitter",
    "dates": "datetime_jitter",
    "string": "random_string",
}

# column-name substrings that hint at a semantic type
_NAME_HINTS = ("name", "surname", "firstname", "lastname", "fullname", "contact")
_EMAIL_HINTS = ("email", "e-mail", "mail")
_ID_HINTS = ("id", "uuid", "guid", "account", "ssn", "customer", "cust", "user")
# date hints are matched per name *token* (equality or prefix) rather than as a
# bare substring, so "signup_date"/"created_at" match but "candidate"/"mandate"
# /"update_reason" (which merely contain "date"/"update") do not
_DATE_HINTS = (
    "date", "datetime", "dob", "birth", "birthday", "timestamp", "created",
    "updated", "modified", "joined", "signup", "expire", "expiry",
    "expiration", "hired", "start", "end",
)


def _has_date_token(lname: str) -> bool:
    tokens = [t for t in re.split(r"[^a-z0-9]+", lname) if t]
    return any(t == h or t.startswith(h) for t in tokens for h in _DATE_HINTS)


def suggest_type(stats: ColumnStats) -> str:
    """Suggest an anonymizer kind for a column from its :class:`ColumnStats`.

    Heuristics, in order: datetime columns → ``datetime_jitter``; email-ish and
    name-ish column names → ``hex`` / ``names``; date-ish column names (even
    when the dates were loaded as strings) → ``datetime_jitter``; id-ish names
    that are almost all unique → ``sequential_id``; numeric → ``numeric_jitter``;
    free text → ``random_string``. Categorical / boolean columns default to
    ``"none"`` because they are usually what you stratify on — anonymizing them
    would destroy the categories the sample is built to preserve.

    Suggestions are advisory: every entry point (the interactive wizard, the
    TUI, ``--suggest``) lets you review and override them before anything runs.
    """
    lname = stats.name.lower()

    def name_has(hints: Iterable[str]) -> bool:
        return any(h in lname for h in hints)

    if stats.kind == "datetime":
        return "datetime_jitter"
    if name_has(_EMAIL_HINTS):
        return "hex"
    if name_has(_NAME_HINTS):
        return "names"
    # only string-ish columns can hold parseable date strings; "other" kinds
    # (TIME, INTERVAL, nested types) must never be datetime-jittered
    if stats.kind in ("categorical", "text") and _has_date_token(lname):
        return "datetime_jitter"
    id_like = name_has(_ID_HINTS) and stats.unique_pct >= 90
    if stats.kind == "numeric":
        return "sequential_id" if id_like else "numeric_jitter"
    if id_like:
        return "sequential_id"
    if stats.kind == "text":
        return "random_string"
    return "none"


@dataclass
class AnonymizationPlan:
    """A column → (anonymizer kind, options) plan for a set of columns.

    ``assignments`` maps each column name to a ``(kind, options)`` pair, where
    ``kind`` is ``"none"`` or a key of :data:`data_sampler.anonymize.KINDS` and
    ``options`` are that anonymizer's keyword arguments.
    """

    assignments: dict[str, tuple[str, dict]] = field(default_factory=dict)

    # ── construction ────────────────────────────────────────────────────────

    @classmethod
    def for_columns(cls, columns: Iterable[str]) -> "AnonymizationPlan":
        """An empty plan (every column ``"none"``) for the given columns."""
        return cls({str(c): ("none", {}) for c in columns})

    @classmethod
    def suggest(
        cls, df: pd.DataFrame, columns: Iterable[str] | None = None
    ) -> "AnonymizationPlan":
        """Build a plan by auto-inferring a type for each column via
        :func:`suggest_type`."""
        cols = list(df.columns) if columns is None else [str(c) for c in columns]
        plan = cls()
        n_rows = len(df)
        for col in cols:
            stats = compute_column_stats(df[col], n_rows)
            kind = suggest_type(stats)
            plan.assignments[col] = (kind, {})
            log.debug("suggest_type(%r) → %s", col, kind)
        return plan

    # ── editing (pre-specify through a function) ──────────────────────────────

    def assign(self, column: str, kind: str, **options) -> "AnonymizationPlan":
        """Assign ``column`` an anonymizer ``kind`` with ``options``.

        Validates eagerly: an unknown kind or bad options raise immediately
        (via :func:`~data_sampler.anonymize.make_anonymizer`). ``"none"`` clears
        any assignment. Returns ``self`` for chaining.
        """
        key = kind.strip().lower()
        if key == "none":
            self.assignments[column] = ("none", {})
            return self
        make_anonymizer(key, **options)  # validate kind + options now
        # store the canonical kind so menus (wizard/TUI) recognize it — an
        # alias like "seq" would otherwise default the wizard back to "none"
        self.assignments[column] = (_CANONICAL_KINDS.get(key, key), dict(options))
        return self

    def clear(self, column: str) -> "AnonymizationPlan":
        """Set ``column`` back to ``"none"``."""
        self.assignments[column] = ("none", {})
        return self

    def type_of(self, column: str) -> str:
        """The assigned kind for ``column`` (``"none"`` if unassigned)."""
        return self.assignments.get(column, ("none", {}))[0]

    # ── interactive (choose from options) ─────────────────────────────────────

    def choose_interactively(
        self,
        df: pd.DataFrame | None = None,
        columns: Iterable[str] | None = None,
        *,
        prompt: Callable[[str], str] = input,
        echo: Callable[[str], None] = print,
    ) -> "AnonymizationPlan":
        """Walk the columns and let the user pick a type for each from a menu.

        For every column a numbered list of :data:`TYPE_OPTIONS` is shown; the
        suggested type (from :func:`suggest_type`, when ``df`` is given) is
        marked and used as the default when the user just presses Enter. An
        out-of-range or non-numeric entry falls back to the suggestion.

        ``prompt`` and ``echo`` are injectable for testing (defaults are the
        builtin :func:`input` / :func:`print`). Returns ``self``.
        """
        if columns is not None:
            cols = [str(c) for c in columns]
        elif df is not None:
            cols = list(df.columns)
        else:
            cols = list(self.assignments)

        echo("Anonymization workflow — choose a type for each column.")
        echo("Press Enter to accept the [suggested] type, or type an option number.")
        echo("")
        n_rows = len(df) if df is not None else 0
        for col in cols:
            suggestion = "none"
            info = ""
            if df is not None and col in df.columns:
                stats = compute_column_stats(df[col], n_rows)
                suggestion = suggest_type(stats)
                info = (
                    f"  ({stats.kind}, {stats.unique:,} unique, "
                    f"{stats.missing_pct:.0f}% missing)"
                )
            # an assignment already on the plan (e.g. seeded from CLI --anon)
            # wins as the default; otherwise fall back to the suggestion
            current = self.assignments.get(col, ("none", {}))[0]
            default_kind = current if current and current != "none" else suggestion
            echo(f"Column: {col}{info}")
            for i, (kind, label, help_) in enumerate(TYPE_OPTIONS, 1):
                mark = "  [suggested]" if kind == suggestion else ""
                echo(f"  {i}. {label}{mark} — {help_}")
            default_idx = TYPE_KINDS.index(default_kind) + 1 if default_kind in TYPE_KINDS else 1
            raw = prompt(f"  choice [{default_idx}]: ").strip()
            choice = default_idx
            if raw:
                try:
                    n = int(raw)
                    if 1 <= n <= len(TYPE_OPTIONS):
                        choice = n
                except ValueError:
                    pass
            kind = TYPE_OPTIONS[choice - 1][0]
            # accepting the column's current kind keeps its options (e.g. a
            # seeded --anon "id=sequential_id:start=1000"); choosing a
            # different kind starts from that kind's defaults
            cur_kind, cur_opts = self.assignments.get(col, ("none", {}))
            self.assignments[col] = (kind, dict(cur_opts) if kind == cur_kind else {})
            echo(f"  → {TYPE_LABELS[kind]}")
            echo("")
        return self

    # ── output ────────────────────────────────────────────────────────────────

    def build_spec(self) -> dict:
        """The ``{column: anonymizer}`` spec for :func:`data_sampler.anonymize`.

        Columns typed ``"none"`` are omitted.
        """
        spec = {}
        for col, (kind, options) in self.assignments.items():
            if kind and kind != "none":
                spec[col] = make_anonymizer(kind, **options)
        return spec

    def active_columns(self) -> list[str]:
        """Columns with a non-``none`` type, in assignment order."""
        return [c for c, (k, _) in self.assignments.items() if k and k != "none"]

    def apply(self, df: pd.DataFrame, seed: int | None = None) -> pd.DataFrame:
        """Anonymize ``df`` according to this plan (returns a new frame)."""
        return anonymize(df, self.build_spec(), seed=seed)

    def summary(self) -> str:
        """A short one-line-per-column summary of the active assignments."""
        active = self.active_columns()
        if not active:
            return "no columns anonymized"
        return "\n".join(
            f"  {col}  →  {TYPE_LABELS.get(self.assignments[col][0], self.assignments[col][0])}"
            for col in active
        )

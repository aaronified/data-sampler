"""Headless TUI tests via Textual's pilot (no real terminal needed)."""

import asyncio

import pandas as pd
import pytest
from textual.widgets import DataTable, Input, Select, Static, Switch

from data_sampler.tui.app import (
    ColumnsDataTable,
    ColumnsScreen,
    DataSamplerApp,
    FileScreen,
    ReportScreen,
    anon_label,
    build_anonymizer,
    ColumnConfig,
)


def run(coro):
    asyncio.run(coro)


@pytest.fixture
def csv_file(tmp_path, demo_df):
    src = tmp_path / "data.csv"
    demo_df.to_csv(src, index=False)
    return src


async def wait_for_screen(app, pilot, screen_type, tries=100):
    for _ in range(tries):
        if isinstance(app.screen, screen_type):
            # no-arg pause drains the message queue: the screen's on_mount
            # (table population, initial RowHighlighted) must settle before
            # the test interacts with it
            await pilot.pause()
            return app.screen
        await pilot.pause(0.05)
    raise AssertionError(f"never reached {screen_type.__name__}; on {type(app.screen).__name__}")


async def wait_until(pilot, predicate, what="condition", tries=150):
    """Wait for cross-widget state to converge before asserting.

    Every Textual widget runs its own message pump, so a single
    ``pilot.pause()`` does not guarantee a posted Changed message has been
    DELIVERED to the screen yet — assert-after-one-pause is a timing lottery
    that only loses on slow CI runners. The app state is eventually
    consistent; tests must wait for it.
    """
    for _ in range(tries):
        if predicate():
            return
        await pilot.pause(0.05)
    raise AssertionError(f"never converged: {what}")


def test_file_screen_rejects_missing_file(tmp_path):
    async def go():
        app = DataSamplerApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, FileScreen)
            screen.query_one("#path", Input).value = str(tmp_path / "ghost.csv")
            screen.action_load()
            await pilot.pause()
            status = screen.query_one("#file-status", Static)
            assert "not found" in str(status.content)

    run(go())


def test_columns_screen_shows_stats(csv_file):
    async def go():
        app = DataSamplerApp(path=str(csv_file))
        async with app.run_test(size=(140, 45)) as pilot:
            screen = await wait_for_screen(app, pilot, ColumnsScreen)
            table = screen.query_one("#columns-table", DataTable)
            assert table.row_count == 7  # demo_df has 7 columns
            assert set(screen.configs) == {
                "id", "name", "region", "tier", "score", "active", "notes"
            }
            assert screen.stats["region"].stratifiable
            assert not screen.stats["notes"].stratifiable

    run(go())


def test_full_flow_sample_and_anonymize(csv_file, tmp_path):
    async def go():
        app = DataSamplerApp(path=str(csv_file))
        async with app.run_test(size=(140, 45)) as pilot:
            screen = await wait_for_screen(app, pilot, ColumnsScreen)
            # configure: anonymize the highlighted column (first = "id")
            assert screen.selected == "id"
            screen.query_one("#anon-kind", Select).value = "sequential_id"
            await wait_until(
                pilot, lambda: screen.configs["id"].kind == "sequential_id",
                "anon kind applied",
            )
            screen.query_one("#opt-seq-start", Input).value = "5000"
            await wait_until(
                pilot, lambda: screen.configs["id"].options.get("start") == "5000",
                "start option applied",
            )
            # skip a column from stratification
            screen.configs["region"].skip_strat = True
            # run
            screen.query_one("#count", Input).value = "40"
            screen.query_one("#seed", Input).value = "3"
            screen.action_run()
            await wait_for_screen(app, pilot, ReportScreen)
            out = tmp_path / "data_sample_40_anon.csv"
            assert out.exists()
            df = pd.read_csv(out)
            assert len(df) == 40
            assert df["id"].min() >= 5000

    run(go())


def test_report_screen_shows_column_histograms(csv_file):
    async def go():
        app = DataSamplerApp(path=str(csv_file))
        async with app.run_test(size=(160, 48)) as pilot:
            screen = await wait_for_screen(app, pilot, ColumnsScreen)
            screen.query_one("#count", Input).value = "50"
            screen.query_one("#seed", Input).value = "1"
            screen.action_run()
            report = await wait_for_screen(app, pilot, ReportScreen)
            report.query_one("#hist-panel")  # panel exists
            content = str(report.query_one("#hist-text", Static).content)
            assert "score" in content  # a numeric column was charted
            assert "region" in content  # a categorical column was charted
            assert "%" in content and "sample" in content

    run(go())


def test_duplicate_row_highlight_does_not_clobber_pending_edit(csv_file):
    """Regression: the CI-only race behind test_full_flow flakiness.

    A user edit queues Select.Changed("sequential_id"); if a late mount-time
    RowHighlighted for the SAME row then re-syncs the panel, it queues a stale
    Changed("none") that lands after _syncing clears and resets the config.
    A duplicate highlight must be a no-op.
    """
    from types import SimpleNamespace

    async def go():
        app = DataSamplerApp(path=str(csv_file))
        async with app.run_test(size=(140, 45)) as pilot:
            screen = await wait_for_screen(app, pilot, ColumnsScreen)
            assert screen.selected == "id"
            # user edit: queued as a Changed message, not yet processed
            screen.query_one("#anon-kind", Select).value = "sequential_id"
            # simulate the late duplicate mount-time highlight arriving first
            fake = SimpleNamespace(row_key=SimpleNamespace(value="id"))
            screen.on_data_table_row_highlighted(fake)
            await wait_until(
                pilot, lambda: screen.configs["id"].kind == "sequential_id",
                "kind survives duplicate highlight",
            )
            # and the widget was not reset underneath the user
            assert screen.query_one("#anon-kind", Select).value == "sequential_id"

    run(go())


def test_stale_changed_messages_do_not_clobber_config(csv_file):
    """Regression: the CI-only race that blocked the v3.3.0 release.

    Widgets have independent message pumps, so a stale Select.Changed("none")
    (mount-time echo or superseded edit) can be delivered AFTER the user set
    "sequential_id" and AFTER their option edits. Applying it resets the kind
    and wipes the options (KeyError: 'start' on CI). A Changed whose value no
    longer matches the widget's current value must be dropped as stale.
    """
    async def go():
        app = DataSamplerApp(path=str(csv_file))
        async with app.run_test(size=(140, 45)) as pilot:
            screen = await wait_for_screen(app, pilot, ColumnsScreen)
            assert screen.selected == "id"
            sel = screen.query_one("#anon-kind", Select)
            sel.value = "sequential_id"
            await wait_until(
                pilot, lambda: screen.configs["id"].kind == "sequential_id",
                "anon kind applied",
            )
            start_input = screen.query_one("#opt-seq-start", Input)
            start_input.value = "5000"
            await wait_until(
                pilot, lambda: screen.configs["id"].options.get("start") == "5000",
                "start option applied",
            )
            # deliver a stale Changed("none"): widget shows "sequential_id",
            # so the handler must drop it instead of resetting the config
            screen.on_select_changed(Select.Changed(sel, "none"))
            assert screen.configs["id"].kind == "sequential_id"
            assert screen.configs["id"].options["start"] == "5000"
            # stale Input.Changed likewise (widget holds "5000", message "1")
            screen.on_input_changed(Input.Changed(start_input, "1"))
            assert screen.configs["id"].options["start"] == "5000"

    run(go())


def test_full_flow_with_pca_reduction(csv_file, tmp_path):
    async def go():
        app = DataSamplerApp(path=str(csv_file))
        async with app.run_test(size=(160, 45)) as pilot:
            screen = await wait_for_screen(app, pilot, ColumnsScreen)
            screen.query_one("#count", Input).value = "40"
            screen.query_one("#seed", Input).value = "3"
            screen.query_one("#reduce-mode", Select).value = "components"
            screen.query_one("#reduce-value", Input).value = "1"
            screen.action_run()
            report = await wait_for_screen(app, pilot, ReportScreen)
            out = tmp_path / "data_sample_40_pca1.csv"
            assert out.exists()
            df = pd.read_csv(out)
            assert len(df) == 40
            assert "PC1" in df.columns
            assert "score" not in df.columns  # consumed with id into PC1
            assert "region" in df.columns  # non-numeric preserved
            content = str(report.query_one("#report-text", Static).content)
            assert "COLUMN REDUCTION (PCA)" in content

    run(go())


def test_invalid_reduce_value_notifies_instead_of_running(csv_file):
    async def go():
        app = DataSamplerApp(path=str(csv_file))
        async with app.run_test(size=(160, 45)) as pilot:
            screen = await wait_for_screen(app, pilot, ColumnsScreen)
            screen.query_one("#reduce-mode", Select).value = "variance"
            screen.query_one("#reduce-value", Input).value = "1.5"
            screen.action_run()
            await pilot.pause()
            assert isinstance(app.screen, ColumnsScreen)  # still here, no crash

    run(go())


def test_invalid_count_notifies_instead_of_running(csv_file):
    async def go():
        app = DataSamplerApp(path=str(csv_file))
        async with app.run_test(size=(140, 45)) as pilot:
            screen = await wait_for_screen(app, pilot, ColumnsScreen)
            screen.query_one("#count", Input).value = "not-a-number"
            screen.action_run()
            await pilot.pause()
            assert isinstance(app.screen, ColumnsScreen)  # still here, no crash

    run(go())


def test_toggle_skip_action(csv_file):
    async def go():
        app = DataSamplerApp(path=str(csv_file))
        async with app.run_test(size=(140, 45)) as pilot:
            screen = await wait_for_screen(app, pilot, ColumnsScreen)
            col = screen.selected
            assert not screen.configs[col].skip_strat
            screen.action_toggle_skip()
            assert screen.configs[col].skip_strat
            switch = screen.query_one("#skip-strat", Switch)
            await pilot.pause()
            assert switch.value

    run(go())


def test_auto_suggest_action_fills_column_types(csv_file):
    async def go():
        app = DataSamplerApp(path=str(csv_file))
        async with app.run_test(size=(140, 45)) as pilot:
            screen = await wait_for_screen(app, pilot, ColumnsScreen)
            # nothing configured yet
            assert all(cfg.kind == "none" for cfg in screen.configs.values())
            screen.action_suggest()
            await pilot.pause()
            # id-like numeric, name column, numeric score, free text → typed;
            # low-cardinality categoricals left as none
            assert screen.configs["name"].kind == "names"
            assert screen.configs["score"].kind == "numeric_jitter"
            assert screen.configs["id"].kind == "sequential_id"
            assert screen.configs["notes"].kind == "random_string"
            assert screen.configs["region"].kind == "none"
            # the config panel reflects the suggestion for the selected column
            sel = screen.selected
            assert screen.query_one("#anon-kind", Select).value == screen.configs[sel].kind

    run(go())


def test_stat_columns_replace_summary(csv_file):
    """The columns table exposes mean/median/mode/sd (+ reduce) columns."""
    async def go():
        app = DataSamplerApp(path=str(csv_file))
        async with app.run_test(size=(180, 45)) as pilot:
            screen = await wait_for_screen(app, pilot, ColumnsScreen)
            table = screen.query_one("#columns-table", DataTable)
            labels = [str(c.label) for c in table.columns.values()]
            for expected in ("reduce", "mean", "median", "mode", "sd"):
                assert expected in labels
            assert "summary" not in labels
            # numeric column: mean/median/sd populated; mode present
            cells = {
                str(c.label): cell
                for c, cell in zip(table.columns.values(), screen._row_cells("score"))
            }
            assert str(cells["mean"]) not in ("—", "")
            assert str(cells["sd"]) not in ("—", "")
            assert str(cells["mode"]) != "—"
            # non-numeric column: mean/median/sd are dashes, mode is the top value
            cells = {
                str(c.label): cell
                for c, cell in zip(table.columns.values(), screen._row_cells("region"))
            }
            assert str(cells["mean"]) == "—"
            assert str(cells["median"]) == "—"
            assert str(cells["sd"]) == "—"
            assert str(cells["mode"]) in {"North", "South", "East", "West"}

    run(go())


def test_multi_select_bulk_applies_anonymizer(csv_file):
    """A multi-selection makes a single anonymizer choice apply to every
    selected column at once."""
    async def go():
        app = DataSamplerApp(path=str(csv_file))
        async with app.run_test(size=(160, 45)) as pilot:
            screen = await wait_for_screen(app, pilot, ColumnsScreen)
            screen.selection = {"id", "score"}
            screen.selected = "id"
            screen.query_one("#anon-kind", Select).value = "random_string"
            await wait_until(
                pilot,
                lambda: screen.configs["id"].kind == "random_string"
                and screen.configs["score"].kind == "random_string",
                "bulk anon applied to both selected columns",
            )
            # an unselected column is untouched
            assert screen.configs["region"].kind == "none"

    run(go())


def test_ctrl_click_toggles_and_shift_click_ranges(csv_file):
    async def go():
        app = DataSamplerApp(path=str(csv_file))
        async with app.run_test(size=(160, 45)) as pilot:
            screen = await wait_for_screen(app, pilot, ColumnsScreen)
            MC = ColumnsDataTable.ModifierClick
            # ctrl-click adds individual rows and sets the anchor
            screen.on_columns_table_modifier_click(MC("id", ctrl=True, shift=False))
            screen.on_columns_table_modifier_click(MC("score", ctrl=True, shift=False))
            assert screen.selection == {"id", "score"}
            # ctrl-click an already-selected row removes it (and re-anchors there)
            screen.on_columns_table_modifier_click(MC("id", ctrl=True, shift=False))
            assert screen.selection == {"score"}
            # shift-click selects the contiguous range from the anchor (id) to
            # the clicked row (region), in dataframe column order
            screen.on_columns_table_modifier_click(MC("region", ctrl=False, shift=True))
            assert screen.selection == {"id", "name", "region"}
            # clearing empties the selection
            screen.action_clear_select()
            assert screen.selection == set()

    run(go())


def test_bulk_toggle_reduce_skip_over_selection(csv_file):
    async def go():
        app = DataSamplerApp(path=str(csv_file))
        async with app.run_test(size=(160, 45)) as pilot:
            screen = await wait_for_screen(app, pilot, ColumnsScreen)
            screen.selection = {"id", "score"}
            screen.selected = "id"
            screen.action_toggle_reduce_skip()
            assert screen.configs["id"].skip_reduce
            assert screen.configs["score"].skip_reduce
            assert not screen.configs["region"].skip_reduce

    run(go())


def test_skip_reduce_column_survives_pca(tmp_path):
    """A column flagged skip-from-reduction is preserved, not folded into a PC.

    Needs three numeric columns so excluding one still leaves two to combine.
    """
    import numpy as np

    rng = np.random.default_rng(0)
    a = rng.normal(0, 1, 60)
    frame = pd.DataFrame(
        {
            "a": a,
            "b": a * 2 + rng.normal(0, 0.01, 60),  # correlated with a
            "keep": rng.normal(5, 1, 60),
            "grp": rng.choice(["x", "y"], 60),
        }
    )
    src = tmp_path / "multi.csv"
    frame.to_csv(src, index=False)

    async def go():
        app = DataSamplerApp(path=str(src))
        async with app.run_test(size=(160, 45)) as pilot:
            screen = await wait_for_screen(app, pilot, ColumnsScreen)
            screen.configs["keep"].skip_reduce = True
            screen.query_one("#count", Input).value = "40"
            screen.query_one("#seed", Input).value = "3"
            screen.query_one("#reduce-mode", Select).value = "components"
            screen.query_one("#reduce-value", Input).value = "1"
            screen.action_run()
            await wait_for_screen(app, pilot, ReportScreen)
            out = tmp_path / "multi_sample_40_pca1.csv"
            assert out.exists()
            df = pd.read_csv(out)
            assert "keep" in df.columns   # excluded from the reduction
            assert "PC1" in df.columns    # a + b were reduced
            assert "a" not in df.columns  # consumed into PC1

    run(go())


def test_undo_redo_restores_config(csv_file):
    async def go():
        app = DataSamplerApp(path=str(csv_file))
        async with app.run_test(size=(160, 45)) as pilot:
            screen = await wait_for_screen(app, pilot, ColumnsScreen)
            assert screen.configs["id"].kind == "none"
            screen.query_one("#anon-kind", Select).value = "sequential_id"
            await wait_until(
                pilot, lambda: screen.configs["id"].kind == "sequential_id",
                "kind applied",
            )
            screen.action_undo()
            assert screen.configs["id"].kind == "none"
            screen.action_redo()
            assert screen.configs["id"].kind == "sequential_id"

    run(go())


def test_undo_keeps_at_least_ten_steps(csv_file):
    """The undo history holds well over the required ten steps."""
    async def go():
        app = DataSamplerApp(path=str(csv_file))
        async with app.run_test(size=(160, 45)) as pilot:
            screen = await wait_for_screen(app, pilot, ColumnsScreen)
            # twelve discrete structural changes, each preceded by a checkpoint
            # (the same call the real edit handlers make before mutating)
            kinds = ["names", "hex", "random_string", "sequential_id"] * 3
            for k in kinds:
                screen._checkpoint()
                screen.configs["id"].kind = k
            assert screen.configs["id"].kind == kinds[-1]
            for _ in range(10):
                screen.action_undo()
            # ten undos walked back ten distinct steps without exhausting history
            assert screen.configs["id"].kind == kinds[-11]
            assert screen._undo  # history still has room to spare

    run(go())


def test_deselected_row_is_not_bulk_edited(csv_file):
    """Regression: a row toggled OFF the selection must not receive bulk edits,
    even when it is still the cursor row."""
    async def go():
        app = DataSamplerApp(path=str(csv_file))
        async with app.run_test(size=(160, 45)) as pilot:
            screen = await wait_for_screen(app, pilot, ColumnsScreen)
            MC = ColumnsDataTable.ModifierClick
            screen.on_columns_table_modifier_click(MC("id", ctrl=True, shift=False))
            screen.on_columns_table_modifier_click(MC("score", ctrl=True, shift=False))
            assert screen.selection == {"id", "score"}
            # a real ctrl-click leaves the cursor on the clicked row; emulate
            # that, then deselect "id"
            screen.selected = "id"
            screen.on_columns_table_modifier_click(MC("id", ctrl=True, shift=False))
            assert screen.selection == {"score"}
            screen.action_toggle_reduce_skip()
            assert screen.configs["score"].skip_reduce      # still selected
            assert not screen.configs["id"].skip_reduce      # deselected → untouched

    run(go())


def test_undo_does_not_merge_edits_across_columns(csv_file):
    """Regression: editing the same option field on two different columns must
    stay two separate undo steps (the coalesce key is per target set)."""
    async def go():
        app = DataSamplerApp(path=str(csv_file))
        async with app.run_test(size=(160, 45)) as pilot:
            screen = await wait_for_screen(app, pilot, ColumnsScreen)
            screen.configs["id"].kind = "numeric_jitter"
            screen.configs["score"].kind = "numeric_jitter"
            inp = screen.query_one("#opt-jit-pct", Input)
            # edit pct on "id"
            screen.selected = "id"
            inp.value = "50"
            screen.on_input_changed(Input.Changed(inp, "50"))
            assert screen.configs["id"].options.get("pct") == "50"
            # move the cursor to "score" and edit the SAME field
            screen.selected = "score"
            inp.value = "30"
            screen.on_input_changed(Input.Changed(inp, "30"))
            assert screen.configs["score"].options.get("pct") == "30"
            assert screen.configs["id"].options.get("pct") == "50"
            # one undo reverts only score's edit; id's edit is preserved
            screen.action_undo()
            assert screen.configs["score"].options.get("pct") is None
            assert screen.configs["id"].options.get("pct") == "50"

    run(go())


def test_suggest_twice_pushes_no_phantom_undo(csv_file):
    """Regression: a second auto-suggest that changes nothing must not push an
    undo step (or wipe redo)."""
    async def go():
        app = DataSamplerApp(path=str(csv_file))
        async with app.run_test(size=(160, 45)) as pilot:
            screen = await wait_for_screen(app, pilot, ColumnsScreen)
            screen.action_suggest()
            depth = len(screen._undo)
            assert depth >= 1
            screen.action_suggest()  # idempotent — suggestions already applied
            assert len(screen._undo) == depth

    run(go())


def test_non_numeric_skip_reduce_shows_dash(csv_file):
    """A skip-from-reduction flag on a non-numeric column is a no-op and must
    not render as an active '✗ skip' exclusion."""
    async def go():
        app = DataSamplerApp(path=str(csv_file))
        async with app.run_test(size=(160, 45)) as pilot:
            screen = await wait_for_screen(app, pilot, ColumnsScreen)
            screen.configs["region"].skip_reduce = True  # region is categorical
            table = screen.query_one("#columns-table", DataTable)
            cells = {
                str(c.label): cell
                for c, cell in zip(table.columns.values(), screen._row_cells("region"))
            }
            assert str(cells["reduce"]) == "—"

    run(go())


def test_refresh_browser_action(csv_file):
    async def go():
        app = DataSamplerApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, FileScreen)
            screen.action_refresh_browser()
            await pilot.pause()
            status = str(screen.query_one("#file-status", Static).content)
            assert "refreshed" in status

    run(go())


def test_save_report_as_text(csv_file, tmp_path):
    async def go():
        app = DataSamplerApp(path=str(csv_file))
        async with app.run_test(size=(160, 48)) as pilot:
            screen = await wait_for_screen(app, pilot, ColumnsScreen)
            screen.query_one("#count", Input).value = "50"
            screen.query_one("#seed", Input).value = "1"
            screen.action_run()
            report = await wait_for_screen(app, pilot, ReportScreen)
            report.action_save_text()
            await pilot.pause()
            txt = tmp_path / "data_sample_50_report.txt"
            assert txt.exists()
            content = txt.read_text(encoding="utf-8")
            assert "COLUMN DISTRIBUTIONS" in content  # histogram section saved
            assert "score" in content
            status = str(report.query_one("#save-status", Static).content)
            assert "saved" in status

    run(go())


# ── pure helpers (no app needed) ─────────────────────────────────────────────

def test_build_anonymizer_parses_option_strings():
    cfg = ColumnConfig(kind="sequential_id", options={"start": "100", "interval": "5"})
    anon = build_anonymizer(cfg)
    mapping = anon.build_mapping(["a", "b"], __import__("random").Random(0))
    assert mapping == {"a": 100, "b": 105}

    cfg = ColumnConfig(kind="numeric_jitter", options={"pct": "10"})
    assert abs(build_anonymizer(cfg).pct - 0.10) < 1e-9


def test_build_anonymizer_bad_option_raises():
    with pytest.raises(ValueError):
        build_anonymizer(ColumnConfig(kind="numeric_jitter", options={"pct": "500"}))


def test_anon_label():
    assert anon_label(ColumnConfig()) == "—"
    assert anon_label(ColumnConfig(kind="sequential_id", options={"start": "9"})) == "seq 9+1"
    assert "±15%" in anon_label(ColumnConfig(kind="numeric_jitter", options={"pct": "15"}))
    assert anon_label(ColumnConfig(kind="hex", options={"length": "12"})) == "hex[12]"

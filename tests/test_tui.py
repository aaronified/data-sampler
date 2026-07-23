"""Headless TUI tests via Textual's pilot (no real terminal needed)."""

import asyncio

import pandas as pd
import pytest
from textual.widgets import DataTable, Input, Select, Static, Switch

from data_sampler.tui.app import (
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
            await pilot.pause()
            assert screen.configs["id"].kind == "sequential_id"
            screen.query_one("#opt-seq-start", Input).value = "5000"
            await pilot.pause()
            assert screen.configs["id"].options["start"] == "5000"
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
            await pilot.pause()
            assert screen.configs["id"].kind == "sequential_id"
            # and the widget was not reset underneath the user
            assert screen.query_one("#anon-kind", Select).value == "sequential_id"

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

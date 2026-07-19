import pandas as pd
import pytest

from data_sampler.cli import main, parse_anon_option


@pytest.fixture
def csv_file(tmp_path, demo_df):
    src = tmp_path / "data.csv"
    demo_df.to_csv(src, index=False)
    return src


def test_parse_anon_option_forms():
    assert parse_anon_option("name=names") == ("name", "names", {})
    assert parse_anon_option("id=sequential_id:start=1000,interval=7") == (
        "id", "sequential_id", {"start": 1000, "interval": 7}
    )
    assert parse_anon_option("salary=numeric_jitter:pct=0.1") == (
        "salary", "numeric_jitter", {"pct": 0.1}
    )
    assert parse_anon_option("e=hex:length=12,prefix=x") == (
        "e", "hex", {"length": 12, "prefix": "x"}
    )


@pytest.mark.parametrize("bad", ["nokind", "=names", "col=", "col=kind:junk"])
def test_parse_anon_option_rejects_malformed(bad):
    with pytest.raises(ValueError):
        parse_anon_option(bad)


def test_cli_basic_sample(csv_file, capsys):
    assert main([str(csv_file), "50", "--seed", "1"]) == 0
    out = capsys.readouterr().out
    assert "Loaded 1000 rows" in out
    assert "Sampled 50 rows" in out
    result = csv_file.parent / "data_sample_50.csv"
    assert result.exists()
    assert len(pd.read_csv(result)) == 50


def test_cli_anonymize_and_skip(csv_file, capsys):
    code = main([
        str(csv_file), "50", "--seed", "1",
        "--skip", "region",
        "--anon", "name=names",
        "--anon", "id=sequential_id:start=1000,interval=7",
        "--anon", "score=numeric_jitter",
    ])
    assert code == 0
    out = capsys.readouterr().out
    assert "excluded from stratification: region" in out
    assert "Anonymized columns" in out
    result_path = csv_file.parent / "data_sample_50_anon.csv"
    df = pd.read_csv(result_path)
    assert not df["name"].str.startswith("Person").any()
    assert df["id"].min() >= 1000


def test_cli_random_mode(csv_file, capsys):
    assert main([str(csv_file), "30", "--random"]) == 0
    assert "random" in capsys.readouterr().out.lower()
    assert (csv_file.parent / "data_sample_30.csv").exists()


def test_cli_outdir(csv_file, tmp_path):
    out_dir = tmp_path / "elsewhere"
    assert main([str(csv_file), "20", "--outdir", str(out_dir)]) == 0
    assert (out_dir / "data_sample_20.csv").exists()


def test_cli_unknown_anon_kind_exits(csv_file):
    with pytest.raises(SystemExit):
        main([str(csv_file), "10", "--anon", "name=rot13"])


def test_cli_unknown_anon_column_exits(csv_file):
    with pytest.raises(SystemExit):
        main([str(csv_file), "10", "--anon", "ghost=names"])


def test_cli_count_missing_exits(csv_file):
    with pytest.raises(SystemExit):
        main([str(csv_file)])


def test_cli_version(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0
    assert "data-sampler" in capsys.readouterr().out

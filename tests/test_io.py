import pandas as pd
import pytest

from data_sampler.io import list_sheets, load_file, save_output


@pytest.fixture
def small_df():
    return pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})


@pytest.mark.parametrize("ext", [".csv", ".tsv", ".json", ".xlsx", ".parquet"])
def test_roundtrip_all_formats(tmp_path, small_df, ext):
    src = tmp_path / f"data{ext}"
    # write via save_output itself (tag trick) after seeding with pandas
    if ext == ".csv":
        small_df.to_csv(src, index=False)
    elif ext == ".tsv":
        small_df.to_csv(src, sep="\t", index=False)
    elif ext == ".json":
        small_df.to_json(src, orient="records")
    elif ext == ".xlsx":
        small_df.to_excel(src, index=False)
    elif ext == ".parquet":
        small_df.to_parquet(src, index=False)

    df = load_file(src)
    assert len(df) == 3
    assert list(df.columns) == ["a", "b"]

    out = save_output(df, src, tag="sample_2")
    assert out.name == f"data_sample_2{ext}"
    assert out.exists()
    assert len(load_file(out)) == 3


def test_unsupported_extension_raises(tmp_path):
    bad = tmp_path / "data.docx"
    bad.write_text("not a table")
    with pytest.raises(ValueError, match="Unsupported file type"):
        load_file(bad)


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_file(tmp_path / "nope.csv")


def test_excel_sheet_selection(tmp_path, small_df):
    src = tmp_path / "book.xlsx"
    with pd.ExcelWriter(src) as writer:
        small_df.to_excel(writer, sheet_name="first", index=False)
        small_df.head(1).to_excel(writer, sheet_name="second", index=False)

    assert list_sheets(src) == ["first", "second"]
    assert len(load_file(src, sheet="second")) == 1
    assert len(load_file(src)) == 3  # default: first sheet
    assert list_sheets(tmp_path / "x.csv") == []


def test_save_output_creates_output_folder(tmp_path, small_df):
    src = tmp_path / "data.csv"
    small_df.to_csv(src, index=False)
    out = save_output(small_df, src, tag="t", output_folder=tmp_path / "deep" / "dir")
    assert out.exists()
    assert out.parent == tmp_path / "deep" / "dir"


def test_unicode_content_roundtrip(tmp_path):
    df = pd.DataFrame({"név": ["Ürün", "データ", "naïve — ✓"], "n": [1, 2, 3]})
    src = tmp_path / "uni.csv"
    df.to_csv(src, index=False)
    loaded = load_file(src)
    assert loaded["név"].tolist() == ["Ürün", "データ", "naïve — ✓"]

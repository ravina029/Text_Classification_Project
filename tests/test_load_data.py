from src.load_data import load_imdb_to_df
def test_load_missing_dir(tmp_path):
    df = load_imdb_to_df(tmp_path / "no_such_dir")
    assert df.shape[0] == 0

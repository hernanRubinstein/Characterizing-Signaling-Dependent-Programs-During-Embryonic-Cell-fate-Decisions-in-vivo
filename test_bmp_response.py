import pytest

@pytest.mark.parametrize("file, expected_dims", 
                         [("./egc_atlas.csv", 1961)])
def test_load_csv_egc_atlas(file, expected_dims):
    from bmp_response import load_csv
    df = load_csv(file)
    assert df.shape[1] == expected_dims

@pytest.mark.parametrize("file, expected_dims", 
                         [("./metadata_atlas.csv", 3)])
def test_load_csv_metadata_atlas(file, expected_dims):
    from bmp_response import load_csv
    df = load_csv(file)
    assert df.shape[1] == expected_dims

@pytest.mark.parametrize("file, expected_dims", 
                         [("./egc_query.csv", 3)])
def test_load_csv_egc_query(file, expected_dims):
    from bmp_response import load_csv
    df = load_csv(file)
    assert df.shape[1] == expected_dims

@pytest.mark.parametrize("file, expected_dims", 
                         [("./metadata_query.csv", 5)])
def test_load_csv_metadata_query(file, expected_dims):
    from bmp_response import load_csv
    df = load_csv(file)
    assert df.shape[1] == expected_dims

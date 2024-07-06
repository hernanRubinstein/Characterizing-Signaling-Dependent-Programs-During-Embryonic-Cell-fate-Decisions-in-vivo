import pytest
import pandas as pd
from bmp_response import load_data

# Define the tests
def test_load_data():
    metadata, expression_matrix = load_data('data/metadata.csv', 'data/expression_matrix.csv')
    assert isinstance(metadata, pd.DataFrame)
    assert isinstance(expression_matrix, pd.DataFrame)
    assert not metadata.empty
    assert not expression_matrix.empty

# Additional tests can be added for each function

from src.data_loader import load_data
from src.preprocessing import split_data
import pandas as pd

def test_split_data():
    """
    Unit test for the split_data() function.

    Verifies that:
    - Data is split into non-empty training and testing sets.
    - Feature and label sets are of compatible lengths.
    - Data types are correct (DataFrame or Series).
    """
    data = load_data()
    X_train, X_test, y_train, y_test = split_data(data)

    # Ensure all parts are not None
    assert all(v is not None for v in [X_train, X_test, y_train, y_test]), "Split returned None values"

    # Ensure sets are not empty
    assert len(X_train) > 0, "Training data should not be empty"
    assert len(X_test) > 0, "Test data should not be empty"

    # Ensure lengths match
    assert len(X_train) == len(y_train), "Mismatch in training features and labels"
    assert len(X_test) == len(y_test), "Mismatch in test features and labels"

    # Optional: Ensure correct types
    assert isinstance(X_train, (pd.DataFrame)), "X_train should be a DataFrame"
    assert isinstance(y_train, (pd.Series, pd.DataFrame)), "y_train should be a Series or DataFrame"

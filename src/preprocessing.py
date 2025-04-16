from sklearn.model_selection import train_test_split

def split_data(data, target_column="species", test_size=0.2, random_state=42):
    """
    Splits the dataset into training and testing subsets.

    Parameters:
    data (pandas.DataFrame): The dataset to split.
    target_column (str): The name of the target column. Default is 'species'.
    test_size (float): The proportion of the dataset to include in the test split.
    random_state (int): Controls the shuffling applied to the data before splitting.

    Returns:
    tuple: X_train, X_test, y_train, y_test (training and testing data and labels).
    """
    try:
        X = data.drop(columns=[target_column])
        y = data[target_column]
        return train_test_split(X, y, test_size=test_size, random_state=random_state)
    except KeyError:
        print(f"Error: Target column '{target_column}' not found in the dataset.")
        return None, None, None, None

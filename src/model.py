from sklearn.ensemble import RandomForestClassifier

def train_model(X_train, y_train, n_estimators=100, max_depth=None, random_state=42):
    """
    Trains a RandomForestClassifier on the given training data.

    Parameters:
    X_train (pandas.DataFrame or numpy.ndarray): Training features.
    y_train (pandas.Series or numpy.ndarray): Training labels.
    n_estimators (int): Number of trees in the forest. Default is 100.
    max_depth (int or None): Maximum depth of the tree. Default is None (nodes are expanded until all leaves are pure).
    random_state (int): Random seed for reproducibility. Default is 42.

    Returns:
    RandomForestClassifier: The trained Random Forest model.
    """
    try:
        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state)
        model.fit(X_train, y_train)
        return model
    except Exception as e:
        print(f"Model training failed: {e}")
        return None


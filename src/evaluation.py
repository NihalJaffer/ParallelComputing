from sklearn.metrics import classification_report

def evaluate_model(model, X_test, y_test, target_names=None):
    """
    Evaluates a trained classification model using test data and generates
    a classification report.

    Parameters:
    model: The trained classification model (e.g., RandomForestClassifier).
    X_test (pandas.DataFrame or numpy.ndarray): The test feature set.
    y_test (pandas.Series or numpy.ndarray): The true test labels.
    target_names (list, optional): Class names for the report. Defaults to None.

    Returns:
    str: A classification report as a string.
    """
    try:
        predictions = model.predict(X_test)
        report = classification_report(y_test, predictions, target_names=target_names)
        return report
    except Exception as e:
        return f"Error during model evaluation: {e}"

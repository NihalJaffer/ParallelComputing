from src.model import train_model
from src.data_loader import load_data
from src.preprocessing import split_data
from sklearn.ensemble import RandomForestClassifier

def test_train_model():
    """
    Unit test for the train_model() function.

    Verifies that:
    - The model is not None.
    - The model is an instance of RandomForestClassifier.
    - The model has a 'predict' method.
    """
    data = load_data()
    X_train, _, y_train, _ = split_data(data)

    model = train_model(X_train, y_train)

    assert model is not None, "Model should not be None"
    assert isinstance(model, RandomForestClassifier), "Model should be a RandomForestClassifier instance"
    assert hasattr(model, "predict"), "Trained model should have a 'predict' method"

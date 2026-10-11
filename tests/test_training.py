from sklearn.utils.validation import check_is_fitted

from train import train_model


def test_model_training():
    model, X_test, y_test = train_model()

    check_is_fitted(model)

    predictions = model.predict(X_test)

    assert len(predictions) == len(y_test)
    assert set(predictions).issubset({0, 1})
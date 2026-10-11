import joblib

from train import save_model, train_model


def test_model_saving(tmp_path, monkeypatch):
    import train

    model, X_test, _ = train_model()

    model_path = tmp_path / "model.joblib"
    monkeypatch.setattr(train, "MODEL_PATH", model_path)

    save_model(model)

    assert model_path.is_file()

    loaded_model = joblib.load(model_path)

    original_predictions = model.predict(X_test)
    loaded_predictions = loaded_model.predict(X_test)

    assert (original_predictions == loaded_predictions).all()
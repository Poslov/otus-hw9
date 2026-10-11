from pathlib import Path

import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from dataset import split_dataset


MODEL_PATH = Path(__file__).resolve().parent / "models" / "model.joblib"


def train_model():
    X_train, X_test, y_train, y_test = split_dataset()

    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, random_state=42)
    )

    model.fit(X_train, y_train)

    return model, X_test, y_test


def save_model(model):
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)


if __name__ == "__main__":
    model, X_test, y_test = train_model()

    predictions = model.predict(X_test)
    save_model(model)

    print("Модель успешно обучена")
    print(f"Количество предсказаний: {len(predictions)}")
    print(f"Модель сохранена: {MODEL_PATH}")
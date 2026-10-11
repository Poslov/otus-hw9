from sklearn.metrics import accuracy_score, precision_score, recall_score

from train import train_model


def calculate_metrics(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, pos_label=0),
        "recall": recall_score(y_true, y_pred, pos_label=0),
    }


if __name__ == "__main__":
    model, X_test, y_test = train_model()

    predictions = model.predict(X_test)
    metrics = calculate_metrics(y_test, predictions)

    print("Метрики качества модели:")

    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt

from evaluate import calculate_metrics
from train import train_model


REPORT_PATH = Path(__file__).resolve().parent / "reports" / "metrics.png"


def save_metrics_chart(metrics):
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    names = list(metrics.keys())
    values = list(metrics.values())

    fig, ax = plt.subplots(figsize=(8, 5))

    bars = ax.bar(names, values)

    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Значение метрики")
    ax.set_title("Метрики качества модели")
    ax.bar_label(bars, fmt="%.4f", padding=3)

    fig.tight_layout()
    fig.savefig(REPORT_PATH, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    model, X_test, y_test = train_model()

    predictions = model.predict(X_test)
    metrics = calculate_metrics(y_test, predictions)

    save_metrics_chart(metrics)

    print(f"График сохранён: {REPORT_PATH}")
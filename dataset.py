from pathlib import Path

import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split


DATA_PATH = Path(__file__).resolve().parent / "data" / "dataset.csv"


def load_dataset() -> pd.DataFrame:
    dataset = load_breast_cancer(as_frame=True)

    return dataset.frame


def save_dataset() -> None:
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)

    data = load_dataset()
    data.to_csv(DATA_PATH, index=False)


def split_dataset():
    data = load_dataset()

    X = data.drop(columns=["target"])
    y = data["target"]

    return train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )


if __name__ == "__main__":
    save_dataset()
    print(f"Датасет сохранён: {DATA_PATH}")
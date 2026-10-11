from dataset import load_dataset, split_dataset


def test_dataset_loading():
    data = load_dataset()

    assert not data.empty
    assert len(data) == 569
    assert "target" in data.columns
    assert not data.isnull().values.any()


def test_dataset_split():
    X_train, X_test, y_train, y_test = split_dataset()

    assert len(X_train) == 455
    assert len(X_test) == 114
    assert len(X_train) == len(y_train)
    assert len(X_test) == len(y_test)
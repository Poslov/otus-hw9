# Автотестирование ML-модели на scikit-learn

Домашнее задание по автоматизированному тестированию модели машинного обучения с расчётом метрик качества и визуализацией результатов.

## Используемые технологии

- Python 3.11
- scikit-learn
- pandas
- pytest
- joblib
- matplotlib

## Описание проекта

Проект использует датасет Breast Cancer Wisconsin для обучения модели бинарной классификации `LogisticRegression`.

Датасет содержит 569 записей, 30 числовых признаков и два класса:

- `0` — злокачественная опухоль (malignant).
- `1` — доброкачественная опухоль (benign).

Данные разделяются на обучающую (80%) и тестовую (20%) выборки.

Модель обучается с использованием `StandardScaler` и `LogisticRegression`, объединённых в `Pipeline`.

## Структура проекта

```text
HW9/
├── data/
│   └── dataset.csv
├── models/
│   └── model.joblib
├── reports/
│   └── metrics.png
├── tests/
│   ├── test_data.py
│   ├── test_training.py
│   ├── test_metrics.py
│   └── test_model_save.py
├── dataset.py
├── train.py
├── evaluate.py
├── visualize.py
├── requirements.txt
├── .gitignore
└── README.md
```

## Установка и запуск

### 1. Клонирование репозитория

```powershell
git clone https://github.com/Poslov/otus-hw9.git
cd otus-hw9
```

### 2. Создание виртуального окружения

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Установка зависимостей

```powershell
python -m pip install -r requirements.txt
```

### 4. Подготовка датасета

```powershell
python dataset.py
```

Создаётся файл `data/dataset.csv`.

### 5. Обучение модели

```powershell
python train.py
```

Модель обучается, выполняет предсказания на тестовой выборке и сохраняется в `models/model.joblib`.

### 6. Расчёт метрик

```powershell
python evaluate.py
```

Пример полученных результатов:

| Метрика | Значение |
|---|---:|
| Accuracy | 98,25% |
| Precision | 97,62% |
| Recall | 97,62% |

Для Precision и Recall положительным считается класс `0` — злокачественная опухоль.

### 7. Построение графика

```powershell
python visualize.py
```

Создаётся файл `reports/metrics.png` с визуализацией трёх метрик.

### 8. Запуск автотестов

```powershell
python -m pytest tests/ -v
```

Ожидаемый результат:

```text
tests/test_data.py::test_dataset_loading PASSED
tests/test_data.py::test_dataset_split PASSED
tests/test_metrics.py::test_metrics_calculation PASSED
tests/test_model_save.py::test_model_saving PASSED
tests/test_training.py::test_model_training PASSED

5 passed
```

## Что проверяют автотесты

| Тест | Проверка |
|---|---|
| `test_dataset_loading` | Наличие данных, количество записей и отсутствие пропусков |
| `test_dataset_split` | Разделение данных на обучающую и тестовую выборки |
| `test_model_training` | Успешное обучение и возможность предсказания классов |
| `test_metrics_calculation` | Правильность вычисления Accuracy, Precision и Recall на заранее известных данных |
| `test_model_saving` | Сохранение, загрузка модели и совпадение предсказаний до и после загрузки |

## Особенности реализации

- Используется фиксированный `random_state=42` для воспроизводимости.
- Масштабирование признаков выполняется внутри `Pipeline`.
- Метрики рассчитываются на тестовой выборке.
- Модель сохраняется с использованием `joblib`.
- Автотест сохранения модели использует временную директорию `pytest` и не изменяет основную модель.
- Визуализация создаётся автоматически через `matplotlib`.

Результаты метрик относятся к конкретному разделению датасета и не гарантируют аналогичное качество на других данных.
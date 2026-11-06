# SMS AI Project

Учебный проект: классификация английских SMS на обычные сообщения (`ham`) и спам (`spam`).

## Эксперимент

TF-IDF преобразует текст в признаки, затем небольшая нейросеть MLP с одним скрытым слоем из 32 нейронов выполняет классификацию. Одинаковая архитектура обучается 10 раз: на 10%, 20%, …, 100% обучающей выборки.

Разделение данных: 90% для обучения, 10% для общего теста. Повторяющиеся нормализованные тексты удаляются до разделения; тексты с противоречивыми метками исключаются. TF-IDF обучается только на соответствующей обучающей подвыборке.

## Запуск

```powershell
python -m pip install -r requirements.txt
python main.py
```

Только построение графика по сохранённым результатам:

```powershell
python main.py --plot-only
```

## Результаты

- `results/metrics.csv` — accuracy, precision, recall и F1 для спама.
- `results/accuracy.png` — точность на тесте в зависимости от объёма обучения.

Это один эксперимент с фиксированным random_state=42; показатели не гарантируют качество на произвольных новых SMS.

## Данные

Сейчас `main.py` использует [UCI SMS Spam Collection](https://archive.ics.uci.edu/dataset/228/sms+spam+collection): файл `SMSSpamCollection` в корне проекта или папке `data`.

Дополнительно скачан оригинальный [Super SMS Dataset](https://github.com/smspamresearch/spstudy/tree/main/Data): `data/super_sms/super_sms_dataset.csv`. Столбцы: `SMSes`, `Labels`. Этот CSV пока не подключён к обучению; перед подключением нужно обработать кодировку и проверить соответствие числовых меток классам. Оригинальные байты файла сохранены без изменений.

Источник Super SMS: Salman, Ikram, Kaafar (2024), “Investigating Evasive Techniques in SMS Spam Filtering: A Comparative Analysis of Machine Learning Models”, IEEE Access. При использовании данных учитывайте условия исходных источников и цитируйте авторов.

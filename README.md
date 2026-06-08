# Drone Detection — БПЛА Детекция

Дипломная работа ИТМО: детекция БПЛА/дронов с использованием YOLOv12 / YOLO26 (Ultralytics ≥8.4.4) и Faster R-CNN.

## Классы
| ID | Класс      | Описание              |
|----|------------|-----------------------|
| 0  | DRONE      | Беспилотник / БПЛА    |
| 1  | AIRPLANE   | Самолёт               |
| 2  | HELICOPTER | Вертолёт              |
| 3  | BIRD       | Птица                 |

## Структура проекта

```
Drone_Detection/
├── .cursor/rules/drone-detection.mdc   ← Правила для Claude в Cursor
├── notebooks/
│   ├── 01_dataset_prep.ipynb           ← EDA, конвертация YOLO→COCO
│   ├── 02_yolov12_train.ipynb          ← Обучение YOLOv12n и YOLOv12s
│   ├── 02_yolo26_train.ipynb           ← Обучение YOLO26n и YOLO26s (тот же API)
│   ├── 03_faster_rcnn_train.ipynb      ← Обучение Faster R-CNN
│   ├── 04_evaluation.ipynb             ← Метрики, confusion matrix
│   └── 05_video_inference.ipynb        ← Инференс на видео
├── scripts/
│   ├── convert_to_coco.py              ← Конвертер YOLO→COCO
│   └── video_inference.py              ← Локальный инференс на видео
├── configs/
│   └── drone_data.yaml                 ← Конфиг датасета для YOLO
├── thesis/
│   ├── THESIS_OUTLINE.md               ← Структура диплома
│   └── figures/                        ← Графики (копировать из Drive)
└── reference/                          ← Референсные проекты (добавить вручную)
```

## Порядок запуска (7 дней)

### День 1 — Датасет (Roboflow)
1. Зарегистрироваться на https://universe.roboflow.com
2. Найти датасеты: `drone detection`, `uav bird detection`
3. Создать Workspace → объединить датасеты (Dataset Merge)
4. Экспорт → **YOLOv8 format** → скопировать API key
5. Также экспорт → **COCO format** (нужен для Faster R-CNN)
6. Загрузить в Google Drive подготовленный датасет (см. раздел «Google Drive» ниже)

### День 2 — EDA в Colab
Открыть `notebooks/01_dataset_prep.ipynb` в Google Colab:
- Runtime → Change runtime type → GPU (T4)
- Вставить Roboflow API key в ячейку 4
- Запустить все ячейки

### День 3 — Обучение YOLO (v12 или v26)
Открыть **`notebooks/02_yolov12_train.ipynb`** (YOLOv12n/s) и/или **`notebooks/02_yolo26_train.ipynb`** (YOLO26n/s, см. `reference/YOLO26_Tutorial.ipynb`):
- Один и тот же `data.yaml`; метрики в `results/yolo_metrics.json` **объединяются** по ключам (`yolo12n`/`yolo12s` и `yolo26n`/`yolo26s`)
- Запустить все ячейки (~1–2 часа на T4 на пару моделей)

### День 4 — Обучение Faster R-CNN
Открыть `notebooks/03_faster_rcnn_train.ipynb`:
- Запустить все ячейки (~2–3 часа на T4)

### День 5 — Оценка моделей
Открыть `notebooks/04_evaluation.ipynb`:
- Загрузит результаты из Drive и строит все графики

### День 6 — Инференс на видео
Открыть `notebooks/05_video_inference.ipynb`:
- Добавить тестовые .mp4 в `MyDrive/Colab Notebooks/videos/` (см. структуру Drive ниже)
- Или использовать yt-dlp для скачивания с YouTube

### День 7 — Диплом
Заполнить `thesis/THESIS_OUTLINE.md`, вставить графики из Drive.

---

## Google Drive структура

Ноутбуки по умолчанию используют корень **`MyDrive/Colab Notebooks`** (как у загруженных `prepared/`, `class_map.yaml`, копии `notebooks/`).

```
MyDrive/Colab Notebooks/
├── prepared/
│   ├── yolo/              ← YOLO: images/, labels/, data.yaml
│   └── dataset_coco/      ← COCO: images/, annotations/instances_*.json
├── class_map.yaml         ← справочно (ремап классов); обучение читает prepared/yolo/data.yaml
├── notebooks/             ← опционально: копии .ipynb с GitHub; на пути к данным не влияет
├── weights/               ← создаётся при обучении или вручную
├── results/               ← метрики JSON, графики, CSV
├── videos/                ← входные ролики для 05_video_inference
└── video_results/         ← выход 05 (создаётся скриптом)
```

Если удобнее другой корень (например `MyDrive/DroneDetection`) — в каждом ноутбуке измени одну строку `DRIVE_ROOT = Path('...')`.

**Важно:** в `prepared/yolo/data.yaml` используй **`path: .`** (корень датасета = папка с этим файлом). Так работает и Colab, и локальный ПК. Абсолютный путь с другой ОС (например `C:/Users/...`) в Colab **сломает** загрузку датасета.

## Локальный запуск инференса (после обучения)

```bash
# Установка (uv быстрее pip — как в официальном Ultralytics Colab)
pip install uv
uv pip install ultralytics opencv-python

# Инференс на видео
python scripts/video_inference.py \
    --weights weights/yolo12s_drone_best.pt \
    --source test_video.mp4 \
    --output output_annotated.mp4 \
    --conf 0.35

# Веб-камера
python scripts/video_inference.py \
    --weights weights/yolo12s_drone_best.pt \
    --source 0 \
    --show
```

## Конвертер YOLO → COCO (если нужно вручную)

```bash
python scripts/convert_to_coco.py \
    --yolo_dir /path/to/yolo_dataset \
    --output_dir /path/to/coco_output \
    --classes DRONE AIRPLANE HELICOPTER BIRD
```

## Стек технологий

| Компонент       | Версия / Описание                                   |
|-----------------|-----------------------------------------------------|
| ultralytics     | ≥8.4.4 (YOLOv12 + YOLOv26; install: uv pip install) |
| PyTorch         | ≥2.9.0                                              |
| torchvision     | ≥0.20.0 (Faster R-CNN)                              |
| OpenCV          | ≥4.9                                                |
| albumentations  | ≥1.3.0 (аугментация для Faster R-CNN)               |
| roboflow        | ≥1.1 (API для датасетов)                            |
| pycocotools     | ≥2.0.7 (COCO mAP)                                   |
| Google Colab    | T4 / A100 GPU (torch 2.9.0+cu126 подтверждён)       |

## Ожидаемые результаты (ориентиры)

| Модель       | mAP@0.5 | FPS (T4) | Размер |
|--------------|---------|----------|--------|
| YOLOv12n     | ~75–82% | ~120     | 6 MB   |
| YOLOv12s     | ~82–88% | ~80      | 22 MB  |
| Faster R-CNN | ~88–92% | ~15      | 160 MB |

## Апгрейд до YOLOv26 (когда понадобится)

YOLOv26 уже доступен в `ultralytics>=8.4.4`. Для перехода достаточно изменить одну строку:

```python
# Было:
model = YOLO('yolo12s.pt')
# Стало:
model = YOLO('yolo26s.pt')
```

Все остальные ноутбуки, конфиги и скрипты остаются без изменений.
Модели на Kaggle: https://www.kaggle.com/models/ultralytics/yolo26

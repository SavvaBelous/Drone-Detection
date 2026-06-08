# 🛸 Drone Detection — Обнаружение БПЛА с помощью Computer Vision

> Выпускная квалификационная работа, ИТМО 2026  
> Белоус Савва · [savva.belous@gmail.com](mailto:savva.belous@gmail.com)

Разработка и экспериментальная проверка алгоритмов детекции воздушных объектов на основе глубокого обучения. Реализованы, обучены и сравнены одноэтапные детекторы семейств **YOLOv12** и **YOLOv26** и двухэтапный **Faster R-CNN**. Дополнительно верифицирован пилотный видеоконтур: трекинг объектов, выгрузка траекторий и подготовка кропов для повторной идентификации (Re-ID).

---

## Классы

| ID | Класс | Описание |
|---|---|---|
| 0 | `DRONE` | Беспилотник / БПЛА |
| 1 | `AIRPLANE` | Самолёт |
| 2 | `HELICOPTER` | Вертолёт |
| 3 | `BIRD` | Птица |

---

## Результаты

Обучение на мультиклассовом датасете ~20 000 изображений (train / val / test). Все модели оценены на едином тестовом сплите.

| Модель | mAP@0.5 | mAP@0.5:0.95 | Скорость | Размер весов |
|---|---|---|---|---|
| YOLOv12n | 0.931 | 0.618 | ⚡ высокая | ~6 MB |
| **YOLOv12s** | **0.947** | 0.636 | ⚡ высокая | ~22 MB |
| YOLOv26n | 0.934 | 0.621 | ⚡ высокая | ~6 MB |
| YOLOv26s | 0.943 | **0.638** | ⚡ выше всех | ~22 MB |
| Faster R-CNN | ниже | ниже | 🐢 низкая | ~160 MB |

**Ночная выборка** (`night_drones_vs_planes`): mAP@0.5 = **0.97–0.99** для всех конфигураций YOLO без дообучения на ночных данных.

---

## Структура репозитория

```
Drone_Detection/
├── notebooks/
│   ├── 01_dataset_prep.ipynb       # EDA, конвертация YOLO→COCO
│   ├── 02_yolov12_train.ipynb      # Обучение YOLOv12n и YOLOv12s
│   ├── 02_yolo26_train.ipynb       # Обучение YOLOv26n и YOLOv26s
│   ├── 03_faster_rcnn_train.ipynb  # Обучение Faster R-CNN
│   ├── 04_evaluation.ipynb         # Метрики, confusion matrix, PR-кривые
│   └── 05_video_inference.ipynb    # Инференс на видео, трекинг, Re-ID кропы
├── scripts/
│   ├── convert_to_coco.py          # Конвертер YOLO→COCO
│   └── video_inference.py          # Локальный инференс на видео
├── configs/
│   └── drone_data.yaml             # Конфиг датасета для YOLO
└── thesis/
    ├── THESIS_OUTLINE.md
    └── figures/
```

---

## Быстрый старт

### Инференс на видео (локально)

```bash
pip install uv
uv pip install ultralytics opencv-python

python scripts/video_inference.py \
    --weights weights/yolo12s_drone_best.pt \
    --source test_video.mp4 \
    --output output_annotated.mp4 \
    --conf 0.35
```

Веб-камера:

```bash
python scripts/video_inference.py \
    --weights weights/yolo12s_drone_best.pt \
    --source 0 \
    --show
```

### Конвертация датасета YOLO → COCO

```bash
python scripts/convert_to_coco.py \
    --yolo_dir /path/to/yolo_dataset \
    --output_dir /path/to/coco_output \
    --classes DRONE AIRPLANE HELICOPTER BIRD
```

---

## Воспроизведение обучения (Google Colab)

Ноутбуки рассчитаны на Google Colab (T4/A100). Датасет хранится в Google Drive.

**Ожидаемая структура Drive:**

```
MyDrive/Colab Notebooks/
├── prepared/
│   ├── yolo/           # images/, labels/, data.yaml
│   └── dataset_coco/   # images/, annotations/instances_*.json
├── weights/            # чекпоинты после обучения
├── results/            # метрики JSON, графики, CSV
├── videos/             # входные ролики для 05_video_inference
└── video_results/      # аннотированные видео (создаётся скриптом)
```

Если используется другой корень Drive — поменяйте одну строку в любом ноутбуке:

```python
DRIVE_ROOT = Path('MyDrive/DroneDetection')  # ваш путь
```

**Важно:** в `prepared/yolo/data.yaml` используйте `path: .` — абсолютные пути ломаются при переносе между Colab-сессиями.

---

## Стек

| Компонент | Версия |
|---|---|
| Python | ≥3.10 |
| PyTorch | ≥2.9.0 |
| ultralytics | ≥8.4.4 |
| torchvision | ≥0.20.0 |
| OpenCV | ≥4.9 |
| albumentations | ≥1.3.0 |
| pycocotools | ≥2.0.7 |

---

## Видеоконтур (трекинг и Re-ID)

Ноутбук `05_video_inference.ipynb` реализует полный пайплайн после детекции:

- покадровый инференс с замером FPS
- трекинг объектов и выгрузка траекторий в CSV
- оценка скорости центра ограничивающей рамки (пикс/с)
- вырезки (кропы) объектов по рамкам детекции
- монтаж кропов одного объекта для подготовки данных Re-ID

Обучение модуля Re-ID и метрики межкамерного сопоставления в данную работу не входят.

---


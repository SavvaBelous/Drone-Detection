# Рекомендуемые датасеты на Roboflow Universe

Список датасетов для объединения в финальный датасет. Искать на https://universe.roboflow.com

## Поисковые запросы
- `drone detection`
- `uav detection`
- `drone bird airplane`
- `aerial object detection`

---

## Датасеты (проверены на наличие нужных классов)

### 1. Drone Detection (основной)
- **URL:** https://universe.roboflow.com/search?q=drone+detection
- **Классы для поиска:** drone, UAV, DJI → переименовать в DRONE
- **Размер:** ищи датасеты с 2k+ изображений

### 2. UAV Bird Airplane Detection
- **URL:** https://universe.roboflow.com/search?q=uav+bird+airplane
- **Классы:** bird, airplane, helicopter — все нужны
- **Ремаппинг:** убедись что индексы совпадают с твоими (0=DRONE, 1=AIRPLANE, 2=HELICOPTER, 3=BIRD)

### 3. Aerial Objects (Mixed)
- Ищи датасеты с несколькими классами воздушных объектов
- Отдавай предпочтение датасетам с ≥1000 изображений

---

## Как объединить на Roboflow (рекомендуется)

1. Создай новый проект: **Create Project** → Object Detection
2. В проекте: **Add Data** → **Roboflow Universe** → ищи нужные датасеты
3. При добавлении укажи **Class Mapping**:
   - `uav` → `DRONE`
   - `dji` → `DRONE`
   - `quadrotor` → `DRONE`
   - `plane` → `AIRPLANE`
   - `aircraft` → `AIRPLANE`
   - `birds` → `BIRD`
4. Установи сплиты: Train 70% / Val 20% / Test 10%
5. **Generate** → выбери аугментации (немного, т.к. мы делаем свои)
6. **Export** → YOLOv8 → скопируй код с API key
7. Повтори Export → COCO format

---

## Как скачать через API (в Colab)

```python
from roboflow import Roboflow

rf = Roboflow(api_key="YOUR_KEY")
project = rf.workspace("YOUR_WS").project("YOUR_PROJECT")

# YOLO format
dataset_yolo = project.version(1).download("yolov8", location="/content/dataset")

# COCO format  
dataset_coco = project.version(1).download("coco", location="/content/dataset_coco")
```

---

## Альтернатива: Kaggle датасеты

- https://www.kaggle.com/datasets?search=drone+detection
- https://www.kaggle.com/datasets?search=uav+detection
- Скачать в Colab: `kaggle datasets download -d OWNER/DATASET-NAME`

---

## Целевой размер финального датасета

| Сплит | Изображений | % |
|-------|-------------|---|
| Train | ≥7000       | 70% |
| Val   | ≥1500       | 15% |
| Test  | ≥1500       | 15% |
| **Итого** | **≥10000** | 100% |

### Распределение по классам (желательно сбалансированное)
| Класс      | Минимум объектов |
|------------|-----------------|
| DRONE      | 3000+           |
| AIRPLANE   | 2000+           |
| HELICOPTER | 1500+           |
| BIRD       | 3000+           |

Если классы несбалансированы — используй `copy_paste` аугментацию для редких классов.

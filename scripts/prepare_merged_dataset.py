"""Prepare Roboflow merged export (28 classes, train-only) for thesis pipeline.

Unzips YOLO + COCO exports, remaps classes to 4 (DRONE, AIRPLANE, HELICOPTER, BIRD),
splits into train/val/test (default 70/15/15), writes YOLO ``data.yaml`` and COCO
``instances_*.json`` under ``dataset/prepared/``.

Usage:
    python scripts/prepare_merged_dataset.py --dataset-root dataset
"""

from __future__ import annotations

import argparse
import json
import random
import re
import shutil
import zipfile
from pathlib import Path
from typing import Any

# Filled in main() from dataset/class_map.yaml (single source of truth).
YOLO_TO_FOUR: dict[int, int] = {}

CLASS_NAMES: list[str] = ["DRONE", "AIRPLANE", "HELICOPTER", "BIRD"]


def load_yolo_to_four_from_class_map(path: Path) -> dict[int, int]:
    """Load ``yolo_index_to_four`` from ``class_map.yaml`` (no PyYAML dependency)."""
    text = path.read_text(encoding="utf-8")
    if "yolo_index_to_four:" not in text:
        raise ValueError(f"No yolo_index_to_four block in {path}")
    out: dict[int, int] = {}
    in_block = False
    for line in text.splitlines():
        if not in_block:
            if "yolo_index_to_four:" in line:
                in_block = True
            continue
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("#"):
            continue
        if (line[0] not in (" ", "\t")) and ":" in stripped:
            if not re.match(r"^\d+\s*:", stripped):
                break
        m = re.match(r"^\s*(\d+)\s*:\s*(\d+)", line)
        if m:
            out[int(m.group(1))] = int(m.group(2))
    for k in range(28):
        if k not in out:
            raise ValueError(f"Missing mapping for YOLO index {k} in {path}")
    return out


def remap_yolo_label_file(text: str) -> str:
    """Remap YOLO label lines; drop lines with unknown class ids."""
    out_lines: list[str] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        old = int(parts[0])
        if old not in YOLO_TO_FOUR:
            continue
        new_c = YOLO_TO_FOUR[old]
        out_lines.append(f"{new_c} {' '.join(parts[1:])}")
    return "\n".join(out_lines) + ("\n" if out_lines else "")


def extract_zip(zip_path: Path, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(dest)
    print(f"Extracted -> {dest}")


def split_names(names: list[str], ratios: tuple[float, float, float], seed: int) -> tuple[set[str], set[str], set[str]]:
    assert abs(sum(ratios) - 1.0) < 1e-6
    rng = random.Random(seed)
    shuffled = names.copy()
    rng.shuffle(shuffled)
    n = len(shuffled)
    n_train = int(n * ratios[0])
    n_val = int(n * ratios[1])
    n_test = n - n_train - n_val
    train_s = set(shuffled[:n_train])
    val_s = set(shuffled[n_train : n_train + n_val])
    test_s = set(shuffled[n_train + n_val : n_train + n_val + n_test])
    assert len(train_s) + len(val_s) + len(test_s) == n
    return train_s, val_s, test_s


def prepare_yolo(
    staging_yolo: Path,
    out_yolo: Path,
    train_ids: set[str],
    val_ids: set[str],
    test_ids: set[str],
) -> None:
    img_dir = staging_yolo / "train" / "images"
    lbl_dir = staging_yolo / "train" / "labels"
    for split, id_set in [("train", train_ids), ("val", val_ids), ("test", test_ids)]:
        (out_yolo / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_yolo / "labels" / split).mkdir(parents=True, exist_ok=True)

    stems = [p.stem for p in img_dir.glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}]
    for stem in stems:
        if stem in train_ids:
            sp = "train"
        elif stem in val_ids:
            sp = "val"
        elif stem in test_ids:
            sp = "test"
        else:
            raise RuntimeError(f"stem {stem} not in any split")

        img_paths = list(img_dir.glob(f"{stem}.*"))
        if not img_paths:
            continue
        src_img = img_paths[0]
        shutil.copy2(src_img, out_yolo / "images" / sp / src_img.name)

        src_lbl = lbl_dir / f"{stem}.txt"
        if src_lbl.exists():
            raw = src_lbl.read_text(encoding="utf-8", errors="replace")
            new_txt = remap_yolo_label_file(raw)
            (out_yolo / "labels" / sp / f"{stem}.txt").write_text(new_txt, encoding="utf-8")
        else:
            (out_yolo / "labels" / sp / f"{stem}.txt").write_text("", encoding="utf-8")

    # path: . — корень датасета = каталог этого data.yaml (переносимо Windows/macOS/Linux/Colab).
    # Абсолютный Windows-путь в path ломает Ultralytics на Linux (склейка с /content/datasets/...).
    yaml_text = f"""# Prepared by scripts/prepare_merged_dataset.py
path: .
train: images/train
val: images/val
test: images/test
nc: {len(CLASS_NAMES)}
names: {CLASS_NAMES}
"""
    (out_yolo / "data.yaml").write_text(yaml_text, encoding="utf-8")
    print(f"Wrote {out_yolo / 'data.yaml'}")


def coco_cat_id_to_new(old_cid: int) -> int | None:
    """COCO category_id from merged export: 1..28 -> yolo index 0..27; 0 unused."""
    if old_cid == 0:
        return None
    yidx = old_cid - 1
    if yidx not in YOLO_TO_FOUR:
        return None
    return YOLO_TO_FOUR[yidx]


def prepare_coco(
    staging_coco: Path,
    coco_json_path: Path,
    out_coco: Path,
    train_ids: set[str],
    val_ids: set[str],
    test_ids: set[str],
) -> None:
    raw = json.loads(coco_json_path.read_text(encoding="utf-8"))
    images: list[dict[str, Any]] = raw["images"]
    annotations: list[dict[str, Any]] = raw["annotations"]

    id_to_fname = {im["id"]: im["file_name"] for im in images}
    fname_to_id = {im["file_name"]: im["id"] for im in images}

    new_categories = [
        {"id": i, "name": n, "supercategory": "aerial"}
        for i, n in enumerate(CLASS_NAMES)
    ]

    def split_for_fname(fname: str) -> str:
        stem = Path(fname).stem
        if stem in train_ids:
            return "train"
        if stem in val_ids:
            return "val"
        if stem in test_ids:
            return "test"
        raise RuntimeError(f"file {fname} not in any split")

    # Partition images; new image ids are 1..N within each split file
    split_images: dict[str, list[dict]] = {"train": [], "val": [], "test": []}
    old_im_id_to_new: dict[tuple[str, int], int] = {}

    for im in images:
        fname = im["file_name"]
        sp = split_for_fname(fname)
        new_rec = {k: v for k, v in im.items() if k != "id"}
        idx = len(split_images[sp]) + 1
        new_rec["id"] = idx
        old_im_id_to_new[(sp, im["id"])] = idx
        split_images[sp].append(new_rec)

    split_anns: dict[str, list[dict]] = {"train": [], "val": [], "test": []}
    ann_id = 1
    for an in annotations:
        im_old = an["image_id"]
        fname = id_to_fname[im_old]
        sp = split_for_fname(fname)
        new_c = coco_cat_id_to_new(an["category_id"])
        if new_c is None:
            continue
        new_iid = old_im_id_to_new[(sp, im_old)]
        new_an = {
            "id": ann_id,
            "image_id": new_iid,
            "category_id": new_c,
            "bbox": an["bbox"],
            "area": an["area"],
            "iscrowd": an.get("iscrowd", 0),
        }
        if "segmentation" in an:
            new_an["segmentation"] = an["segmentation"]
        split_anns[sp].append(new_an)
        ann_id += 1

    train_dir = staging_coco / "train"
    for sp in ["train", "val", "test"]:
        (out_coco / "images" / sp).mkdir(parents=True, exist_ok=True)

    for im in images:
        fname = im["file_name"]
        sp = split_for_fname(fname)
        src = train_dir / fname
        if not src.exists():
            src = staging_coco / "train" / fname
        dst = out_coco / "images" / sp / fname
        shutil.copy2(src, dst)

    (out_coco / "annotations").mkdir(parents=True, exist_ok=True)
    for sp in ["train", "val", "test"]:
        coco_out = {
            "info": raw.get("info", {}),
            "licenses": raw.get("licenses", []),
            "categories": new_categories,
            "images": split_images[sp],
            "annotations": split_anns[sp],
        }
        out_path = out_coco / "annotations" / f"instances_{sp}.json"
        out_path.write_text(json.dumps(coco_out), encoding="utf-8")
        print(f"Wrote {out_path} ({len(split_images[sp])} images, {len(split_anns[sp])} anns)")

    print(f"COCO images root -> {out_coco / 'images'}")


def main() -> None:
    global YOLO_TO_FOUR
    parser = argparse.ArgumentParser(description="Prepare merged Roboflow zips for YOLO + COCO pipeline.")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "dataset",
        help="Folder containing *_v1.*.zip archives and class_map.yaml",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output root (default: dataset-root/prepared)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--skip-extract", action="store_true", help="Use existing staging dirs")
    args = parser.parse_args()

    root = args.dataset_root.resolve()
    class_map_path = root / "class_map.yaml"
    if not class_map_path.is_file():
        raise FileNotFoundError(f"Not found: {class_map_path}")
    YOLO_TO_FOUR = load_yolo_to_four_from_class_map(class_map_path)
    print(f"Class remap loaded from {class_map_path.name} ({len(YOLO_TO_FOUR)} indices)")

    out_root = (args.output or (root / "prepared")).resolve()
    staging = out_root / ".staging"
    staging_yolo = staging / "yolo"
    staging_coco = staging / "coco"

    yolo_zip = root / "Drone_detection_dataset_v1.yolov7pytorch.zip"
    coco_zip = root / "Drone_detection_dataset_v1.coco.zip"
    if not yolo_zip.is_file() or not coco_zip.is_file():
        raise FileNotFoundError(f"Expected {yolo_zip.name} and {coco_zip.name} in {root}")

    test_ratio = 1.0 - args.train_ratio - args.val_ratio
    if test_ratio < 0:
        raise ValueError("Ratios must sum to <= 1")

    if not args.skip_extract:
        if staging.exists():
            shutil.rmtree(staging)
        staging.mkdir(parents=True)
        extract_zip(yolo_zip, staging_yolo)
        extract_zip(coco_zip, staging_coco)
    else:
        if not staging_yolo.is_dir() or not staging_coco.is_dir():
            raise FileNotFoundError("Staging dirs missing; run without --skip-extract")

    img_dir = staging_yolo / "train" / "images"
    stems = sorted(
        p.stem for p in img_dir.glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
    )
    print(f"Found {len(stems)} images in YOLO staging")

    train_ids, val_ids, test_ids = split_names(
        stems, (args.train_ratio, args.val_ratio, test_ratio), args.seed
    )
    print(f"Split: train={len(train_ids)} val={len(val_ids)} test={len(test_ids)}")

    out_yolo = out_root / "yolo"
    out_coco = out_root / "dataset_coco"
    if out_yolo.exists():
        shutil.rmtree(out_yolo)
    if out_coco.exists():
        shutil.rmtree(out_coco)
    out_yolo.mkdir(parents=True)
    out_coco.mkdir(parents=True)

    prepare_yolo(staging_yolo, out_yolo, train_ids, val_ids, test_ids)
    prepare_coco(
        staging_coco,
        staging_coco / "train" / "_annotations.coco.json",
        out_coco,
        train_ids,
        val_ids,
        test_ids,
    )

    print("\nDone.")
    print(f"YOLO:  {out_yolo / 'data.yaml'}")
    print(f"COCO:  {out_coco / 'annotations'}")
    print("Point Colab DRIVE_ROOT dataset paths to these folders (or copy prepared/ to Google Drive).")


if __name__ == "__main__":
    main()

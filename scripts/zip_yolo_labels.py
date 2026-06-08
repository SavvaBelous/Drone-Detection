"""Pack all YOLO label files under ``dataset/prepared/yolo/labels`` into one zip.

Usage:
    python scripts/zip_yolo_labels.py
    python scripts/zip_yolo_labels.py --out dataset/prepared/yolo_labels_all.zip
"""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Zip YOLO labels tree (train/val/test).")
    parser.add_argument(
        "--labels-root",
        type=Path,
        default=Path("dataset/prepared/yolo/labels"),
        help="Path to labels directory",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("dataset/prepared/yolo_labels_all.zip"),
        help="Output zip path",
    )
    args = parser.parse_args()

    root = args.labels_root.resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Нет папки: {root}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    txt_files = sorted(root.rglob("*.txt"))
    if not txt_files:
        raise FileNotFoundError(f"Нет .txt в {root}")

    # В архиве: labels/train/*.txt, labels/val/*.txt, labels/test/*.txt
    with zipfile.ZipFile(args.out, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in txt_files:
            arc_name = p.relative_to(root.parent).as_posix()
            zf.write(p, arcname=arc_name)

    size_mb = args.out.stat().st_size / (1024**2)
    print(f"OK: {len(txt_files)} files -> {args.out} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()

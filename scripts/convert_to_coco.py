"""Convert YOLO format dataset to COCO JSON format.

Usage:
    python scripts/convert_to_coco.py \
        --yolo_dir /path/to/yolo_dataset \
        --output_dir /path/to/coco_dataset \
        --classes DRONE AIRPLANE HELICOPTER BIRD
"""

import argparse
import json
import shutil
from pathlib import Path

from PIL import Image
from tqdm import tqdm


CLASS_NAMES_DEFAULT = ['DRONE', 'AIRPLANE', 'HELICOPTER', 'BIRD']


def yolo_to_coco(
    yolo_data_dir: Path,
    coco_output_dir: Path,
    class_names: list[str],
    splits: list[str] | None = None,
) -> None:
    """Convert YOLO format dataset to COCO JSON format.

    Args:
        yolo_data_dir: Root of YOLO dataset (images/ and labels/ subdirs).
        coco_output_dir: Output directory for COCO format dataset.
        class_names: Ordered list of class names (index = class id).
        splits: List of split names to convert. Defaults to train/val/test.
    """
    if splits is None:
        splits = ['train', 'val', 'test']

    categories = [
        {'id': i, 'name': n, 'supercategory': 'aerial_object'}
        for i, n in enumerate(class_names)
    ]

    for split in splits:
        img_src = yolo_data_dir / 'images' / split
        lbl_src = yolo_data_dir / 'labels' / split

        # Also check 'valid' as Roboflow sometimes uses it
        if not img_src.exists():
            img_src = yolo_data_dir / 'images' / 'valid'
            lbl_src = yolo_data_dir / 'labels' / 'valid'
            if not img_src.exists():
                print(f'  Skip {split} — directory not found')
                continue

        img_dst = coco_output_dir / 'images' / split
        ann_dst = coco_output_dir / 'annotations'
        img_dst.mkdir(parents=True, exist_ok=True)
        ann_dst.mkdir(parents=True, exist_ok=True)

        coco: dict = {
            'info': {
                'description': f'Drone Detection Dataset — {split}',
                'version': '1.0',
                'year': 2025,
            },
            'images': [],
            'annotations': [],
            'categories': categories,
        }
        ann_id = 1
        skipped = 0

        all_imgs = sorted(
            list(img_src.glob('*.jpg'))
            + list(img_src.glob('*.jpeg'))
            + list(img_src.glob('*.png'))
        )

        for img_id, img_path in enumerate(
            tqdm(all_imgs, desc=f'  Converting {split}'), start=1
        ):
            try:
                with Image.open(img_path) as im:
                    w, h = im.size
            except Exception as e:
                print(f'  Warning: cannot open {img_path.name}: {e}')
                skipped += 1
                continue

            shutil.copy(img_path, img_dst / img_path.name)

            coco['images'].append({
                'id': img_id,
                'file_name': img_path.name,
                'width': w,
                'height': h,
            })

            lbl_path = lbl_src / f'{img_path.stem}.txt'
            if not lbl_path.exists():
                continue

            for line in lbl_path.read_text().strip().splitlines():
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                cls_id = int(parts[0])
                cx, cy, bw, bh = map(float, parts[1:])

                if not (0 < bw <= 1 and 0 < bh <= 1):
                    continue

                x_min = (cx - bw / 2) * w
                y_min = (cy - bh / 2) * h
                pw    = bw * w
                ph    = bh * h
                area  = pw * ph

                coco['annotations'].append({
                    'id':          ann_id,
                    'image_id':    img_id,
                    'category_id': cls_id,
                    'bbox':        [
                        round(x_min, 2),
                        round(y_min, 2),
                        round(pw,    2),
                        round(ph,    2),
                    ],
                    'area':     round(area, 2),
                    'iscrowd':  0,
                    'segmentation': [],
                })
                ann_id += 1

        out_json = ann_dst / f'instances_{split}.json'
        with open(out_json, 'w') as f:
            json.dump(coco, f, separators=(',', ':'))

        n_imgs = len(coco['images'])
        n_anns = len(coco['annotations'])
        print(
            f'  {split}: {n_imgs} images, {n_anns} annotations'
            + (f', {skipped} skipped' if skipped else '')
            + f' → {out_json.name}'
        )


def main() -> None:
    parser = argparse.ArgumentParser(description='Convert YOLO dataset to COCO format')
    parser.add_argument('--yolo_dir',   required=True, help='Path to YOLO format dataset root')
    parser.add_argument('--output_dir', required=True, help='Output path for COCO dataset')
    parser.add_argument('--classes',    nargs='+', default=CLASS_NAMES_DEFAULT,
                        help='Class names in order (space-separated)')
    parser.add_argument('--splits',     nargs='+', default=['train', 'val', 'test'],
                        help='Dataset splits to convert')
    args = parser.parse_args()

    yolo_dir   = Path(args.yolo_dir)
    output_dir = Path(args.output_dir)

    if not yolo_dir.exists():
        raise FileNotFoundError(f'YOLO dataset not found: {yolo_dir}')

    print(f'Converting YOLO → COCO')
    print(f'  Source:  {yolo_dir}')
    print(f'  Target:  {output_dir}')
    print(f'  Classes: {args.classes}')
    print(f'  Splits:  {args.splits}')
    print()

    yolo_to_coco(yolo_dir, output_dir, args.classes, args.splits)

    print('\nConversion complete!')
    print(f'COCO dataset at: {output_dir}')


if __name__ == '__main__':
    main()

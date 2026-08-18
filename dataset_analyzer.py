#!/usr/bin/env python3
"""
Dataset Analyzer
================

Deep recursive object-detection dataset analyzer.

Supported annotation formats:
- Client nested JSON:
    {
      "file_name": "image.jpg",
      "dimensions": [HEIGHT, WIDTH],
      "data": {
        "ball": [
          {"entire": {"rect": [X1, Y1, X2, Y2]}}
        ]
      }
    }
- Generic JSON bbox annotations
- YOLO TXT
- COCO JSON

Client nested JSON coordinates are interpreted exactly as:
    rect = [x1, y1, x2, y2]
and:
    dimensions = [height, width]

No coordinate conversion is applied to that client format.

Outputs:
- dataset_summary.xlsx
- image_statistics.csv
- bbox_statistics.csv
- folder_statistics.csv
- duplicate_images.csv
- invalid_annotations.csv
- resize_impact_<N>px.csv
- blurry_images.csv
- report.html
- plots/*.png

Example:
    python dataset_analyzer.py --root /path/to/dataset --out ./analysis
"""

import argparse
import gc
import json
import pickle
import sys
import traceback
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageOps

Image.MAX_IMAGE_PIXELS = None

IMAGE_EXTS = {
    ".jpg", ".jpeg", ".png", ".bmp",
    ".tif", ".tiff", ".webp"
}

UNSUPPORTED_EXT_WARN = {
    ".jfif", ".heic", ".avif", ".gif", ".psd"
}

SIZE_BUCKETS = [
    ("Tiny", 0.0, 0.002),
    ("Small", 0.002, 0.01),
    ("Medium", 0.01, 0.05),
    ("Large", 0.05, 1.01),
]

BRIGHTNESS_BUCKETS = [
    ("Very Dark", 0, 50),
    ("Dark", 50, 100),
    ("Normal", 100, 180),
    ("Bright", 180, 256),
]

BLUR_BUCKETS = [
    ("Very Blurry", 0, 30),
    ("Blurry", 30, 80),
    ("Acceptable", 80, 150),
    ("Sharp", 150, float("inf")),
]

EDGE_TOUCH_MARGIN_PX = 2


try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable


try:
    from skimage.measure import shannon_entropy as sk_entropy
except ImportError:
    sk_entropy = None


def bucket_for(value, buckets):
    for name, lo, hi in buckets:
        if lo <= value < hi:
            return name
    return buckets[-1][0]


def safe_relpath(path, root):
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def custom_dhash(gray_small):
    diff = gray_small[:, 1:] > gray_small[:, :-1]
    h = 0
    for value in diff.flatten():
        h = (h << 1) | int(value)
    return h


def custom_phash(gray32):
    dct = cv2.dct(np.float32(gray32))
    dct_low = dct[:8, :8]
    med = np.median(dct_low[1:, 1:])
    bits = dct_low > med
    h = 0
    for value in bits.flatten():
        h = (h << 1) | int(value)
    return h


def hamming(a, b):
    return bin(a ^ b).count("1")


def compute_entropy(gray):
    if sk_entropy is not None:
        try:
            return float(sk_entropy(gray))
        except Exception:
            pass

    hist, _ = np.histogram(
        gray,
        bins=256,
        range=(0, 256)
    )

    prob = hist / max(hist.sum(), 1)
    prob = prob[prob > 0]

    return float(-np.sum(prob * np.log2(prob)))


def compute_image_hashes(gray):
    small_d = cv2.resize(
        gray, (9, 8), interpolation=cv2.INTER_AREA
    )
    small_p = cv2.resize(
        gray, (32, 32), interpolation=cv2.INTER_AREA
    )
    return custom_dhash(small_d), custom_phash(small_p)


def analyze_image_pixels(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    b, g, r = cv2.split(img_bgr.astype(np.float64))

    brightness = float(gray.mean())
    contrast = float(gray.std())

    blur_score = float(
        cv2.Laplacian(gray, cv2.CV_64F).var()
    )

    saturation = float(hsv[:, :, 1].mean())
    entropy = compute_entropy(gray)
    dhash_value, phash_value = compute_image_hashes(gray)

    dark_frac = float((gray < 30).mean())
    bright_frac = float((gray > 225).mean())

    if bright_frac > 0.35:
        exposure = "Overexposed"
    elif dark_frac > 0.35:
        exposure = "Underexposed"
    else:
        exposure = "Normal"

    return {
        "mean_r": float(r.mean()),
        "mean_g": float(g.mean()),
        "mean_b": float(b.mean()),
        "std_r": float(r.std()),
        "std_g": float(g.std()),
        "std_b": float(b.std()),
        "brightness": brightness,
        "brightness_bucket": bucket_for(
            brightness, BRIGHTNESS_BUCKETS
        ),
        "contrast": contrast,
        "blur_score": blur_score,
        "blur_bucket": bucket_for(
            blur_score, BLUR_BUCKETS
        ),
        "saturation": saturation,
        "entropy": entropy,
        "exposure": exposure,
        "dhash": dhash_value,
        "phash": phash_value,
    }


def _find_rects_in_obj(obj, region_type=""):
    found = []

    if isinstance(obj, dict):
        rect = obj.get("rect")

        if (
            isinstance(rect, (list, tuple))
            and len(rect) == 4
        ):
            found.append((
                list(rect),
                region_type or "region"
            ))

        for key, value in obj.items():
            if key == "rect":
                continue

            if isinstance(value, (dict, list)):
                found.extend(
                    _find_rects_in_obj(value, key)
                )

    elif isinstance(obj, list):
        for item in obj:
            found.extend(
                _find_rects_in_obj(item, region_type)
            )

    return found


def parse_nested_data_json_label(json_path, img_w, img_h):
    """
    Parse the client format exactly.

    dimensions = [HEIGHT, WIDTH]
    rect       = [X1, Y1, X2, Y2]
    """

    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data_full = json.load(f)
    except Exception as e:
        return [], f"invalid_json: {e}", None

    data = data_full.get("data")

    if not isinstance(data, dict):
        return [], "no_recognizable_object_list", None

    warning = None
    dims = data_full.get("dimensions")

    if isinstance(dims, (list, tuple)) and len(dims) == 2:
        try:
            declared_height = float(dims[0])
            declared_width = float(dims[1])

            if (
                abs(declared_width - img_w) > 2
                or abs(declared_height - img_h) > 2
            ):
                warning = (
                    "declared_dimensions_mismatch("
                    f"json={declared_width}x{declared_height},"
                    f"actual_image={img_w}x{img_h})"
                )
        except Exception:
            warning = "invalid_dimensions_field"

    boxes = []

    for cls_name, obj_list in data.items():
        if not isinstance(obj_list, list):
            continue

        for object_index, obj in enumerate(obj_list):
            if not isinstance(obj, dict):
                continue

            rects = _find_rects_in_obj(obj)

            for raw_rect, region_type in rects:
                if len(raw_rect) != 4:
                    continue

                try:
                    x1 = float(raw_rect[0])
                    y1 = float(raw_rect[1])
                    x2 = float(raw_rect[2])
                    y2 = float(raw_rect[3])
                except (ValueError, TypeError):
                    continue

                boxes.append({
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "cls": str(cls_name),
                    "region_type": region_type,
                    "object_index": object_index,
                })

    if not boxes:
        return [], "empty_labels", warning

    return boxes, None, warning


BBOX_KEY_SETS = [
    ("xmin", "ymin", "xmax", "ymax"),
    ("x1", "y1", "x2", "y2"),
    ("x_min", "y_min", "x_max", "y_max"),
    ("x", "y", "width", "height"),
    ("x", "y", "w", "h"),
    ("left", "top", "right", "bottom"),
]

CLASS_KEYS = [
    "class",
    "class_name",
    "label",
    "category",
    "name",
    "category_name",
    "cls",
]

OBJECT_LIST_KEYS = [
    "objects",
    "annotations",
    "shapes",
    "boxes",
    "labels",
    "bboxes",
    "detections",
]


def _extract_bbox_from_dict(d):
    if (
        isinstance(d.get("bbox"), (list, tuple))
        and len(d["bbox"]) == 4
    ):
        return [float(v) for v in d["bbox"]], "bbox_field"

    for keys in BBOX_KEY_SETS:
        if all(k in d for k in keys):
            return [float(d[k]) for k in keys], "keys"

    if (
        isinstance(d.get("points"), (list, tuple))
        and len(d["points"]) >= 2
    ):
        pts = np.array(d["points"], dtype=float)
        return [
            pts[:, 0].min(),
            pts[:, 1].min(),
            pts[:, 0].max(),
            pts[:, 1].max(),
        ], "polygon_points"

    return None, None


def _resolve_xyxy(vals, img_w, img_h):
    v = list(vals)

    normalized = all(0.0 <= x <= 1.5 for x in v)

    if normalized:
        cx = v[0] * img_w
        cy = v[1] * img_h
        bw = v[2] * img_w
        bh = v[3] * img_h
        return [
            cx - bw / 2,
            cy - bh / 2,
            cx + bw / 2,
            cy + bh / 2,
        ]

    x1, y1, a, b = v

    if (
        a > x1
        and b > y1
        and a <= img_w * 1.05
        and b <= img_h * 1.05
    ):
        return [x1, y1, a, b]

    return [x1, y1, x1 + a, y1 + b]


def parse_generic_json_label(json_path, img_w, img_h):
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        return [], f"invalid_json: {e}"

    objs = None

    if isinstance(data, list):
        objs = data

    elif isinstance(data, dict):
        for key in OBJECT_LIST_KEYS:
            if isinstance(data.get(key), list):
                objs = data[key]
                break

        if objs is None:
            bbox, _ = _extract_bbox_from_dict(data)
            if bbox is not None:
                objs = [data]

    if objs is None:
        return [], "no_recognizable_object_list"

    if not objs:
        return [], "empty_labels"

    boxes = []

    for obj in objs:
        if not isinstance(obj, dict):
            continue

        raw, _ = _extract_bbox_from_dict(obj)

        if raw is None:
            continue

        x1, y1, x2, y2 = _resolve_xyxy(
            raw, img_w, img_h
        )

        cls = "object"

        for key in CLASS_KEYS:
            if key in obj:
                cls = str(obj[key])
                break

        boxes.append({
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
            "cls": cls,
        })

    return boxes, None


def parse_yolo_label(txt_path, img_w, img_h, class_names=None):
    try:
        with open(txt_path, "r", encoding="utf-8") as f:
            lines = [
                line.strip()
                for line in f.read().splitlines()
                if line.strip()
            ]
    except Exception as e:
        return [], f"invalid_txt: {e}"

    if not lines:
        return [], "empty_labels"

    boxes = []

    for line in lines:
        parts = line.split()

        if len(parts) < 5:
            continue

        try:
            cls_idx = int(float(parts[0]))
            cx, cy, bw, bh = [
                float(x) for x in parts[1:5]
            ]
        except ValueError:
            continue

        x1 = (cx - bw / 2) * img_w
        y1 = (cy - bh / 2) * img_h
        x2 = (cx + bw / 2) * img_w
        y2 = (cy + bh / 2) * img_h

        cls = (
            class_names[cls_idx]
            if class_names and 0 <= cls_idx < len(class_names)
            else str(cls_idx)
        )

        boxes.append({
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
            "cls": cls,
        })

    return boxes, None


def load_coco_json(coco_path):
    with open(coco_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    id2name = {
        c["id"]: c.get("name", str(c["id"]))
        for c in data.get("categories", [])
    }

    imgid2file = {
        im["id"]: im["file_name"]
        for im in data.get("images", [])
    }

    per_image = defaultdict(list)

    for ann in data.get("annotations", []):
        fname = imgid2file.get(ann.get("image_id"))

        if fname is None:
            continue

        bbox = ann.get("bbox")

        if not bbox or len(bbox) != 4:
            continue

        x, y, w, h = bbox

        per_image[fname].append({
            "x1": x,
            "y1": y,
            "x2": x + w,
            "y2": y + h,
            "cls": id2name.get(
                ann.get("category_id"),
                str(ann.get("category_id"))
            ),
        })

    return per_image, set(imgid2file.values())


def compute_letterbox_transform(img_w, img_h, target_size):
    if not img_w or not img_h:
        return 1.0, 0.0, 0.0, target_size, target_size

    scale = min(
        target_size / img_w,
        target_size / img_h
    )

    content_w = img_w * scale
    content_h = img_h * scale

    pad_left = (target_size - content_w) / 2
    pad_top = (target_size - content_h) / 2

    return scale, pad_left, pad_top, content_w, content_h


def compute_bbox_metrics(
    box,
    img_w,
    img_h,
    target_size=320,
    tiny_px_threshold=4.0
):
    x1 = box["x1"]
    y1 = box["y1"]
    x2 = box["x2"]
    y2 = box["y2"]

    width = x2 - x1
    height = y2 - y1

    issues = []

    if width <= 0 or height <= 0:
        issues.append("zero_or_negative_size")

    if min(x1, y1, x2, y2) < 0:
        issues.append("negative_coordinate")

    if x2 > img_w + 1 or y2 > img_h + 1:
        issues.append("exceeds_image_bounds")

    if x1 > img_w or y1 > img_h or x2 < 0 or y2 < 0:
        issues.append("bbox_outside_image")

    area = max(width, 0) * max(height, 0)
    image_area = max(img_w * img_h, 1)
    relative_area = area / image_area

    aspect_ratio = width / height if height > 0 else 0

    center_x = x1 + width / 2
    center_y = y1 + height / 2

    touching_edge = (
        x1 <= EDGE_TOUCH_MARGIN_PX
        or y1 <= EDGE_TOUCH_MARGIN_PX
        or x2 >= img_w - EDGE_TOUCH_MARGIN_PX
        or y2 >= img_h - EDGE_TOUCH_MARGIN_PX
    )

    truncated = touching_edge and relative_area > 0

    scale, pad_left, pad_top, _, _ = compute_letterbox_transform(
        img_w, img_h, target_size
    )

    resized_width = width * scale
    resized_height = height * scale
    resized_area = resized_width * resized_height

    resized_x1 = x1 * scale + pad_left
    resized_y1 = y1 * scale + pad_top
    resized_x2 = x2 * scale + pad_left
    resized_y2 = y2 * scale + pad_top

    resized_relative_area_pct = (
        100 * resized_area / (target_size * target_size)
    )

    return {
        "cls": box["cls"],
        "region_type": box.get("region_type", ""),
        "object_index": box.get("object_index", ""),
        "x1": x1,
        "y1": y1,
        "x2": x2,
        "y2": y2,
        "width": width,
        "height": height,
        "area": area,
        "rel_area_pct": relative_area * 100,
        "aspect_ratio": aspect_ratio,
        "center_x_rel": center_x / max(img_w, 1),
        "center_y_rel": center_y / max(img_h, 1),
        "touching_edge": touching_edge,
        "truncated": truncated,
        "size_bucket": bucket_for(
            relative_area, SIZE_BUCKETS
        ),
        "issues": ";".join(issues),
        "valid": len(issues) == 0,
        f"resized_{target_size}_width_px": round(
            resized_width, 2
        ),
        f"resized_{target_size}_height_px": round(
            resized_height, 2
        ),
        f"resized_{target_size}_area_px": round(
            resized_area, 2
        ),
        f"resized_{target_size}_rel_area_pct": round(
            resized_relative_area_pct, 4
        ),
        f"resized_{target_size}_x1": round(
            resized_x1, 2
        ),
        f"resized_{target_size}_y1": round(
            resized_y1, 2
        ),
        f"resized_{target_size}_x2": round(
            resized_x2, 2
        ),
        f"resized_{target_size}_y2": round(
            resized_y2, 2
        ),
        f"resized_{target_size}_size_bucket": bucket_for(
            relative_area, SIZE_BUCKETS
        ),
        f"tiny_after_resize_{target_size}": (
            resized_width < tiny_px_threshold
            or resized_height < tiny_px_threshold
        ),
    }


def detect_top_folder_format(folder):
    json_files = list(folder.rglob("*.json"))

    img_files = [
        p for p in folder.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    ]

    img_stems = {p.stem for p in img_files}

    for jf in json_files:
        try:
            with open(jf, "r", encoding="utf-8") as f:
                head = f.read(4096)

            if (
                '"images"' in head
                and '"annotations"' in head
                and '"categories"' in head
            ):
                return "coco", jf
        except Exception:
            continue

    for jf in json_files:
        if jf.stem not in img_stems:
            continue

        try:
            with open(jf, "r", encoding="utf-8") as f:
                data = json.load(f)

            if (
                isinstance(data, dict)
                and isinstance(data.get("data"), dict)
            ):
                return "json_nested", None
        except Exception:
            continue

    txt_files = list(folder.rglob("*.txt"))

    if txt_files and img_files:
        txt_stems = {
            p.stem for p in txt_files
            if p.name.lower() not in {
                "classes.txt", "data.yaml"
            }
        }

        if img_stems & txt_stems:
            return "yolo", None

    json_stems = {p.stem for p in json_files}

    if img_stems & json_stems:
        return "json", None

    return "none", None


def load_yolo_classnames(folder):
    for name in ("classes.txt", "obj.names", "names.txt"):
        p = folder / name

        if p.exists():
            try:
                return [
                    line.strip()
                    for line in p.read_text(
                        encoding="utf-8"
                    ).splitlines()
                    if line.strip()
                ]
            except Exception:
                pass

    return None


@dataclass
class AnalyzerConfig:
    root: Path
    out_dir: Path
    hash_threshold: int = 5
    sample: int = 0
    make_plots: bool = True
    make_html: bool = True
    checkpoint: bool = True
    resume: bool = False
    target_size: int = 320
    tiny_px_threshold: float = 4.0


class DatasetAnalyzer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.root = cfg.root

        self.image_rows = []
        self.bbox_rows = []
        self.folder_rows = []

        self.missing_labels = []
        self.missing_images = []
        self.invalid_json_rows = []
        self.duplicate_rows = []
        self.naming_issue_rows = []
        self.errors = []

        self._hash_index = {}
        self._name_index = defaultdict(list)
        self._done_relpaths = set()

        self._checkpoint_path = cfg.out_dir / "_checkpoint.pkl"

    def run(self):
        if self.cfg.resume:
            self._load_checkpoint()

        print(f"Scanning dataset root: {self.root}")

        top_folders = self.top_level_folders()

        if not top_folders:
            top_folders = [self.root]

        for folder in top_folders:
            self._process_top_folder(folder)

        self._cross_folder_duplicates()
        self._build_folder_aggregates()

        print(
            f"Done. Images analyzed: {len(self.image_rows)} | "
            f"Boxes: {len(self.bbox_rows)}"
        )

        if self._checkpoint_path.exists():
            try:
                self._checkpoint_path.unlink()
            except Exception:
                pass

    def top_level_folders(self):
        return sorted(
            p for p in self.root.iterdir() if p.is_dir()
        )

    def _save_checkpoint(self):
        try:
            self.cfg.out_dir.mkdir(
                parents=True,
                exist_ok=True
            )

            state = {
                "image_rows": self.image_rows,
                "bbox_rows": self.bbox_rows,
                "missing_labels": self.missing_labels,
                "missing_images": self.missing_images,
                "invalid_json_rows": self.invalid_json_rows,
                "naming_issue_rows": self.naming_issue_rows,
                "errors": self.errors,
                "hash_index": self._hash_index,
                "name_index": dict(self._name_index),
            }

            tmp = self._checkpoint_path.with_suffix(".tmp")

            with open(tmp, "wb") as f:
                pickle.dump(state, f)

            tmp.replace(self._checkpoint_path)
        except Exception as e:
            print(f"Checkpoint warning: {e}")

    def _load_checkpoint(self):
        if not self._checkpoint_path.exists():
            return

        try:
            with open(self._checkpoint_path, "rb") as f:
                state = pickle.load(f)

            for attr in (
                "image_rows",
                "bbox_rows",
                "missing_labels",
                "missing_images",
                "invalid_json_rows",
                "naming_issue_rows",
                "errors",
            ):
                setattr(
                    self,
                    attr,
                    state.get(attr, [])
                )

            self._hash_index = state.get(
                "hash_index", {}
            )

            self._name_index = defaultdict(
                list,
                state.get("name_index", {})
            )

            self._done_relpaths = {
                r["relpath"]
                for r in self.image_rows
                if r.get("relpath")
            }

            print(
                f"Resumed: {len(self._done_relpaths)} images."
            )
        except Exception as e:
            print(f"Could not load checkpoint: {e}")

    def _process_top_folder(self, folder):
        fmt, coco_path = detect_top_folder_format(folder)

        print(
            f"\nFolder: {folder.name} -> "
            f"annotation format: {fmt}"
        )

        coco_lookup = {}

        if fmt == "coco" and coco_path:
            try:
                coco_lookup, _ = load_coco_json(coco_path)
            except Exception as e:
                self.invalid_json_rows.append({
                    "file": str(coco_path),
                    "error": str(e)
                })

        yolo_classes = (
            load_yolo_classnames(folder)
            if fmt == "yolo"
            else None
        )

        label_index = None

        if fmt in {"json", "json_nested", "yolo"}:
            label_index = self._build_label_index(
                folder,
                ".txt" if fmt == "yolo" else ".json"
            )

        all_images = [
            p for p in folder.rglob("*")
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS
        ]

        if self.cfg.sample:
            all_images = all_images[:self.cfg.sample]

        all_images = [
            p for p in all_images
            if safe_relpath(p, self.root)
            not in self._done_relpaths
        ]

        label_files_seen = set()

        iterator = tqdm(
            all_images,
            desc=folder.name,
            total=len(all_images)
        )

        for i, img_path in enumerate(iterator, start=1):
            self._process_one_image(
                img_path,
                folder,
                fmt,
                coco_lookup,
                yolo_classes,
                label_files_seen,
                label_index
            )

            if i % 100 == 0:
                gc.collect()

            if self.cfg.checkpoint and i % 250 == 0:
                self._save_checkpoint()

        if self.cfg.checkpoint:
            self._save_checkpoint()

        if fmt in {"json", "json_nested", "yolo"}:
            ext = (
                ".json"
                if fmt in {"json", "json_nested"}
                else ".txt"
            )

            all_labels = [
                p for p in folder.rglob(f"*{ext}")
                if p.name.lower() not in {
                    "classes.txt",
                    "obj.names",
                    "names.txt"
                }
            ]

            img_stems = {p.stem for p in all_images}

            for lp in all_labels:
                if lp.stem not in img_stems:
                    self.missing_images.append({
                        "label_file": safe_relpath(
                            lp, self.root
                        ),
                        "top_folder": folder.name,
                    })

        self._check_naming(all_images, folder)

    def _build_label_index(self, folder, ext):
        index = {}

        skip_names = {
            "classes.txt",
            "obj.names",
            "names.txt"
        }

        for lp in folder.rglob(f"*{ext}"):
            if lp.name.lower() in skip_names:
                continue

            if lp.stem in index:
                self.naming_issue_rows.append({
                    "file": (
                        f"{safe_relpath(index[lp.stem], self.root)}; "
                        f"{safe_relpath(lp, self.root)}"
                    ),
                    "top_folder": folder.name,
                    "issues":
                        "duplicate_label_stem_multiple_files",
                })
                continue

            index[lp.stem] = lp

        return index

    def _process_one_image(
        self,
        img_path,
        top_folder,
        fmt,
        coco_lookup,
        yolo_classes,
        label_files_seen,
        label_index
    ):
        relpath = safe_relpath(img_path, self.root)
        subfolder = safe_relpath(img_path.parent, self.root)
        class_folder = img_path.parent.name

        try:
            img_bgr = cv2.imread(
                str(img_path),
                cv2.IMREAD_COLOR
            )

            if img_bgr is None:
                pil_img = Image.open(img_path)
                pil_img = ImageOps.exif_transpose(
                    pil_img
                ).convert("RGB")

                img_bgr = cv2.cvtColor(
                    np.array(pil_img),
                    cv2.COLOR_RGB2BGR
                )

            h, w = img_bgr.shape[:2]

            max_stat_side = 4000
            stat_img = img_bgr

            if max(h, w) > max_stat_side:
                scale = max_stat_side / max(h, w)
                stat_img = cv2.resize(
                    img_bgr,
                    (
                        max(1, int(w * scale)),
                        max(1, int(h * scale))
                    ),
                    interpolation=cv2.INTER_AREA
                )

        except Exception as e:
            self.errors.append({
                "file": relpath,
                "error": f"corrupt_or_unreadable: {e}"
            })

            self.image_rows.append({
                "relpath": relpath,
                "top_folder": top_folder.name,
                "subfolder": subfolder,
                "class_folder": class_folder,
                "corrupt": True,
                "width": None,
                "height": None,
                "n_objects": 0,
            })

            return

        try:
            pixel_stats = analyze_image_pixels(stat_img)
            aspect = w / h if h else 0

            orientation = (
                "Square"
                if abs(aspect - 1) < 0.02
                else ("Landscape" if aspect > 1 else "Portrait")
            )

            ts = self.cfg.target_size

            (
                lb_scale,
                _,
                _,
                lb_content_w,
                lb_content_h
            ) = compute_letterbox_transform(w, h, ts)

            pad_pct = (
                100
                * (
                    1
                    - (lb_content_w * lb_content_h)
                    / (ts * ts)
                )
            )

            row = {
                "relpath": relpath,
                "top_folder": top_folder.name,
                "subfolder": subfolder,
                "class_folder": class_folder,
                "corrupt": False,
                "width": w,
                "height": h,
                "resolution": f"{w}x{h}",
                "aspect_ratio": round(aspect, 4),
                "orientation": orientation,
                "file_size_kb": round(
                    img_path.stat().st_size / 1024,
                    2
                ),
                **pixel_stats,
                f"letterbox_{ts}_scale": round(
                    lb_scale, 5
                ),
                f"letterbox_{ts}_content_w_px": round(
                    lb_content_w, 2
                ),
                f"letterbox_{ts}_content_h_px": round(
                    lb_content_h, 2
                ),
                f"letterbox_{ts}_pad_pct_of_canvas": round(
                    pad_pct, 2
                ),
            }

            self._hash_index.setdefault(
                row["phash"], []
            ).append(relpath)

            (
                boxes,
                label_relpath,
                ann_error,
                ann_warning
            ) = self._load_boxes_for_image(
                img_path,
                fmt,
                coco_lookup,
                yolo_classes,
                w,
                h,
                label_index
            )

            if label_relpath:
                label_files_seen.add(label_relpath)

            if ann_warning:
                self.invalid_json_rows.append({
                    "file": label_relpath or relpath,
                    "error": ann_warning
                })

            if ann_error == "missing":
                self.missing_labels.append({
                    "image_file": relpath,
                    "top_folder": top_folder.name,
                })
                row["n_objects"] = 0

            elif ann_error:
                self.invalid_json_rows.append({
                    "file": label_relpath or relpath,
                    "error": ann_error
                })
                row["n_objects"] = 0

            else:
                row["n_objects"] = len(boxes)

                for box in boxes:
                    metrics = compute_bbox_metrics(
                        box,
                        w,
                        h,
                        target_size=ts,
                        tiny_px_threshold=self.cfg.tiny_px_threshold
                    )

                    metrics.update({
                        "image_relpath": relpath,
                        "top_folder": top_folder.name,
                        "subfolder": subfolder,
                    })

                    self.bbox_rows.append(metrics)

                    if metrics["issues"]:
                        self.invalid_json_rows.append({
                            "file": label_relpath or relpath,
                            "error": metrics["issues"],
                            "class": metrics["cls"],
                            "image": relpath,
                        })

            self.image_rows.append(row)

        except Exception as e:
            self.errors.append({
                "file": relpath,
                "error": (
                    f"processing_failed: "
                    f"{type(e).__name__}: {e}"
                )
            })
        finally:
            del img_bgr

    def _load_boxes_for_image(
        self,
        img_path,
        fmt,
        coco_lookup,
        yolo_classes,
        w,
        h,
        label_index=None
    ):
        if fmt == "coco":
            boxes = coco_lookup.get(img_path.name)

            if boxes is None:
                return [], None, "missing", None

            return boxes, img_path.name, None, None

        if fmt == "yolo":
            txt_path = (
                (label_index or {}).get(img_path.stem)
                or img_path.with_suffix(".txt")
            )

            if not txt_path.exists():
                return [], None, "missing", None

            boxes, err = parse_yolo_label(
                txt_path,
                w,
                h,
                yolo_classes
            )

            return (
                boxes,
                safe_relpath(txt_path, self.root),
                err,
                None
            )

        if fmt == "json_nested":
            json_path = (
                (label_index or {}).get(img_path.stem)
                or img_path.with_suffix(".json")
            )

            if not json_path.exists():
                return [], None, "missing", None

            boxes, err, warning = parse_nested_data_json_label(
                json_path,
                w,
                h
            )

            return (
                boxes,
                safe_relpath(json_path, self.root),
                err,
                warning
            )

        if fmt == "json":
            json_path = (
                (label_index or {}).get(img_path.stem)
                or img_path.with_suffix(".json")
            )

            if not json_path.exists():
                return [], None, "missing", None

            boxes, err = parse_generic_json_label(
                json_path,
                w,
                h
            )

            return (
                boxes,
                safe_relpath(json_path, self.root),
                err,
                None
            )

        return [], None, "missing", None

    def _check_naming(self, images, folder):
        seen_lower = defaultdict(list)

        for p in images:
            self._name_index[p.name.lower()].append(
                safe_relpath(p, self.root)
            )

            seen_lower[p.name.lower()].append(p)

            issues = []

            if any(ord(ch) > 127 for ch in p.stem):
                issues.append("non_ascii_filename")

            if p.suffix.lower() in UNSUPPORTED_EXT_WARN:
                issues.append("unsupported_or_risky_extension")

            if issues:
                self.naming_issue_rows.append({
                    "file": safe_relpath(p, self.root),
                    "top_folder": folder.name,
                    "issues": ";".join(issues),
                })

        for paths in seen_lower.values():
            if len(paths) > 1:
                self.naming_issue_rows.append({
                    "file": "; ".join(
                        safe_relpath(x, self.root)
                        for x in paths
                    ),
                    "top_folder": folder.name,
                    "issues": "duplicate_filename_same_folder",
                })

    def _cross_folder_duplicates(self):
        print("Computing duplicate / near-duplicate groups...")

        items = list(self._hash_index.items())

        for h, paths in items:
            if len(paths) > 1:
                for p in paths:
                    self.duplicate_rows.append({
                        "file": p,
                        "hash": f"{h:016x}",
                        "match_type":
                            "exact_or_phash_identical",
                        "group_size": len(paths),
                        "group_members": "; ".join(paths),
                    })

        hashes = [h for h, _ in items]
        n = len(hashes)
        max_pairwise = 6000

        if n <= max_pairwise:
            for i in range(n):
                for j in range(i + 1, n):
                    d = hamming(hashes[i], hashes[j])

                    if 0 < d <= self.cfg.hash_threshold:
                        for p1 in items[i][1]:
                            for p2 in items[j][1]:
                                self.duplicate_rows.append({
                                    "file": p1,
                                    "hash":
                                        f"{hashes[i]:016x}",
                                    "match_type":
                                        f"near_duplicate(d={d})",
                                    "group_size": 2,
                                    "group_members":
                                        f"{p1}; {p2}",
                                })
        else:
            print(
                f"Skipping exhaustive near-duplicate search: "
                f"{n} unique hashes > {max_pairwise}."
            )

        for paths in self._name_index.values():
            folders = {
                Path(p).parts[0] for p in paths
            }

            if len(paths) > 1 and len(folders) > 1:
                self.naming_issue_rows.append({
                    "file": "; ".join(paths),
                    "top_folder": "; ".join(sorted(folders)),
                    "issues": "duplicate_filename_across_folders",
                })

    def _build_folder_aggregates(self):
        if not self.image_rows:
            return

        img_df = pd.DataFrame(self.image_rows)

        box_df = (
            pd.DataFrame(self.bbox_rows)
            if self.bbox_rows
            else pd.DataFrame(
                columns=["top_folder", "subfolder", "cls"]
            )
        )

        for (top, sub), group in img_df.groupby(
            ["top_folder", "subfolder"],
            dropna=False
        ):
            valid = group[group["corrupt"] == False]

            if not box_df.empty:
                boxes = box_df[
                    (box_df["top_folder"] == top)
                    & (box_df["subfolder"] == sub)
                ]
            else:
                boxes = box_df

            self.folder_rows.append({
                "top_folder": top,
                "subfolder": sub,
                "images": len(group),
                "corrupt_images": int(group["corrupt"].sum()),
                "objects": len(boxes),
                "avg_objects_per_image": round(
                    len(boxes) / max(len(valid), 1),
                    3
                ),
                "mean_width": round(
                    valid["width"].mean(), 1
                ) if len(valid) else None,
                "mean_height": round(
                    valid["height"].mean(), 1
                ) if len(valid) else None,
                "mean_brightness": round(
                    valid["brightness"].mean(), 2
                ) if len(valid) else None,
                "mean_contrast": round(
                    valid["contrast"].mean(), 2
                ) if len(valid) else None,
                "mean_blur_score": round(
                    valid["blur_score"].mean(), 2
                ) if len(valid) else None,
                "pct_blurry": round(
                    100 * valid["blur_bucket"].isin(
                        ["Very Blurry", "Blurry"]
                    ).mean(),
                    2
                ) if len(valid) else None,
                "pct_over_under_exposed": round(
                    100 * (
                        valid["exposure"] != "Normal"
                    ).mean(),
                    2
                ) if len(valid) else None,
            })


class ResultFrames:
    def __init__(self, analyzer):
        self.images = pd.DataFrame(analyzer.image_rows)
        self.bboxes = pd.DataFrame(analyzer.bbox_rows)
        self.folders = pd.DataFrame(analyzer.folder_rows)

        self.missing_labels = pd.DataFrame(
            analyzer.missing_labels
        )
        self.missing_images = pd.DataFrame(
            analyzer.missing_images
        )
        self.invalid = pd.DataFrame(
            analyzer.invalid_json_rows
        )
        self.duplicates = pd.DataFrame(
            analyzer.duplicate_rows
        )
        self.naming = pd.DataFrame(
            analyzer.naming_issue_rows
        )
        self.errors = pd.DataFrame(analyzer.errors)

        self.target_size = analyzer.cfg.target_size

    def class_distribution(self):
        if self.bboxes.empty:
            return pd.DataFrame(
                columns=["cls", "objects", "images_with_class"]
            )

        obj_counts = (
            self.bboxes.groupby("cls")
            .size()
            .rename("objects")
        )

        img_counts = (
            self.bboxes.groupby("cls")["image_relpath"]
            .nunique()
            .rename("images_with_class")
        )

        return pd.concat(
            [obj_counts, img_counts], axis=1
        ).reset_index().sort_values(
            "objects",
            ascending=False
        )

    def resolution_distribution(self):
        if self.images.empty:
            return pd.DataFrame(
                columns=["resolution", "count"]
            )

        return (
            self.images[
                self.images["corrupt"] == False
            ]
            .groupby("resolution")
            .size()
            .rename("count")
            .reset_index()
            .sort_values("count", ascending=False)
        )

    def object_count_histogram(self):
        if self.images.empty:
            return pd.DataFrame(
                columns=["n_objects", "count"]
            )

        return (
            self.images.groupby("n_objects")
            .size()
            .rename("count")
            .reset_index()
            .sort_values("n_objects")
        )

    def ball_size_distribution(self):
        if self.bboxes.empty:
            return pd.DataFrame(
                columns=["size_bucket", "count", "pct"]
            )

        vc = self.bboxes["size_bucket"].value_counts()

        df = vc.rename("count").reset_index()
        df.columns = ["size_bucket", "count"]

        df["pct"] = round(
            100 * df["count"] / df["count"].sum(),
            2
        )

        order = [b[0] for b in SIZE_BUCKETS]

        df["size_bucket"] = pd.Categorical(
            df["size_bucket"],
            categories=order,
            ordered=True
        )

        return df.sort_values("size_bucket")

    def resized_size_distribution(self):
        col = (
            f"resized_{self.target_size}"
            f"_size_bucket"
        )

        if (
            self.bboxes.empty
            or col not in self.bboxes.columns
        ):
            return pd.DataFrame(
                columns=["size_bucket", "count", "pct"]
            )

        vc = self.bboxes[col].value_counts()

        df = vc.rename("count").reset_index()
        df.columns = ["size_bucket", "count"]

        df["pct"] = round(
            100 * df["count"] / df["count"].sum(),
            2
        )

        order = [b[0] for b in SIZE_BUCKETS]

        df["size_bucket"] = pd.Categorical(
            df["size_bucket"],
            categories=order,
            ordered=True
        )

        return df.sort_values("size_bucket")

    def resize_impact_summary(self):
        ts = self.target_size

        w_col = f"resized_{ts}_width_px"
        h_col = f"resized_{ts}_height_px"
        tiny_col = f"tiny_after_resize_{ts}"

        required = {
            "cls",
            "width",
            "height",
            w_col,
            h_col,
            tiny_col
        }

        if (
            self.bboxes.empty
            or not required.issubset(self.bboxes.columns)
        ):
            return pd.DataFrame()

        g = self.bboxes.groupby("cls")

        out = pd.DataFrame({
            "objects": g.size(),
            "avg_orig_width_px":
                g["width"].mean().round(1),
            "avg_orig_height_px":
                g["height"].mean().round(1),
            f"avg_resized_{ts}_width_px":
                g[w_col].mean().round(2),
            f"avg_resized_{ts}_height_px":
                g[h_col].mean().round(2),
            f"pct_tiny_after_resize_{ts}":
                (100 * g[tiny_col].mean()).round(2),
        }).reset_index()

        return out.sort_values(
            "objects",
            ascending=False
        )

    def coco_readiness(self):
        corrupt = (
            int(self.images["corrupt"].sum())
            if not self.images.empty
            else 0
        )

        duplicate_files = (
            self.duplicates["file"].nunique()
            if not self.duplicates.empty
            else 0
        )

        return pd.DataFrame([
            {
                "check": "Missing labels",
                "count": len(self.missing_labels),
                "status":
                    "OK" if len(self.missing_labels) == 0
                    else "REVIEW"
            },
            {
                "check": "Missing images",
                "count": len(self.missing_images),
                "status":
                    "OK" if len(self.missing_images) == 0
                    else "REVIEW"
            },
            {
                "check": "Invalid / malformed annotations",
                "count": len(self.invalid),
                "status":
                    "OK" if len(self.invalid) == 0
                    else "REVIEW"
            },
            {
                "check": "Corrupt images",
                "count": corrupt,
                "status":
                    "OK" if corrupt == 0
                    else "REVIEW"
            },
            {
                "check": "Duplicate images",
                "count": duplicate_files,
                "status":
                    "OK" if duplicate_files == 0
                    else "REVIEW"
            },
        ])


def generate_plots(rf, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    saved = []

    def savefig(name):
        path = plots_dir / name
        plt.tight_layout()
        plt.savefig(path, dpi=130)
        plt.close()
        saved.append(path)

    valid_imgs = (
        rf.images[rf.images["corrupt"] == False]
        if not rf.images.empty
        else rf.images
    )

    if not valid_imgs.empty:
        res = rf.resolution_distribution().head(20)

        if len(res):
            plt.figure(figsize=(9, 5))
            plt.barh(
                res["resolution"].astype(str),
                res["count"]
            )
            plt.gca().invert_yaxis()
            plt.xlabel("Image count")
            plt.title("Top 20 Resolutions")
            savefig("resolution.png")

    for col, title, filename in [
        ("brightness", "Brightness Distribution", "brightness.png"),
        ("contrast", "Contrast Distribution", "contrast.png"),
        ("blur_score", "Blur Distribution", "blur.png"),
        ("saturation", "Saturation Distribution", "saturation.png"),
        ("entropy", "Entropy Distribution", "entropy.png"),
    ]:
        if not valid_imgs.empty and col in valid_imgs.columns:
            plt.figure(figsize=(8, 5))
            plt.hist(valid_imgs[col].dropna(), bins=40)
            plt.xlabel(col)
            plt.ylabel("Images")
            plt.title(title)
            savefig(filename)

    if (
        not valid_imgs.empty
        and {"mean_r", "mean_g", "mean_b"}.issubset(
            valid_imgs.columns
        )
    ):
        plt.figure(figsize=(8, 5))

        for channel in ("mean_r", "mean_g", "mean_b"):
            plt.hist(
                valid_imgs[channel].dropna(),
                bins=40,
                alpha=0.5,
                label=channel[-1].upper()
            )

        plt.legend()
        plt.title("Mean RGB Channel Distribution")
        plt.xlabel("Mean channel value")
        savefig("rgb_histogram.png")

    oc = rf.object_count_histogram()

    if len(oc):
        plt.figure(figsize=(8, 5))
        plt.bar(
            oc["n_objects"].astype(str),
            oc["count"]
        )
        plt.xlabel("Objects per image")
        plt.ylabel("Image count")
        plt.title("Object Count Histogram")

        if len(oc) > 25:
            plt.xticks(rotation=90)

        savefig("object_count.png")

    if not rf.bboxes.empty:
        upper = rf.bboxes["rel_area_pct"].quantile(0.99)

        plt.figure(figsize=(8, 5))
        plt.hist(
            rf.bboxes["rel_area_pct"].clip(upper=upper),
            bins=50
        )
        plt.xlabel("BBox area as % of image")
        plt.ylabel("Boxes")
        plt.title("Bounding Box Relative Area")
        savefig("bbox_area.png")

        plt.figure(figsize=(7, 7))

        sample = rf.bboxes.sample(
            min(len(rf.bboxes), 5000),
            random_state=0
        )

        plt.scatter(
            sample["width"],
            sample["height"],
            s=6,
            alpha=0.4
        )

        plt.xlabel("BBox width (px)")
        plt.ylabel("BBox height (px)")
        plt.title("Bounding Box Width vs Height")
        savefig("bbox_scatter.png")

        plt.figure(figsize=(6, 6))

        heat, _, _ = np.histogram2d(
            rf.bboxes["center_x_rel"],
            rf.bboxes["center_y_rel"],
            bins=25,
            range=[[0, 1], [0, 1]]
        )

        plt.imshow(
            heat.T,
            origin="upper",
            extent=[0, 1, 1, 0],
            aspect="auto"
        )

        plt.colorbar(label="Box count")
        plt.xlabel("Relative X")
        plt.ylabel("Relative Y")
        plt.title("Bounding Box Center Heatmap")
        savefig("heatmap.png")

    cdist = rf.class_distribution()

    if len(cdist):
        plt.figure(figsize=(9, 5))
        top = cdist.head(30)

        plt.bar(
            top["cls"].astype(str),
            top["objects"]
        )

        plt.xticks(rotation=60, ha="right")
        plt.ylabel("Object count")
        plt.title("Class Distribution")
        savefig("class_distribution.png")

    if not rf.folders.empty:
        agg = (
            rf.folders
            .groupby("top_folder")
            .agg(
                images=("images", "sum"),
                mean_brightness=(
                    "mean_brightness",
                    "mean"
                ),
                mean_blur_score=(
                    "mean_blur_score",
                    "mean"
                )
            )
            .reset_index()
        )

        if len(agg):
            fig, axes = plt.subplots(
                1, 2, figsize=(12, 5)
            )

            axes[0].bar(
                agg["top_folder"],
                agg["mean_brightness"]
            )
            axes[0].set_title(
                "Mean Brightness by Folder"
            )
            axes[0].tick_params(
                axis="x",
                rotation=75
            )

            axes[1].bar(
                agg["top_folder"],
                agg["mean_blur_score"]
            )
            axes[1].set_title(
                "Mean Blur Score by Folder"
            )
            axes[1].tick_params(
                axis="x",
                rotation=75
            )

            savefig("folder_comparison.png")

    bsd = rf.ball_size_distribution()

    if len(bsd):
        plt.figure(figsize=(6, 6))
        plt.pie(
            bsd["count"],
            labels=[
                f"{r.size_bucket}\n{r.pct}%"
                for r in bsd.itertuples()
            ]
        )
        plt.title("Object Size Buckets")
        savefig("size_buckets.png")

    ts = rf.target_size
    w_col = f"resized_{ts}_width_px"
    h_col = f"resized_{ts}_height_px"

    if (
        not rf.bboxes.empty
        and {w_col, h_col}.issubset(rf.bboxes.columns)
    ):
        fig, axes = plt.subplots(
            1, 2, figsize=(12, 5)
        )

        max_side = max(
            rf.bboxes["width"].quantile(0.99),
            rf.bboxes["height"].quantile(0.99),
            1
        )

        bins = np.linspace(0, max_side, 40)

        axes[0].hist(
            rf.bboxes["width"].clip(upper=max_side),
            bins=bins,
            alpha=0.6,
            label="width"
        )
        axes[0].hist(
            rf.bboxes["height"].clip(upper=max_side),
            bins=bins,
            alpha=0.6,
            label="height"
        )
        axes[0].set_title(
            "Original Object Size (px)"
        )
        axes[0].set_xlabel("pixels")
        axes[0].legend()

        max_side_r = max(
            rf.bboxes[w_col].quantile(0.99),
            rf.bboxes[h_col].quantile(0.99),
            1
        )

        bins_r = np.linspace(0, max_side_r, 40)

        axes[1].hist(
            rf.bboxes[w_col].clip(upper=max_side_r),
            bins=bins_r,
            alpha=0.6,
            label="width"
        )
        axes[1].hist(
            rf.bboxes[h_col].clip(upper=max_side_r),
            bins=bins_r,
            alpha=0.6,
            label="height"
        )
        axes[1].set_title(
            f"Object Size After {ts}x{ts} Letterbox Resize"
        )
        axes[1].set_xlabel("pixels")
        axes[1].legend()

        savefig(f"resize_impact_{ts}px.png")

        rsd = rf.resized_size_distribution()

        if len(rsd):
            plt.figure(figsize=(6, 6))
            plt.pie(
                rsd["count"],
                labels=[
                    f"{r.size_bucket}\n{r.pct}%"
                    for r in rsd.itertuples()
                ]
            )
            plt.title(
                f"Object Size Buckets After "
                f"{ts}x{ts} Resize"
            )
            savefig(
                f"size_buckets_after_resize_{ts}px.png"
            )

    print(f"Saved {len(saved)} plots to {plots_dir}")
    return saved


def _autosize_excel(xlsx_path, max_width=40):
    try:
        from openpyxl import load_workbook

        wb = load_workbook(xlsx_path)

        for ws in wb.worksheets:
            for col_cells in ws.columns:
                length = max(
                    (
                        len(str(c.value))
                        if c.value is not None
                        else 0
                    )
                    for c in col_cells
                )

                col_letter = col_cells[0].column_letter
                ws.column_dimensions[col_letter].width = min(
                    max(length + 2, 10),
                    max_width
                )

            ws.freeze_panes = "A2"

        wb.save(xlsx_path)

    except Exception as e:
        print(f"Excel autosize warning: {e}")


def export_excel_and_csv(rf, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)

    xlsx_path = out_dir / "dataset_summary.xlsx"

    sheets = {
        "Folder Stats": rf.folders,
        "Class Distribution": rf.class_distribution(),
        "Resolution Distribution": rf.resolution_distribution(),
        "Object Count Histogram": rf.object_count_histogram(),
        "Ball Size Distribution": rf.ball_size_distribution(),
        f"Resize Impact {rf.target_size}px":
            rf.resize_impact_summary(),
        "Resized Size Distribution":
            rf.resized_size_distribution(),
        "COCO Readiness": rf.coco_readiness(),
        "Missing Labels": rf.missing_labels,
        "Missing Images": rf.missing_images,
        "Invalid Annotations": rf.invalid,
        "Duplicate Images": rf.duplicates,
        "Naming Issues": rf.naming,
        "Read Errors": rf.errors,
        "Image Statistics": rf.images,
        "BBox Statistics": rf.bboxes,
    }

    with pd.ExcelWriter(
        xlsx_path,
        engine="openpyxl"
    ) as writer:
        for name, df in sheets.items():
            if df.empty:
                df = pd.DataFrame({"info": ["no rows"]})

            df.to_excel(
                writer,
                sheet_name=name[:31],
                index=False
            )

    _autosize_excel(xlsx_path)

    rf.images.to_csv(
        out_dir / "image_statistics.csv",
        index=False
    )
    rf.bboxes.to_csv(
        out_dir / "bbox_statistics.csv",
        index=False
    )
    rf.folders.to_csv(
        out_dir / "folder_statistics.csv",
        index=False
    )
    rf.duplicates.to_csv(
        out_dir / "duplicate_images.csv",
        index=False
    )
    rf.invalid.to_csv(
        out_dir / "invalid_annotations.csv",
        index=False
    )
    rf.resize_impact_summary().to_csv(
        out_dir / f"resize_impact_{rf.target_size}px.csv",
        index=False
    )

    if not rf.images.empty:
        blurry = rf.images[
            rf.images["blur_bucket"].isin(
                ["Very Blurry", "Blurry"]
            )
        ]

        blurry.to_csv(
            out_dir / "blurry_images.csv",
            index=False
        )

    print(f"Excel workbook: {xlsx_path}")
    print(f"CSV files written to: {out_dir}")

    return xlsx_path


def _df_to_html(df, max_rows=300):
    if df is None or df.empty:
        return "<p><em>None found.</em></p>"

    shown = df.head(max_rows)

    note = ""

    if len(df) > max_rows:
        note = (
            "<p style='color:#888;font-size:12px'>"
            f"Showing {len(shown)} of {len(df)} rows. "
            "Full detail is available in CSV/Excel."
            "</p>"
        )

    return note + shown.to_html(
        index=False,
        border=0,
        escape=True
    )


def generate_html_report(
    rf,
    out_dir,
    root,
    plot_paths
):
    cards = [
        ("Images", len(rf.images)),
        ("Objects", len(rf.bboxes)),
        (
            "Classes",
            rf.bboxes["cls"].nunique()
            if not rf.bboxes.empty else 0
        ),
        (
            "Corrupt Images",
            int(rf.images["corrupt"].sum())
            if not rf.images.empty else 0
        ),
        ("Missing Labels", len(rf.missing_labels)),
        ("Missing Images", len(rf.missing_images)),
        (
            "Duplicate Entries",
            rf.duplicates["file"].nunique()
            if not rf.duplicates.empty else 0
        ),
        ("Invalid Annotations", len(rf.invalid)),
    ]

    summary_cards = "".join(
        f"""
        <div class="card">
            <div class="num">{value}</div>
            <div class="lbl">{label}</div>
        </div>
        """
        for label, value in cards
    )

    coco_html = rf.coco_readiness().to_html(
        index=False,
        border=0
    )

    plot_imgs = "".join(
        f"""
        <div>
            <img src="plots/{p.name}" alt="{p.stem}">
            <div class="caption">
                {p.stem.replace("_", " ").title()}
            </div>
        </div>
        """
        for p in plot_paths
    )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Dataset Audit Report</title>
<style>
body {{
    font-family: -apple-system, BlinkMacSystemFont,
    "Segoe UI", Arial, sans-serif;
    margin: 0;
    background: #f5f6f8;
    color: #1f2430;
}}
header {{
    background: #1f2937;
    color: white;
    padding: 28px 40px;
}}
header h1 {{ margin: 0 0 6px; }}
header p {{ margin: 0; color: #b7c0cc; }}
.wrap {{
    max-width: 1180px;
    margin: auto;
    padding: 24px 40px 60px;
}}
.cards {{
    display: flex;
    gap: 16px;
    flex-wrap: wrap;
    margin: 20px 0 34px;
}}
.card {{
    background: white;
    border-radius: 10px;
    padding: 16px 20px;
    box-shadow: 0 1px 3px rgba(0,0,0,.08);
    min-width: 150px;
    flex: 1;
}}
.num {{ font-size: 26px; font-weight: 700; }}
.lbl {{ font-size: 12px; color: #666; }}
h2 {{
    border-bottom: 2px solid #e5e7eb;
    padding-bottom: 8px;
    margin-top: 46px;
}}
table {{
    border-collapse: collapse;
    width: 100%;
    background: white;
    font-size: 13px;
}}
th, td {{
    border: 1px solid #e5e7eb;
    padding: 6px 10px;
    text-align: left;
}}
th {{ background: #f0f2f5; }}
.scroll {{
    max-height: 480px;
    overflow: auto;
    border: 1px solid #e5e7eb;
    border-radius: 8px;
}}
.plot-grid {{
    display: grid;
    grid-template-columns:
        repeat(auto-fit, minmax(430px, 1fr));
    gap: 18px;
}}
.plot-grid img {{
    width: 100%;
    border-radius: 8px;
    border: 1px solid #e5e7eb;
    background: white;
}}
.caption {{
    text-align: center;
    font-size: 12px;
    color: #666;
    margin-top: 4px;
}}
</style>
</head>
<body>

<header>
<h1>Dataset Audit Report</h1>
<p>
Root: {root}
&nbsp;|&nbsp;
Generated: {datetime.now().strftime("%Y-%m-%d %H:%M")}
</p>
</header>

<div class="wrap">

<div class="cards">
{summary_cards}
</div>

<h2>Object-Detection Readiness</h2>
{coco_html}

<h2>Folder Statistics</h2>
<div class="scroll">
{_df_to_html(rf.folders)}
</div>

<h2>Class Distribution</h2>
<div class="scroll">
{_df_to_html(rf.class_distribution())}
</div>

<h2>
Object Pixel Size:
Original vs. After {rf.target_size}x{rf.target_size} Resize
</h2>

<p>
Resize analysis uses uniform letterbox scaling with centered
padding. No crop and no aspect-ratio distortion are applied.
</p>

<div class="scroll">
{_df_to_html(rf.resize_impact_summary())}
</div>

<h2>Charts</h2>
<div class="plot-grid">
{plot_imgs}
</div>

<h2>Missing Labels</h2>
<div class="scroll">
{_df_to_html(rf.missing_labels)}
</div>

<h2>Missing Images</h2>
<div class="scroll">
{_df_to_html(rf.missing_images)}
</div>

<h2>Invalid / Flagged Annotations</h2>
<div class="scroll">
{_df_to_html(rf.invalid)}
</div>

<h2>Duplicate Images</h2>
<div class="scroll">
{_df_to_html(rf.duplicates)}
</div>

<h2>File Naming Issues</h2>
<div class="scroll">
{_df_to_html(rf.naming)}
</div>

</div>
</body>
</html>
"""

    report_path = out_dir / "report.html"

    report_path.write_text(
        html,
        encoding="utf-8"
    )

    print(f"HTML report: {report_path}")

    return report_path


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Deep recursive dataset analyzer "
            "for object-detection datasets."
        )
    )

    parser.add_argument(
        "--root",
        required=True,
        help="Dataset root directory."
    )

    parser.add_argument(
        "--out",
        default="./analysis",
        help="Output directory."
    )

    parser.add_argument(
        "--hash-threshold",
        type=int,
        default=5,
        help="Perceptual hash Hamming distance."
    )

    parser.add_argument(
        "--target-size",
        type=int,
        default=320,
        help="Target square resize size."
    )

    parser.add_argument(
        "--tiny-px-threshold",
        type=float,
        default=4.0,
        help="Tiny-object threshold after resize."
    )

    parser.add_argument(
        "--sample",
        type=int,
        default=0,
        help="Images per top-level folder; 0 = all."
    )

    parser.add_argument(
        "--no-plots",
        action="store_true"
    )

    parser.add_argument(
        "--no-html",
        action="store_true"
    )

    parser.add_argument(
        "--no-checkpoint",
        action="store_true"
    )

    parser.add_argument(
        "--resume",
        action="store_true"
    )

    args = parser.parse_args()

    root = Path(args.root).expanduser().resolve()
    out_dir = Path(args.out).expanduser().resolve()

    if not root.exists():
        print(
            f"ERROR: dataset path does not exist:\n{root}"
        )
        sys.exit(1)

    cfg = AnalyzerConfig(
        root=root,
        out_dir=out_dir,
        hash_threshold=args.hash_threshold,
        sample=args.sample,
        make_plots=not args.no_plots,
        make_html=not args.no_html,
        checkpoint=not args.no_checkpoint,
        resume=args.resume,
        target_size=args.target_size,
        tiny_px_threshold=args.tiny_px_threshold,
    )

    start = datetime.now()
    analyzer = DatasetAnalyzer(cfg)
    crashed = False

    try:
        analyzer.run()

    except KeyboardInterrupt:
        print("\nInterrupted. Saving checkpoint...")
        analyzer._save_checkpoint()
        crashed = True

    except Exception as e:
        print(f"\nERROR: {e}")
        traceback.print_exc()
        analyzer._save_checkpoint()
        crashed = True

    rf = ResultFrames(analyzer)

    out_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    export_excel_and_csv(rf, out_dir)

    plot_paths = []

    if cfg.make_plots:
        try:
            plot_paths = generate_plots(
                rf, out_dir
            )
        except Exception as e:
            print(f"Plot generation failed: {e}")
            traceback.print_exc()

    if cfg.make_html:
        try:
            generate_html_report(
                rf,
                out_dir,
                root,
                plot_paths
            )
        except Exception as e:
            print(f"HTML generation failed: {e}")
            traceback.print_exc()

    elapsed = (
        datetime.now() - start
    ).total_seconds()

    print("\n" + "=" * 80)

    if crashed:
        print("DATASET ANALYSIS FINISHED WITH PARTIAL RESULTS")
    else:
        print("DATASET ANALYSIS COMPLETE")

    print("=" * 80)

    print(f"Images analyzed : {len(rf.images):,}")
    print(f"Objects found   : {len(rf.bboxes):,}")

    if not rf.bboxes.empty:
        print(
            f"Classes found   : "
            f"{rf.bboxes['cls'].nunique():,}"
        )
    else:
        print("Classes found   : 0")

    print(f"Elapsed time    : {elapsed:.1f} seconds")
    print(f"\nResults directory:\n{out_dir}")

    print("\nClient annotation format:")
    print("  dimensions = [height, width]")
    print("  rect       = [x1, y1, x2, y2]")

    if crashed:
        print("\nRun again with --resume to continue.")
        sys.exit(1)


if __name__ == "__main__":
    main()

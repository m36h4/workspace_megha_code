#!/usr/bin/env python3
"""
================================================================================
 DATASET ANALYZER - Deep Recursive Auditing Tool for Object Detection Datasets
================================================================================

Recursively scans a dataset directory (folder / subfolder / class-wise),
computes image + annotation statistics, flags quality issues, and produces
Excel + CSV + PNG + HTML deliverables suitable for handing to a client.

USAGE
-----
    python dataset_analyzer.py --root /path/to/Dataset --out ./analysis

    Optional flags:
      --bbox-format {auto,xywh,xyxy}   default: auto
      --hash-threshold INT             hamming distance for near-duplicates (default 5)
      --sample N                       only analyze N images per folder (fast preview)
      --workers N                      parallel worker processes (default: 1, safe)
      --no-plots                       skip PNG chart generation
      --no-html                       skip HTML report generation

SUPPORTED ANNOTATION FORMATS (auto-detected per top-level folder)
-------------------------------------------------------------------
  1. COCO          - single *.json in the folder with "images"/"annotations"/"categories" keys
  2. YOLO          - per-image *.txt with "class cx cy w h" normalized lines
  3. Generic JSON  - per-image *.json (LabelMe-like / custom) with a list of objects
                      under a key such as "objects"/"annotations"/"shapes"/"boxes",
                      each holding a bbox (xywh, xyxy, or xmin/ymin/xmax/ymax fields)
                      and a class name field ("class"/"label"/"category"/"name").

Only Python-standard-library-adjacent, commonly available packages are required.
imagehash / tqdm are used if present but are NOT required - the script degrades
gracefully (see OptionalDeps below) so it runs on a bare client machine.

Author: generated for deep client dataset auditing.
================================================================================
"""

import os
import sys
import io
import gc
import json
import math
import pickle
import argparse
import hashlib
import traceback
import unicodedata
from collections import defaultdict, Counter
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# ---- image backends -----------------------------------------------------
import cv2
from PIL import Image, ImageOps
Image.MAX_IMAGE_PIXELS = None  # client datasets can have huge images

# ---- optional deps (script must still run if these are missing) --------
class OptionalDeps:
    tqdm_available = False
    skimage_available = False

try:
    from tqdm import tqdm as _tqdm
    OptionalDeps.tqdm_available = True
except ImportError:
    def _tqdm(iterable, **kwargs):
        total = kwargs.get("total")
        desc = kwargs.get("desc", "")
        n = 0
        step = max(1, (total or 0) // 20) if total else 500
        for item in iterable:
            n += 1
            if n % step == 0 or n == total:
                pct = f"{100*n/total:5.1f}%" if total else f"{n}"
                print(f"\r  [{desc}] {pct}", end="", flush=True)
            yield item
        print()

try:
    from skimage.measure import shannon_entropy as _sk_entropy
    OptionalDeps.skimage_available = True
except ImportError:
    _sk_entropy = None

tqdm = _tqdm

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
LABEL_EXTS = {".json", ".txt", ".xml"}
UNSUPPORTED_EXT_WARN = {".jfif", ".heic", ".avif", ".gif", ".psd"}

# Bounding-box relative-area size buckets (bbox_area / image_area), as %
SIZE_BUCKETS = [
    ("Tiny",   0.0,  0.002),
    ("Small",  0.002, 0.01),
    ("Medium", 0.01,  0.05),
    ("Large",  0.05,  1.01),
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

EDGE_TOUCH_MARGIN_PX = 2  # how close to border counts as "touching"


def bucket_for(value, buckets):
    for name, lo, hi in buckets:
        if lo <= value < hi:
            return name
    return buckets[-1][0]


def is_probably_unicode_name(name: str) -> bool:
    return any(ord(ch) > 127 for ch in name)


def safe_relpath(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


# =============================================================================
# IMAGE METRICS
# =============================================================================

def custom_dhash(gray_small: np.ndarray) -> int:
    """Difference hash (8x8) -> 64-bit int. gray_small must be 9x8."""
    diff = gray_small[:, 1:] > gray_small[:, :-1]
    h = 0
    for v in diff.flatten():
        h = (h << 1) | int(v)
    return h


def custom_phash(gray32: np.ndarray) -> int:
    """Perceptual hash via DCT (32x32 -> top-left 8x8 low freq) -> 64-bit int."""
    dct = cv2.dct(np.float32(gray32))
    dct_low = dct[:8, :8]
    med = np.median(dct_low[1:, 1:])  # skip DC term
    bits = (dct_low > med)
    h = 0
    for v in bits.flatten():
        h = (h << 1) | int(v)
    return h


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def compute_image_hashes(gray: np.ndarray):
    small_d = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA)
    small_p = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA)
    return custom_dhash(small_d), custom_phash(small_p)


def compute_entropy(gray: np.ndarray) -> float:
    if OptionalDeps.skimage_available:
        try:
            return float(_sk_entropy(gray))
        except Exception:
            pass
    hist, _ = np.histogram(gray, bins=256, range=(0, 256))
    prob = hist / max(hist.sum(), 1)
    prob = prob[prob > 0]
    return float(-np.sum(prob * np.log2(prob)))


def analyze_image_pixels(img_bgr: np.ndarray) -> dict:
    """Compute the full pixel/quality metric set for a single loaded BGR image."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    b, g, r = cv2.split(img_bgr.astype(np.float64))

    brightness = float(gray.mean())
    contrast = float(gray.std())
    laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    saturation = float(hsv[:, :, 1].mean())
    entropy = compute_entropy(gray)
    dhash_val, phash_val = compute_image_hashes(gray)

    # exposure classification from brightness distribution
    dark_frac = float((gray < 30).mean())
    bright_frac = float((gray > 225).mean())
    if bright_frac > 0.35:
        exposure = "Overexposed"
    elif dark_frac > 0.35:
        exposure = "Underexposed"
    else:
        exposure = "Normal"

    return {
        "mean_r": float(r.mean()), "mean_g": float(g.mean()), "mean_b": float(b.mean()),
        "std_r": float(r.std()), "std_g": float(g.std()), "std_b": float(b.std()),
        "brightness": brightness,
        "brightness_bucket": bucket_for(brightness, BRIGHTNESS_BUCKETS),
        "contrast": contrast,
        "blur_score": laplacian_var,
        "blur_bucket": bucket_for(laplacian_var, BLUR_BUCKETS),
        "saturation": saturation,
        "entropy": entropy,
        "exposure": exposure,
        "dhash": dhash_val,
        "phash": phash_val,
    }


# =============================================================================
# ANNOTATION PARSING  (COCO / YOLO / Generic-JSON, auto-detected)
# =============================================================================

BBOX_KEY_SETS = [
    ("xmin", "ymin", "xmax", "ymax", "xyxy"),
    ("x1", "y1", "x2", "y2", "xyxy"),
    ("x_min", "y_min", "x_max", "y_max", "xyxy"),
    ("x", "y", "width", "height", "xywh"),
    ("x", "y", "w", "h", "xywh"),
    ("left", "top", "right", "bottom", "xyxy"),
]
CLASS_KEYS = ["class", "class_name", "label", "category", "name", "category_name", "cls"]
OBJECT_LIST_KEYS = ["objects", "annotations", "shapes", "boxes", "labels", "bboxes", "detections"]


def _find_rects_in_obj(obj, region_type=""):
    """
    Recursively search a nested dict/list for any {"rect": [x1,y1,x2,y2]}-style entry,
    e.g. {"entire": {"rect": [...]}} or {"occluded": {"rect": [...]}}. Returns a list of
    (raw_4_values, region_type) tuples. region_type is the parent key name ("entire",
    "truncated", etc.) purely for diagnostics.
    """
    found = []
    if isinstance(obj, dict):
        rect = obj.get("rect")
        if isinstance(rect, (list, tuple)) and len(rect) == 4:
            found.append((list(rect), region_type or "region"))
        for k, v in obj.items():
            if k == "rect":
                continue
            if isinstance(v, (dict, list)):
                found.extend(_find_rects_in_obj(v, k))
    elif isinstance(obj, list):
        for item in obj:
            found.extend(_find_rects_in_obj(item, region_type))
    return found


def _looks_like_nested_data_format(json_path: Path) -> bool:
    """Peek a json file to see if it matches {"data": {"<class>": [{"entire": {"rect":[...]}}]}}."""
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            d = json.load(f)
    except Exception:
        return False
    if not (isinstance(d, dict) and isinstance(d.get("data"), dict)):
        return False
    for v in d["data"].values():
        if isinstance(v, list):
            for obj in v:
                if _find_rects_in_obj(obj):
                    return True
    return False


def parse_nested_data_json_label(json_path: Path, img_w, img_h):
    """
    Parser for the client's annotation schema:
        {
          "file_name": "...",
          "dimensions": [W, H],
          "data": {
            "ball": [ { "entire": { "rect": [x1, y1, x2, y2] } }, ... ],
            "<other_class>": [ ... ]
          }
        }
    The class name is the key under "data" (e.g. "ball"); each object may nest its box
    under a region key ("entire", "truncated", "occluded", ...) or have "rect" directly.
    rect is [x, y, w, h] (top-left corner + width/height), in absolute pixel coords -
    the convention confirmed for this client's schema.
    Returns (boxes, error_or_None, warning_or_None).
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
    if isinstance(dims, (list, tuple)) and len(dims) == 2 and img_w and img_h:
        dw, dh = dims
        if abs(dw - img_w) > 2 or abs(dh - img_h) > 2:
            warning = f"declared_dimensions_mismatch(json={dw}x{dh},actual_image={img_w}x{img_h})"

    boxes = []
    for cls_name, obj_list in data.items():
        if not isinstance(obj_list, list):
            continue
        for obj in obj_list:
            for raw_rect, _region_type in _find_rects_in_obj(obj):
                rx, ry, rw, rh = [float(v) for v in raw_rect]
                # This schema's rect is [x, y, w, h] (top-left + width/height), not
                # [x1, y1, x2, y2]. Convert to xyxy for downstream metrics.
                x1, y1, x2, y2 = rx, ry, rx + rw, ry + rh
                boxes.append({"x1": x1, "y1": y1, "x2": x2, "y2": y2, "cls": str(cls_name)})

    if not boxes:
        return [], "empty_labels", warning
    return boxes, None, warning


def _extract_bbox_from_dict(d: dict):
    """Try every known key convention; return (x1,y1,x2,y2, fmt_used) in pixel/abs coords, or None."""
    if "bbox" in d and isinstance(d["bbox"], (list, tuple)) and len(d["bbox"]) == 4:
        vals = [float(v) for v in d["bbox"]]
        return vals, "bbox_field"
    for kx1, ky1, kx2, ky2, _fmt in BBOX_KEY_SETS:
        if all(k in d for k in (kx1, ky1, kx2, ky2)):
            return [float(d[kx1]), float(d[ky1]), float(d[kx2]), float(d[ky2])], "keys:" + "_".join([kx1, ky1, kx2, ky2])
    if "points" in d and isinstance(d["points"], (list, tuple)) and len(d["points"]) >= 2:
        # LabelMe-style polygon/rectangle -> bounding box of points
        pts = np.array(d["points"], dtype=float)
        x1, y1 = pts[:, 0].min(), pts[:, 1].min()
        x2, y2 = pts[:, 0].max(), pts[:, 1].max()
        return [x1, y1, x2, y2], "polygon_points"
    return None, None


def _resolve_xyxy(vals, img_w, img_h):
    """
    Given 4 raw numbers of unknown convention (xywh vs xyxy, normalized vs absolute),
    return best-guess (x1, y1, x2, y2) in absolute pixel coordinates.
    """
    v = list(vals)
    normalized = all(0.0 <= x <= 1.5 for x in v) and img_w and img_h
    if normalized:
        # Could be normalized xywh (YOLO-ish) or normalized xyxy
        as_xywh = [v[0] * img_w, v[1] * img_h, v[2] * img_w, v[3] * img_h]
        cand_xyxy_from_xywh = [as_xywh[0] - as_xywh[2] / 2, as_xywh[1] - as_xywh[3] / 2,
                                as_xywh[0] + as_xywh[2] / 2, as_xywh[1] + as_xywh[3] / 2]
        return cand_xyxy_from_xywh
    # absolute pixel coords: decide xywh vs xyxy by whether v[2]>v[0] and v[3]>v[1] plausibly as xyxy
    x1, y1, a, b = v
    # Heuristic: if treating (a,b) as x2,y2 gives a valid positive box smaller than image -> xyxy
    if a > x1 and b > y1 and a <= img_w * 1.05 and b <= img_h * 1.05:
        return [x1, y1, a, b]
    # else treat as xywh
    return [x1, y1, x1 + a, y1 + b]


def parse_generic_json_label(json_path: Path, img_w, img_h):
    """Returns (list_of_boxes, error_str_or_None). Each box: dict(x1,y1,x2,y2,cls)."""
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        return [], f"invalid_json: {e}"

    objs = None
    if isinstance(data, list):
        objs = data
    elif isinstance(data, dict):
        for k in OBJECT_LIST_KEYS:
            if k in data and isinstance(data[k], list):
                objs = data[k]
                break
        if objs is None:
            bbox_maybe, _ = _extract_bbox_from_dict(data)
            if bbox_maybe is not None:
                objs = [data]
    if objs is None:
        return [], "no_recognizable_object_list"
    if len(objs) == 0:
        return [], "empty_labels"

    boxes = []
    for obj in objs:
        if not isinstance(obj, dict):
            continue
        raw, _fmt = _extract_bbox_from_dict(obj)
        if raw is None:
            continue
        x1, y1, x2, y2 = _resolve_xyxy(raw, img_w, img_h)
        cls = "object"
        for ck in CLASS_KEYS:
            if ck in obj:
                cls = str(obj[ck])
                break
        boxes.append({"x1": x1, "y1": y1, "x2": x2, "y2": y2, "cls": cls})
    return boxes, None


def parse_yolo_label(txt_path: Path, img_w, img_h, class_names=None):
    try:
        lines = [l.strip() for l in open(txt_path, "r", encoding="utf-8").read().splitlines() if l.strip()]
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
            cx, cy, w, h = [float(x) for x in parts[1:5]]
        except ValueError:
            continue
        x1 = (cx - w / 2) * img_w
        y1 = (cy - h / 2) * img_h
        x2 = (cx + w / 2) * img_w
        y2 = (cy + h / 2) * img_h
        cls = class_names[cls_idx] if class_names and 0 <= cls_idx < len(class_names) else str(cls_idx)
        boxes.append({"x1": x1, "y1": y1, "x2": x2, "y2": y2, "cls": cls})
    return boxes, None


def load_coco_json(coco_path: Path):
    """Returns dict: filename -> list_of_boxes, plus set of all filenames declared."""
    with open(coco_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    id2name = {c["id"]: c.get("name", str(c["id"])) for c in data.get("categories", [])}
    imgid2file = {im["id"]: im["file_name"] for im in data.get("images", [])}
    per_image = defaultdict(list)
    for ann in data.get("annotations", []):
        img_id = ann.get("image_id")
        fname = imgid2file.get(img_id)
        if fname is None:
            continue
        bbox = ann.get("bbox")
        if not bbox or len(bbox) != 4:
            continue
        x, y, w, h = bbox
        cls = id2name.get(ann.get("category_id"), str(ann.get("category_id")))
        per_image[fname].append({"x1": x, "y1": y, "x2": x + w, "y2": y + h, "cls": cls})
    return per_image, set(imgid2file.values())


# =============================================================================
# BBOX METRIC COMPUTATION + VALIDATION
# =============================================================================

def compute_letterbox_transform(img_w, img_h, target_size):
    """
    Compute the scale + padding for a 'letterbox' resize to a target_size x target_size
    canvas: uniform scale so the longer side fits target_size, then pad the shorter side
    to reach a square - no cropping, no aspect-ratio distortion (the standard YOLO-style
    resize). Returns (scale, pad_left, pad_top, content_w, content_h).
    """
    if not img_w or not img_h:
        return 1.0, 0.0, 0.0, target_size, target_size
    scale = min(target_size / img_w, target_size / img_h)
    content_w = img_w * scale
    content_h = img_h * scale
    pad_left = (target_size - content_w) / 2
    pad_top = (target_size - content_h) / 2
    return scale, pad_left, pad_top, content_w, content_h


def compute_bbox_metrics(box, img_w, img_h, target_size=320, tiny_px_threshold=4.0):
    x1, y1, x2, y2 = box["x1"], box["y1"], box["x2"], box["y2"]
    w = x2 - x1
    h = y2 - y1
    issues = []
    if w <= 0 or h <= 0:
        issues.append("zero_or_negative_size")
    if x1 < 0 or y1 < 0:
        issues.append("negative_coordinate")
    if x2 > img_w + 1 or y2 > img_h + 1:
        issues.append("exceeds_image_bounds")
    if x1 > img_w or y1 > img_h or x2 < 0 or y2 < 0:
        issues.append("bbox_outside_image")

    area = max(w, 0) * max(h, 0)
    img_area = max(img_w * img_h, 1)
    rel_area = area / img_area
    aspect = (w / h) if h > 0 else 0
    cx, cy = x1 + w / 2, y1 + h / 2
    rel_cx, rel_cy = cx / max(img_w, 1), cy / max(img_h, 1)

    touching_edge = (
        x1 <= EDGE_TOUCH_MARGIN_PX or y1 <= EDGE_TOUCH_MARGIN_PX or
        x2 >= img_w - EDGE_TOUCH_MARGIN_PX or y2 >= img_h - EDGE_TOUCH_MARGIN_PX
    )
    truncated = touching_edge and rel_area > 0  # box likely cut off by frame edge

    size_bucket = bucket_for(rel_area, SIZE_BUCKETS) if rel_area >= 0 else "Tiny"

    # -------- pixel size after a letterbox resize to target_size x target_size --------
    # (uniform scale to fit, padded to square - no crop, no distortion)
    scale, pad_left, pad_top, _cw, _ch = compute_letterbox_transform(img_w, img_h, target_size)
    rw, rh = w * scale, h * scale
    r_area = rw * rh
    r_x1, r_y1 = x1 * scale + pad_left, y1 * scale + pad_top
    r_x2, r_y2 = x2 * scale + pad_left, y2 * scale + pad_top
    resized_rel_area_pct = 100 * r_area / (target_size * target_size)
    resized_size_bucket = bucket_for(resized_rel_area_pct / 100, SIZE_BUCKETS) if resized_rel_area_pct >= 0 else "Tiny"
    tiny_after_resize = (rw < tiny_px_threshold) or (rh < tiny_px_threshold)

    return {
        "cls": box["cls"], "x1": x1, "y1": y1, "x2": x2, "y2": y2,
        "width": w, "height": h, "area": area, "rel_area_pct": rel_area * 100,
        "aspect_ratio": aspect, "center_x_rel": rel_cx, "center_y_rel": rel_cy,
        "touching_edge": touching_edge, "truncated": truncated,
        "size_bucket": size_bucket, "issues": ";".join(issues) if issues else "",
        "valid": len(issues) == 0,
        # -- post-resize (letterboxed to target_size x target_size) --
        f"resized_{target_size}_width_px": round(rw, 2), f"resized_{target_size}_height_px": round(rh, 2),
        f"resized_{target_size}_area_px": round(r_area, 2),
        f"resized_{target_size}_rel_area_pct": round(resized_rel_area_pct, 4),
        f"resized_{target_size}_x1": round(r_x1, 2), f"resized_{target_size}_y1": round(r_y1, 2),
        f"resized_{target_size}_x2": round(r_x2, 2), f"resized_{target_size}_y2": round(r_y2, 2),
        f"resized_{target_size}_size_bucket": resized_size_bucket,
        f"tiny_after_resize_{target_size}": tiny_after_resize,
    }




# =============================================================================
# FOLDER / FORMAT DETECTION
# =============================================================================

def detect_top_folder_format(folder: Path):
    """
    Inspect a top-level dataset folder and decide which annotation format it uses.
    Returns one of: ('coco', coco_json_path), ('yolo', None), ('json_nested', None),
    ('json', None), ('none', None)

    Checked in this order: COCO (single json w/ images+annotations+categories) ->
    nested class-keyed json (e.g. {"data": {"ball": [{"entire": {"rect": [...]}}]}}) ->
    YOLO txt -> generic per-image json (flat objects/annotations list).
    """
    json_files = list(folder.rglob("*.json"))
    img_files = [p for p in folder.rglob("*") if p.suffix.lower() in IMAGE_EXTS]
    img_stems = {p.stem for p in img_files}

    # COCO: scan (a bounded sample of) json files for the tell-tale keys. A dataset with
    # one real coco.json is fast; a dataset with per-image json files is capped so this
    # stays quick even with tens of thousands of files.
    for jf in json_files[:500]:
        try:
            with open(jf, "r", encoding="utf-8") as f:
                head = f.read(4096)
            if '"images"' in head and '"annotations"' in head and '"categories"' in head:
                return "coco", jf
        except Exception:
            continue

    # Nested class-keyed json (client's "data": {"<class>": [{"entire": {"rect": [...]}}]})
    sample_jsons = [jf for jf in json_files if jf.stem in img_stems][:8]
    for jf in sample_jsons:
        if _looks_like_nested_data_format(jf):
            return "json_nested", None

    txt_files = list(folder.rglob("*.txt"))
    if txt_files and img_files:
        txt_stems = {p.stem for p in txt_files if p.name.lower() not in ("classes.txt", "data.yaml")}
        if len(img_stems & txt_stems) > 0:
            return "yolo", None
    if json_files and img_files:
        json_stems = {p.stem for p in json_files}
        if len(img_stems & json_stems) > 0:
            return "json", None
    return "none", None


def load_yolo_classnames(folder: Path):
    for cand in ("classes.txt", "obj.names", "names.txt"):
        p = folder / cand
        if p.exists():
            try:
                return [l.strip() for l in open(p, encoding="utf-8").read().splitlines() if l.strip()]
            except Exception:
                pass
    for p in folder.rglob(cand):
        pass
    return None


# =============================================================================
# MAIN ANALYZER
# =============================================================================

@dataclass
class AnalyzerConfig:
    root: Path
    out_dir: Path
    bbox_format: str = "auto"
    hash_threshold: int = 5
    sample: int = 0
    make_plots: bool = True
    make_html: bool = True
    checkpoint: bool = True
    resume: bool = False
    target_size: int = 320
    tiny_px_threshold: float = 4.0


class DatasetAnalyzer:
    def __init__(self, cfg: AnalyzerConfig):
        self.cfg = cfg
        self.root = cfg.root
        self.image_rows = []          # per-image records
        self.bbox_rows = []           # per-bbox records
        self.folder_rows = []         # per-folder aggregate records
        self.missing_labels = []      # images without a label
        self.missing_images = []      # labels without an image
        self.invalid_json_rows = []   # unreadable / malformed annotation files
        self.duplicate_rows = []      # exact + near duplicate pairs
        self.naming_issue_rows = []   # unicode / duplicate-name / unsupported-ext
        self.errors = []              # (path, error) for anything that crashed mid-read

        self._hash_index = {}   # hash_int -> list of (relpath, folder)
        self._name_index = defaultdict(list)  # lower filename -> list of relpaths (cross-folder dup names)
        self._done_relpaths = set()  # populated from a loaded checkpoint, for --resume

        self._checkpoint_path = cfg.out_dir / "_checkpoint.pkl"

    # ------------------------------------------------------------------ scan
    def top_level_folders(self):
        return sorted([p for p in self.root.iterdir() if p.is_dir()])

    def run(self):
        if self.cfg.resume:
            self._load_checkpoint()

        print(f"Scanning dataset root: {self.root}")
        top_folders = self.top_level_folders()
        if not top_folders:
            # dataset root itself may directly contain images (no subfolders)
            top_folders = [self.root]

        for folder in top_folders:
            self._process_top_folder(folder)

        self._cross_folder_duplicates()
        self._build_folder_aggregates()
        print(f"Done. Images analyzed: {len(self.image_rows)}  |  Boxes: {len(self.bbox_rows)}")
        if self.cfg.checkpoint and self._checkpoint_path.exists():
            try:
                self._checkpoint_path.unlink()  # clean up on a full successful run
            except Exception:
                pass

    # -------------------------------------------------------------- checkpoint
    def _save_checkpoint(self):
        try:
            self.cfg.out_dir.mkdir(parents=True, exist_ok=True)
            state = {
                "image_rows": self.image_rows, "bbox_rows": self.bbox_rows,
                "missing_labels": self.missing_labels, "missing_images": self.missing_images,
                "invalid_json_rows": self.invalid_json_rows, "naming_issue_rows": self.naming_issue_rows,
                "errors": self.errors, "hash_index": self._hash_index, "name_index": dict(self._name_index),
            }
            tmp = self._checkpoint_path.with_suffix(".pkl.tmp")
            with open(tmp, "wb") as f:
                pickle.dump(state, f)
            tmp.replace(self._checkpoint_path)  # atomic-ish swap so a crash mid-write can't corrupt it
        except Exception as e:
            print(f"  (non-fatal) checkpoint save failed: {e}")

    def _load_checkpoint(self):
        if not self._checkpoint_path.exists():
            print("  --resume requested but no checkpoint found; starting fresh.")
            return
        try:
            with open(self._checkpoint_path, "rb") as f:
                state = pickle.load(f)
            self.image_rows = state.get("image_rows", [])
            self.bbox_rows = state.get("bbox_rows", [])
            self.missing_labels = state.get("missing_labels", [])
            self.missing_images = state.get("missing_images", [])
            self.invalid_json_rows = state.get("invalid_json_rows", [])
            self.naming_issue_rows = state.get("naming_issue_rows", [])
            self.errors = state.get("errors", [])
            self._hash_index = state.get("hash_index", {})
            self._name_index = defaultdict(list, state.get("name_index", {}))
            self._done_relpaths = {r["relpath"] for r in self.image_rows if r.get("relpath")}
            print(f"  Resumed from checkpoint: {len(self._done_relpaths)} images already processed.")
        except Exception as e:
            print(f"  (non-fatal) checkpoint load failed, starting fresh: {e}")

    # ---------------------------------------------------------- per top folder
    def _process_top_folder(self, folder: Path):
        fmt, coco_path = detect_top_folder_format(folder)
        print(f"\nFolder: {folder.name}  ->  detected annotation format: {fmt}")

        coco_lookup = {}
        if fmt == "coco" and coco_path is not None:
            try:
                coco_lookup, _ = load_coco_json(coco_path)
            except Exception as e:
                self.invalid_json_rows.append({"file": str(coco_path), "error": str(e)})

        yolo_classes = load_yolo_classnames(folder) if fmt == "yolo" else None

        # Build a stem -> label-file index for this top folder. This is what lets us
        # find a label even when it does NOT sit right next to its image, e.g.
        #   americanfootball/imgs/image1.jpg   +   americanfootball/labels/image1.json
        # (a plain img_path.with_suffix(".json") would only find a co-located file).
        label_index = None
        if fmt in ("json", "json_nested", "yolo"):
            label_ext = ".txt" if fmt == "yolo" else ".json"
            label_index = self._build_label_index(folder, label_ext)

        all_images = [p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
        if self.cfg.sample and self.cfg.sample > 0:
            all_images = all_images[: self.cfg.sample]

        # resume support: skip images already recorded in a loaded checkpoint
        if self._done_relpaths:
            before = len(all_images)
            all_images = [p for p in all_images if safe_relpath(p, self.root) not in self._done_relpaths]
            skipped = before - len(all_images)
            if skipped:
                print(f"  Resuming: skipping {skipped} already-processed images in this folder.")

        label_files_seen = set()
        CHECKPOINT_EVERY = 250
        GC_EVERY = 100

        for i, img_path in enumerate(tqdm(all_images, desc=folder.name, total=len(all_images)), start=1):
            self._process_one_image(img_path, folder, fmt, coco_lookup, yolo_classes, label_files_seen, label_index)
            if i % GC_EVERY == 0:
                gc.collect()
            if self.cfg.checkpoint and i % CHECKPOINT_EVERY == 0:
                self._save_checkpoint()

        if self.cfg.checkpoint:
            self._save_checkpoint()

        # labels that exist but have no matching image (only meaningful for json/yolo per-file formats)
        if fmt in ("json", "json_nested", "yolo"):
            ext = ".json" if fmt in ("json", "json_nested") else ".txt"
            all_labels = [p for p in folder.rglob(f"*{ext}") if p.name.lower() not in ("classes.txt", "obj.names", "names.txt")]
            img_stems = {p.stem for p in all_images}
            for lp in all_labels:
                if lp.stem not in img_stems:
                    self.missing_images.append({
                        "label_file": safe_relpath(lp, self.root), "top_folder": folder.name,
                    })

        # naming checks across this folder's images
        self._check_naming(all_images, folder)

    # ----------------------------------------------------------- label lookup
    def _build_label_index(self, folder: Path, ext: str):
        """
        Map image stem -> label file path, scanned once per top-level folder via
        rglob. This is what makes label lookup work regardless of whether the label
        sits next to its image or in a sibling folder (e.g. imgs/ + labels/).

        If two different label files share the same stem (e.g. same filename reused
        under two different class subfolders), the first one found wins and the
        collision is logged as a naming issue rather than silently overwritten.
        """
        index = {}
        skip_names = {"classes.txt", "obj.names", "names.txt"}
        for lp in folder.rglob(f"*{ext}"):
            if lp.name.lower() in skip_names:
                continue
            stem = lp.stem
            if stem in index and index[stem] != lp:
                self.naming_issue_rows.append({
                    "file": f"{safe_relpath(index[stem], self.root)}; {safe_relpath(lp, self.root)}",
                    "top_folder": folder.name, "issues": "duplicate_label_stem_multiple_files",
                })
                continue
            index[stem] = lp
        return index

    # ----------------------------------------------------------- per image
    def _process_one_image(self, img_path: Path, top_folder: Path, fmt, coco_lookup, yolo_classes, label_files_seen, label_index=None):
        """
        Everything in this method is guarded — a single unreadable, corrupt, or
        abnormally huge image (including OpenCV MemoryError/cv2.error) must NEVER
        kill the whole run. Any failure here is recorded in self.errors and the
        scan moves on to the next image.
        """
        relpath = safe_relpath(img_path, self.root)
        subfolder = safe_relpath(img_path.parent, self.root)
        cls_from_folder = img_path.parent.name  # e.g. basketball/rugby/soccer/volleyball

        img_bgr = None
        try:
            # robust load: cv2 first, PIL fallback (handles more formats, unicode paths
            # that cv2.imread can't open on Windows, and EXIF orientation)
            img_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
            if img_bgr is None:
                pil_img = Image.open(img_path)
                pil_img = ImageOps.exif_transpose(pil_img).convert("RGB")
                img_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            h, w = img_bgr.shape[:2]

            # Guard against extreme-resolution images (scanned raws, stitched panoramas,
            # etc.) blowing memory during float64 pixel-stat computation. Cap the copy
            # used for pixel/quality metrics; width/height/resolution stats still use the
            # real dimensions.
            MAX_STAT_SIDE = 4000
            stat_img = img_bgr
            if max(h, w) > MAX_STAT_SIDE:
                scale = MAX_STAT_SIDE / max(h, w)
                stat_img = cv2.resize(img_bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        except Exception as e:
            self.errors.append({"file": relpath, "error": f"corrupt_or_unreadable: {e}"})
            self.image_rows.append({
                "relpath": relpath, "top_folder": top_folder.name, "subfolder": subfolder,
                "class_folder": cls_from_folder, "corrupt": True, "width": None, "height": None,
            })
            return

        try:
            pixel_stats = analyze_image_pixels(stat_img)
            aspect = w / h if h else 0
            orientation = "Square" if abs(aspect - 1) < 0.02 else ("Landscape" if aspect > 1 else "Portrait")

            # letterbox resize preview: scale-to-fit + pad to target_size x target_size,
            # no crop, no aspect-ratio distortion
            ts = self.cfg.target_size
            lb_scale, lb_pad_left, lb_pad_top, lb_content_w, lb_content_h = compute_letterbox_transform(w, h, ts)
            pad_pct_of_canvas = 100 * (1 - (lb_content_w * lb_content_h) / (ts * ts))

            row = {
                "relpath": relpath, "top_folder": top_folder.name, "subfolder": subfolder,
                "class_folder": cls_from_folder, "corrupt": False,
                "width": w, "height": h, "resolution": f"{w}x{h}",
                "aspect_ratio": round(aspect, 4), "orientation": orientation,
                "file_size_kb": round(img_path.stat().st_size / 1024, 2),
                **pixel_stats,
                f"letterbox_{ts}_scale": round(lb_scale, 5),
                f"letterbox_{ts}_content_w_px": round(lb_content_w, 2),
                f"letterbox_{ts}_content_h_px": round(lb_content_h, 2),
                f"letterbox_{ts}_pad_pct_of_canvas": round(pad_pct_of_canvas, 2),
            }

            # duplicate hash index (global, cross-folder)
            self._hash_index.setdefault(row["phash"], []).append(relpath)

            # -------- annotations --------
            boxes, label_relpath, ann_error, ann_warning = self._load_boxes_for_image(
                img_path, fmt, coco_lookup, yolo_classes, w, h, label_index)
            if label_relpath:
                label_files_seen.add(label_relpath)
            if ann_warning:
                # non-fatal issue (e.g. declared vs actual image dimensions mismatch) -
                # logged for visibility but boxes below are still used normally.
                self.invalid_json_rows.append({"file": label_relpath or relpath, "error": ann_warning})
            if ann_error == "missing":
                self.missing_labels.append({"image_file": relpath, "top_folder": top_folder.name})
                row["n_objects"] = 0
            elif ann_error:
                self.invalid_json_rows.append({"file": label_relpath or relpath, "error": ann_error})
                row["n_objects"] = 0
            else:
                row["n_objects"] = len(boxes)
                for b in boxes:
                    m = compute_bbox_metrics(b, w, h, target_size=ts, tiny_px_threshold=self.cfg.tiny_px_threshold)
                    m.update({"image_relpath": relpath, "top_folder": top_folder.name, "subfolder": subfolder})
                    self.bbox_rows.append(m)
                    if m["issues"]:
                        self.invalid_json_rows.append({
                            "file": label_relpath or relpath, "error": m["issues"],
                            "class": m["cls"], "image": relpath,
                        })

            self.image_rows.append(row)
        except (MemoryError, cv2.error, Exception) as e:
            # Catch-all: whatever the exact failure (OpenCV OutOfMemoryError, malformed
            # array, unexpected annotation shape, etc.) we log it and keep going instead
            # of losing the rest of the dataset.
            self.errors.append({"file": relpath, "error": f"processing_failed: {type(e).__name__}: {e}"})
            self.image_rows.append({
                "relpath": relpath, "top_folder": top_folder.name, "subfolder": subfolder,
                "class_folder": cls_from_folder, "corrupt": True, "width": None, "height": None,
            })
        finally:
            del img_bgr

    def _load_boxes_for_image(self, img_path: Path, fmt, coco_lookup, yolo_classes, w, h, label_index=None):
        """
        Returns (boxes, label_relpath, error_or_None, warning_or_None).

        For the json/json_nested/yolo formats, the label file is looked up by image
        stem in `label_index` first (built once per top folder via a recursive scan -
        this is what finds a label that lives in a different subfolder than its image,
        e.g. <class>/imgs/x.jpg + <class>/labels/x.json). Falls back to a co-located
        file (img_path.with_suffix(...)) if no index was built or the stem isn't in it,
        so plain flat layouts keep working unchanged.
        """
        if fmt == "coco":
            boxes = coco_lookup.get(img_path.name)
            if boxes is None:
                return [], None, "missing", None
            return boxes, img_path.name, None, None
        elif fmt == "yolo":
            txt_path = (label_index or {}).get(img_path.stem) or img_path.with_suffix(".txt")
            if not txt_path.exists():
                return [], None, "missing", None
            boxes, err = parse_yolo_label(txt_path, w, h, yolo_classes)
            return boxes, safe_relpath(txt_path, self.root), err, None
        elif fmt == "json_nested":
            json_path = (label_index or {}).get(img_path.stem) or img_path.with_suffix(".json")
            if not json_path.exists():
                return [], None, "missing", None
            boxes, err, warning = parse_nested_data_json_label(json_path, w, h)
            return boxes, safe_relpath(json_path, self.root), err, warning
        elif fmt == "json":
            json_path = (label_index or {}).get(img_path.stem) or img_path.with_suffix(".json")
            if not json_path.exists():
                return [], None, "missing", None
            boxes, err = parse_generic_json_label(json_path, w, h)
            return boxes, safe_relpath(json_path, self.root), err, None
        else:
            return [], None, "missing", None

    # ------------------------------------------------------------- naming
    def _check_naming(self, images, folder: Path):
        seen_lower = defaultdict(list)
        for p in images:
            self._name_index[p.name.lower()].append(safe_relpath(p, self.root))
            seen_lower[p.name.lower()].append(p)
            issues = []
            if is_probably_unicode_name(p.stem):
                issues.append("non_ascii_filename")
            if p.suffix.lower() in UNSUPPORTED_EXT_WARN:
                issues.append("unsupported_or_risky_extension")
            if issues:
                self.naming_issue_rows.append({
                    "file": safe_relpath(p, self.root), "top_folder": folder.name,
                    "issues": ";".join(issues),
                })
        for lname, plist in seen_lower.items():
            if len(plist) > 1:
                self.naming_issue_rows.append({
                    "file": "; ".join(safe_relpath(x, self.root) for x in plist),
                    "top_folder": folder.name, "issues": "duplicate_filename_same_folder",
                })

    # ---------------------------------------------------- cross-folder dupes
    def _cross_folder_duplicates(self):
        print("\nComputing duplicate / near-duplicate image groups...")
        items = list(self._hash_index.items())
        exact_groups = [(h, paths) for h, paths in items if len(paths) > 1]
        for h, paths in exact_groups:
            for p in paths:
                self.duplicate_rows.append({"file": p, "hash": f"{h:016x}", "match_type": "exact_or_phash_identical",
                                             "group_size": len(paths), "group_members": "; ".join(paths)})

        # near-duplicates via hamming distance (only within a bounded sample to stay fast on huge sets)
        hashes = [h for h, _ in items]
        n = len(hashes)
        MAX_PAIRWISE = 6000  # guard: full O(n^2) only below this many unique hashes
        if n <= MAX_PAIRWISE:
            for i in range(n):
                for j in range(i + 1, n):
                    d = hamming(hashes[i], hashes[j])
                    if 0 < d <= self.cfg.hash_threshold:
                        for p1 in items[i][1]:
                            for p2 in items[j][1]:
                                self.duplicate_rows.append({
                                    "file": p1, "hash": f"{hashes[i]:016x}", "match_type": f"near_duplicate(d={d})",
                                    "group_size": 2, "group_members": f"{p1}; {p2}",
                                })
        else:
            print(f"  Skipping exhaustive near-duplicate search ({n} unique hashes > {MAX_PAIRWISE}); "
                  f"exact-duplicate detection still ran.")

        # cross-folder same filename (informational)
        for lname, paths in self._name_index.items():
            folders = {Path(p).parts[0] for p in paths}
            if len(paths) > 1 and len(folders) > 1:
                self.naming_issue_rows.append({
                    "file": "; ".join(paths), "top_folder": "; ".join(sorted(folders)),
                    "issues": "duplicate_filename_across_folders",
                })

    # -------------------------------------------------------- aggregation
    def _build_folder_aggregates(self):
        if not self.image_rows:
            return
        img_df = pd.DataFrame(self.image_rows)
        box_df = pd.DataFrame(self.bbox_rows) if self.bbox_rows else pd.DataFrame(
            columns=["top_folder", "subfolder", "cls"])

        for (top, sub), g in img_df.groupby(["top_folder", "subfolder"], dropna=False):
            valid = g[g["corrupt"] == False]  # noqa: E712
            b = box_df[(box_df["top_folder"] == top) & (box_df["subfolder"] == sub)] if len(box_df) else box_df
            self.folder_rows.append({
                "top_folder": top, "subfolder": sub,
                "images": len(g), "corrupt_images": int(g["corrupt"].sum()),
                "objects": len(b),
                "avg_objects_per_image": round(len(b) / max(len(valid), 1), 3),
                "mean_width": round(valid["width"].mean(), 1) if len(valid) else None,
                "mean_height": round(valid["height"].mean(), 1) if len(valid) else None,
                "mean_brightness": round(valid["brightness"].mean(), 2) if len(valid) else None,
                "mean_contrast": round(valid["contrast"].mean(), 2) if len(valid) else None,
                "mean_blur_score": round(valid["blur_score"].mean(), 2) if len(valid) else None,
                "pct_blurry": round(100 * (valid["blur_bucket"].isin(["Very Blurry", "Blurry"])).mean(), 2) if len(valid) else None,
                "pct_over_under_exposed": round(100 * (valid["exposure"] != "Normal").mean(), 2) if len(valid) else None,
            })


# =============================================================================
# EXPORT: DATAFRAMES
# =============================================================================

class ResultFrames:
    def __init__(self, analyzer: DatasetAnalyzer):
        self.images = pd.DataFrame(analyzer.image_rows)
        self.bboxes = pd.DataFrame(analyzer.bbox_rows)
        self.folders = pd.DataFrame(analyzer.folder_rows)
        self.missing_labels = pd.DataFrame(analyzer.missing_labels)
        self.missing_images = pd.DataFrame(analyzer.missing_images)
        self.invalid = pd.DataFrame(analyzer.invalid_json_rows)
        self.duplicates = pd.DataFrame(analyzer.duplicate_rows)
        self.naming = pd.DataFrame(analyzer.naming_issue_rows)
        self.errors = pd.DataFrame(analyzer.errors)
        self.target_size = analyzer.cfg.target_size

    # ---- derived summary tables ----
    def class_distribution(self):
        if self.bboxes.empty:
            return pd.DataFrame(columns=["cls", "objects", "images_with_class"])
        obj_counts = self.bboxes.groupby("cls").size().rename("objects")
        img_counts = self.bboxes.groupby("cls")["image_relpath"].nunique().rename("images_with_class")
        out = pd.concat([obj_counts, img_counts], axis=1).reset_index().sort_values("objects", ascending=False)
        if len(out):
            mx, mn = out["objects"].max(), max(out["objects"].min(), 1)
            out["imbalance_ratio_vs_max"] = round(mx / out["objects"].replace(0, np.nan), 2)
        return out

    def resolution_distribution(self):
        if self.images.empty:
            return pd.DataFrame(columns=["resolution", "count"])
        return (self.images[self.images["corrupt"] == False]  # noqa: E712
                .groupby("resolution").size().rename("count")
                .reset_index().sort_values("count", ascending=False))

    def object_count_histogram(self):
        if self.images.empty:
            return pd.DataFrame(columns=["n_objects", "count"])
        return (self.images.groupby("n_objects").size().rename("count").reset_index()
                .sort_values("n_objects"))

    def ball_size_distribution(self):
        if self.bboxes.empty:
            return pd.DataFrame(columns=["size_bucket", "count", "pct"])
        vc = self.bboxes["size_bucket"].value_counts()
        df = vc.rename("count").reset_index()
        df.columns = ["size_bucket", "count"]
        df["pct"] = round(100 * df["count"] / df["count"].sum(), 2)
        order = [b[0] for b in SIZE_BUCKETS]
        df["size_bucket"] = pd.Categorical(df["size_bucket"], categories=order, ordered=True)
        return df.sort_values("size_bucket")

    def resized_size_distribution(self):
        """Same Tiny/Small/Medium/Large buckets, computed on the post-letterbox-resize area."""
        col = f"resized_{self.target_size}_size_bucket"
        if self.bboxes.empty or col not in self.bboxes.columns:
            return pd.DataFrame(columns=["size_bucket", "count", "pct"])
        vc = self.bboxes[col].value_counts()
        df = vc.rename("count").reset_index()
        df.columns = ["size_bucket", "count"]
        df["pct"] = round(100 * df["count"] / df["count"].sum(), 2)
        order = [b[0] for b in SIZE_BUCKETS]
        df["size_bucket"] = pd.Categorical(df["size_bucket"], categories=order, ordered=True)
        return df.sort_values("size_bucket")

    def resize_impact_summary(self):
        """
        Per-class summary of object pixel size before vs. after a letterbox resize to
        target_size x target_size (uniform scale + pad, no crop/distortion).
        """
        ts = self.target_size
        w_col, h_col = f"resized_{ts}_width_px", f"resized_{ts}_height_px"
        tiny_col = f"tiny_after_resize_{ts}"
        cols_needed = {"cls", "width", "height", w_col, h_col, tiny_col}
        if self.bboxes.empty or not cols_needed.issubset(self.bboxes.columns):
            return pd.DataFrame(columns=[
                "cls", "objects", "avg_orig_width_px", "avg_orig_height_px",
                f"avg_resized_{ts}_width_px", f"avg_resized_{ts}_height_px",
                f"pct_tiny_after_resize_{ts}",
            ])
        g = self.bboxes.groupby("cls")
        out = pd.DataFrame({
            "objects": g.size(),
            "avg_orig_width_px": g["width"].mean().round(1),
            "avg_orig_height_px": g["height"].mean().round(1),
            f"avg_resized_{ts}_width_px": g[w_col].mean().round(2),
            f"avg_resized_{ts}_height_px": g[h_col].mean().round(2),
            f"pct_tiny_after_resize_{ts}": (100 * g[tiny_col].mean()).round(2),
        }).reset_index().sort_values("objects", ascending=False)
        return out

    def coco_readiness(self):
        checks = {
            "Missing labels": len(self.missing_labels),
            "Missing images (label with no image)": len(self.missing_images),
            "Invalid / malformed annotation entries": len(self.invalid),
            "Corrupt images": int(self.images["corrupt"].sum()) if not self.images.empty else 0,
            "Duplicate images (exact/near)": self.duplicates["file"].nunique() if not self.duplicates.empty else 0,
            "Empty-label files": int((self.invalid.get("error", pd.Series(dtype=str)) == "empty_labels").sum()) if not self.invalid.empty else 0,
        }
        rows = [{"check": k, "count": v, "status": "OK" if v == 0 else "REVIEW"} for k, v in checks.items()]
        return pd.DataFrame(rows)


# =============================================================================
# PLOTS
# =============================================================================

def generate_plots(rf: ResultFrames, out_dir: Path):
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

    valid_imgs = rf.images[rf.images.get("corrupt", pd.Series(dtype=bool)) == False] if not rf.images.empty else rf.images  # noqa

    # Resolution histogram (top 20)
    if not valid_imgs.empty:
        res = rf.resolution_distribution().head(20)
        if len(res):
            plt.figure(figsize=(9, 5))
            plt.barh(res["resolution"].astype(str), res["count"], color="#4C72B0")
            plt.gca().invert_yaxis()
            plt.xlabel("Image count"); plt.title("Top 20 Resolutions")
            savefig("resolution.png")

    # Brightness / Contrast / Blur / Saturation / Entropy histograms
    for col, title, fname in [
        ("brightness", "Brightness Distribution", "brightness.png"),
        ("contrast", "Contrast Distribution", "contrast.png"),
        ("blur_score", "Blur (Variance of Laplacian) Distribution", "blur.png"),
        ("saturation", "Saturation Distribution", "saturation.png"),
        ("entropy", "Entropy Distribution", "entropy.png"),
    ]:
        if not valid_imgs.empty and col in valid_imgs:
            plt.figure(figsize=(8, 5))
            plt.hist(valid_imgs[col].dropna(), bins=40, color="#55A868")
            plt.xlabel(col); plt.ylabel("Images"); plt.title(title)
            savefig(fname)

    # RGB mean histogram overlay
    if not valid_imgs.empty and {"mean_r", "mean_g", "mean_b"}.issubset(valid_imgs.columns):
        plt.figure(figsize=(8, 5))
        plt.hist(valid_imgs["mean_r"].dropna(), bins=40, alpha=0.5, label="R", color="red")
        plt.hist(valid_imgs["mean_g"].dropna(), bins=40, alpha=0.5, label="G", color="green")
        plt.hist(valid_imgs["mean_b"].dropna(), bins=40, alpha=0.5, label="B", color="blue")
        plt.legend(); plt.title("Mean RGB Channel Distribution"); plt.xlabel("Mean channel value")
        savefig("rgb_histogram.png")

    # Object count histogram
    oc = rf.object_count_histogram()
    if len(oc):
        plt.figure(figsize=(8, 5))
        plt.bar(oc["n_objects"].astype(str), oc["count"], color="#C44E52")
        plt.xlabel("Objects per image"); plt.ylabel("Image count"); plt.title("Object Count Histogram")
        plt.xticks(rotation=90 if len(oc) > 25 else 0)
        savefig("object_count.png")

    # BBox area (relative %) histogram
    if not rf.bboxes.empty:
        plt.figure(figsize=(8, 5))
        plt.hist(rf.bboxes["rel_area_pct"].clip(upper=rf.bboxes["rel_area_pct"].quantile(0.99)), bins=50, color="#8172B2")
        plt.xlabel("BBox area as % of image"); plt.ylabel("Boxes"); plt.title("Bounding Box Relative Area")
        savefig("bbox_area.png")

        # bbox scatter (width vs height)
        plt.figure(figsize=(7, 7))
        s = rf.bboxes.sample(min(len(rf.bboxes), 5000), random_state=0)
        plt.scatter(s["width"], s["height"], s=6, alpha=0.4, color="#4C72B0")
        plt.xlabel("BBox width (px)"); plt.ylabel("BBox height (px)"); plt.title("Bounding Box Width vs Height")
        savefig("bbox_scatter.png")

        # center heatmap
        plt.figure(figsize=(6, 6))
        heat, xedges, yedges = np.histogram2d(rf.bboxes["center_x_rel"], rf.bboxes["center_y_rel"],
                                               bins=25, range=[[0, 1], [0, 1]])
        plt.imshow(heat.T, origin="upper", extent=[0, 1, 1, 0], cmap="inferno", aspect="auto")
        plt.colorbar(label="Box count")
        plt.xlabel("Relative X"); plt.ylabel("Relative Y"); plt.title("Bounding Box Center Heatmap")
        savefig("heatmap.png")

    # Class distribution
    cdist = rf.class_distribution()
    if len(cdist):
        plt.figure(figsize=(9, 5))
        top = cdist.head(30)
        plt.bar(top["cls"].astype(str), top["objects"], color="#DD8452")
        plt.xticks(rotation=60, ha="right"); plt.ylabel("Object count"); plt.title("Class Distribution (objects)")
        savefig("class_distribution.png")

    # Folder comparison (mean brightness / blur per top folder)
    if not rf.folders.empty:
        agg = rf.folders.groupby("top_folder").agg(
            images=("images", "sum"),
            mean_brightness=("mean_brightness", "mean"),
            mean_blur_score=("mean_blur_score", "mean"),
        ).reset_index()
        if len(agg):
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            axes[0].bar(agg["top_folder"], agg["mean_brightness"], color="#4C72B0")
            axes[0].set_title("Mean Brightness by Folder"); axes[0].tick_params(axis="x", rotation=75)
            axes[1].bar(agg["top_folder"], agg["mean_blur_score"], color="#55A868")
            axes[1].set_title("Mean Blur Score by Folder"); axes[1].tick_params(axis="x", rotation=75)
            savefig("folder_comparison.png")

    # Ball / object size bucket distribution
    bsd = rf.ball_size_distribution()
    if len(bsd):
        plt.figure(figsize=(6, 6))
        plt.pie(bsd["count"], labels=[f"{r.size_bucket}\n{r.pct}%" for r in bsd.itertuples()],
                colors=["#4C72B0", "#55A868", "#C44E52", "#8172B2"])
        plt.title("Object Size Buckets (relative area)")
        savefig("size_buckets.png")

    # Object pixel size before vs. after letterbox resize to target_size x target_size
    ts = rf.target_size
    w_col, h_col = f"resized_{ts}_width_px", f"resized_{ts}_height_px"
    if not rf.bboxes.empty and {w_col, h_col}.issubset(rf.bboxes.columns):
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        max_side = max(rf.bboxes["width"].quantile(0.99), rf.bboxes["height"].quantile(0.99), 1)
        bins = np.linspace(0, max_side, 40)
        axes[0].hist(rf.bboxes["width"].clip(upper=max_side), bins=bins, alpha=0.6, label="width", color="#4C72B0")
        axes[0].hist(rf.bboxes["height"].clip(upper=max_side), bins=bins, alpha=0.6, label="height", color="#DD8452")
        axes[0].set_title("Original Object Size (px)"); axes[0].set_xlabel("pixels"); axes[0].legend()

        max_side_r = max(rf.bboxes[w_col].quantile(0.99), rf.bboxes[h_col].quantile(0.99), 1)
        bins_r = np.linspace(0, max_side_r, 40)
        axes[1].hist(rf.bboxes[w_col].clip(upper=max_side_r), bins=bins_r, alpha=0.6, label="width", color="#4C72B0")
        axes[1].hist(rf.bboxes[h_col].clip(upper=max_side_r), bins=bins_r, alpha=0.6, label="height", color="#DD8452")
        axes[1].set_title(f"Object Size After {ts}x{ts} Letterbox Resize (px)"); axes[1].set_xlabel("pixels"); axes[1].legend()
        savefig(f"resize_impact_{ts}px.png")

        rsd = rf.resized_size_distribution()
        if len(rsd):
            plt.figure(figsize=(6, 6))
            plt.pie(rsd["count"], labels=[f"{r.size_bucket}\n{r.pct}%" for r in rsd.itertuples()],
                    colors=["#4C72B0", "#55A868", "#C44E52", "#8172B2"])
            plt.title(f"Object Size Buckets After {ts}x{ts} Resize")
            savefig(f"size_buckets_after_resize_{ts}px.png")

    print(f"Saved {len(saved)} plots to {plots_dir}")
    return saved


# =============================================================================
# EXPORT: EXCEL + CSV
# =============================================================================

def export_excel_and_csv(rf: ResultFrames, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    xlsx_path = out_dir / "dataset_summary.xlsx"

    sheets = {
        "Folder Stats": rf.folders,
        "Class Distribution": rf.class_distribution(),
        "Resolution Distribution": rf.resolution_distribution(),
        "Object Count Histogram": rf.object_count_histogram(),
        "Ball Size Distribution": rf.ball_size_distribution(),
        f"Resize Impact {rf.target_size}px": rf.resize_impact_summary(),
        f"Resized Size Distribution": rf.resized_size_distribution(),
        "COCO Readiness": rf.coco_readiness(),
        "Missing Labels": rf.missing_labels,
        "Missing Images": rf.missing_images,
        "Invalid Annotations": rf.invalid,
        "Duplicate Images": rf.duplicates,
        "Naming Issues": rf.naming,
        "Read Errors": rf.errors,
    }

    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        for name, df in sheets.items():
            safe_name = name[:31]
            (df if not df.empty else pd.DataFrame({"info": ["no rows"]})).to_excel(writer, sheet_name=safe_name, index=False)
        # image + bbox level detail (can be large -> still included, client asked for full audit)
        (rf.images if not rf.images.empty else pd.DataFrame({"info": ["no rows"]})).to_excel(
            writer, sheet_name="Image Statistics", index=False)
        (rf.bboxes if not rf.bboxes.empty else pd.DataFrame({"info": ["no rows"]})).to_excel(
            writer, sheet_name="BBox Statistics", index=False)

    _autosize_excel(xlsx_path)

    # CSVs (client asked for these explicitly, alongside the workbook)
    rf.images.to_csv(out_dir / "image_statistics.csv", index=False)
    rf.bboxes.to_csv(out_dir / "bbox_statistics.csv", index=False)
    rf.folders.to_csv(out_dir / "folder_statistics.csv", index=False)
    rf.duplicates.to_csv(out_dir / "duplicate_images.csv", index=False)
    rf.invalid.to_csv(out_dir / "invalid_annotations.csv", index=False)
    rf.resize_impact_summary().to_csv(out_dir / f"resize_impact_{rf.target_size}px.csv", index=False)
    if not rf.images.empty:
        blurry = rf.images[rf.images.get("blur_bucket", "").isin(["Very Blurry", "Blurry"])]
        blurry.to_csv(out_dir / "blurry_images.csv", index=False)
    combined_summary = rf.folders.copy()
    combined_summary.to_csv(out_dir / "dataset_summary.csv", index=False)

    print(f"Excel workbook: {xlsx_path}")
    print(f"CSV files written to: {out_dir}")
    return xlsx_path


def _autosize_excel(xlsx_path: Path, max_width=40):
    try:
        from openpyxl import load_workbook
        wb = load_workbook(xlsx_path)
        for ws in wb.worksheets:
            for col_cells in ws.columns:
                length = max((len(str(c.value)) if c.value is not None else 0) for c in col_cells)
                col_letter = col_cells[0].column_letter
                ws.column_dimensions[col_letter].width = min(max(length + 2, 10), max_width)
            ws.freeze_panes = "A2"
        wb.save(xlsx_path)
    except Exception as e:
        print(f"  (non-fatal) could not autosize Excel columns: {e}")


# =============================================================================
# HTML REPORT
# =============================================================================

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Dataset Audit Report</title>
<style>
  body {{ font-family: -apple-system, Segoe UI, Roboto, Arial, sans-serif; margin: 0; background: #f5f6f8; color: #1f2430; }}
  header {{ background: #1f2937; color: white; padding: 28px 40px; }}
  header h1 {{ margin: 0 0 6px 0; font-size: 26px; }}
  header p {{ margin: 0; color: #b7c0cc; font-size: 14px; }}
  .wrap {{ max-width: 1180px; margin: 0 auto; padding: 24px 40px 60px; }}
  .cards {{ display: flex; gap: 16px; flex-wrap: wrap; margin: 20px 0 34px; }}
  .card {{ background: white; border-radius: 10px; padding: 16px 20px; box-shadow: 0 1px 3px rgba(0,0,0,.08); min-width: 160px; flex: 1; }}
  .card .num {{ font-size: 26px; font-weight: 700; color: #2563eb; }}
  .card .lbl {{ font-size: 12.5px; color: #666; margin-top: 4px; }}
  h2 {{ border-bottom: 2px solid #e5e7eb; padding-bottom: 8px; margin-top: 46px; font-size: 19px; color: #111827; }}
  table {{ border-collapse: collapse; width: 100%; margin: 14px 0 8px; background: white; font-size: 13px; }}
  th, td {{ border: 1px solid #e5e7eb; padding: 6px 10px; text-align: left; }}
  th {{ background: #f0f2f5; position: sticky; top: 0; }}
  tr:nth-child(even) {{ background: #fafbfc; }}
  .status-OK {{ color: #059669; font-weight: 600; }}
  .status-REVIEW {{ color: #dc2626; font-weight: 600; }}
  .plot-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(430px, 1fr)); gap: 18px; margin: 16px 0; }}
  .plot-grid img {{ width: 100%; border-radius: 8px; border: 1px solid #e5e7eb; background: white; }}
  .scroll {{ max-height: 480px; overflow: auto; border: 1px solid #e5e7eb; border-radius: 8px; }}
  footer {{ text-align: center; color: #999; font-size: 12px; margin-top: 50px; }}
</style>
</head>
<body>
<header>
  <h1>Dataset Audit Report</h1>
  <p>Root: {root} &nbsp;|&nbsp; Generated: {timestamp}</p>
</header>
<div class="wrap">

  <div class="cards">
    {summary_cards}
  </div>

  <h2>COCO / Object-Detection Readiness</h2>
  {coco_table}

  <h2>Folder Statistics</h2>
  <div class="scroll">{folder_table}</div>

  <h2>Class Distribution</h2>
  <div class="scroll">{class_table}</div>

  <h2>Object Pixel Size: Original vs. After {target_size}x{target_size} Resize</h2>
  <p style="color:#666;font-size:13px">Letterbox resize: uniform scale-to-fit + centered padding — no crop, no aspect-ratio distortion.</p>
  <div class="scroll">{resize_table}</div>

  <h2>Charts</h2>
  <div class="plot-grid">
    {plot_imgs}
  </div>

  <h2>Missing Labels ({n_missing_labels} images)</h2>
  <div class="scroll">{missing_labels_table}</div>

  <h2>Missing Images ({n_missing_images} labels)</h2>
  <div class="scroll">{missing_images_table}</div>

  <h2>Invalid / Flagged Annotations ({n_invalid} entries)</h2>
  <div class="scroll">{invalid_table}</div>

  <h2>Duplicate Images ({n_dupes} entries)</h2>
  <div class="scroll">{dup_table}</div>

  <h2>File Naming Issues ({n_naming} entries)</h2>
  <div class="scroll">{naming_table}</div>

  <footer>Generated by dataset_analyzer.py</footer>
</div>
</body>
</html>
"""


def _df_to_html(df: pd.DataFrame, max_rows=300):
    if df is None or df.empty:
        return "<p><em>None found.</em></p>"
    shown = df.head(max_rows)
    note = f"<p style='color:#888;font-size:12px'>Showing {len(shown)} of {len(df)} rows — full detail in the CSV/Excel export.</p>" if len(df) > max_rows else ""
    return note + shown.to_html(index=False, border=0, escape=True)


def generate_html_report(rf: ResultFrames, out_dir: Path, root: Path, plot_paths):
    valid_imgs = rf.images[rf.images.get("corrupt", pd.Series(dtype=bool)) == False] if not rf.images.empty else rf.images  # noqa
    n_images = len(rf.images)
    n_corrupt = int(rf.images["corrupt"].sum()) if not rf.images.empty else 0
    n_objects = len(rf.bboxes)
    n_classes = rf.bboxes["cls"].nunique() if not rf.bboxes.empty else 0

    cards = [
        ("Images", n_images), ("Objects", n_objects), ("Classes", n_classes),
        ("Corrupt Images", n_corrupt),
        ("Missing Labels", len(rf.missing_labels)), ("Missing Images", len(rf.missing_images)),
        ("Duplicate Entries", rf.duplicates["file"].nunique() if not rf.duplicates.empty else 0),
        ("Invalid Annotations", len(rf.invalid)),
    ]
    summary_cards = "".join(
        f'<div class="card"><div class="num">{v}</div><div class="lbl">{k}</div></div>' for k, v in cards
    )

    coco_df = rf.coco_readiness()
    if not coco_df.empty:
        coco_df = coco_df.copy()
        coco_df["status"] = coco_df["status"].apply(lambda s: f'<span class="status-{s}">{s}</span>')
    coco_table = coco_df.to_html(index=False, border=0, escape=False) if not coco_df.empty else "<p>N/A</p>"

    plot_imgs = "".join(
        f'<div><img src="plots/{p.name}" alt="{p.stem}"><div style="text-align:center;font-size:12px;color:#666;margin-top:4px">{p.stem.replace("_"," ").title()}</div></div>'
        for p in plot_paths
    )

    html = HTML_TEMPLATE.format(
        root=str(root), timestamp=datetime.now().strftime("%Y-%m-%d %H:%M"),
        summary_cards=summary_cards,
        coco_table=coco_table,
        folder_table=_df_to_html(rf.folders),
        class_table=_df_to_html(rf.class_distribution()),
        target_size=rf.target_size, resize_table=_df_to_html(rf.resize_impact_summary()),
        plot_imgs=plot_imgs,
        n_missing_labels=len(rf.missing_labels), missing_labels_table=_df_to_html(rf.missing_labels),
        n_missing_images=len(rf.missing_images), missing_images_table=_df_to_html(rf.missing_images),
        n_invalid=len(rf.invalid), invalid_table=_df_to_html(rf.invalid),
        n_dupes=len(rf.duplicates), dup_table=_df_to_html(rf.duplicates),
        n_naming=len(rf.naming), naming_table=_df_to_html(rf.naming),
    )
    report_path = out_dir / "report.html"
    report_path.write_text(html, encoding="utf-8")
    print(f"HTML report: {report_path}")
    return report_path


# =============================================================================
# CLI ENTRY POINT
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Deep recursive dataset auditor for object-detection datasets (images + JSON/YOLO/COCO labels).")
    parser.add_argument("--root", required=True, type=str, help="Path to the dataset root folder.")
    parser.add_argument("--out", default="./analysis", type=str, help="Output directory for the report bundle.")
    parser.add_argument("--bbox-format", default="auto", choices=["auto", "xywh", "xyxy"],
                         help="Hint for generic-JSON bbox interpretation (auto-detection is used by default).")
    parser.add_argument("--hash-threshold", default=5, type=int,
                         help="Hamming distance threshold (0-64) for near-duplicate image detection.")
    parser.add_argument("--target-size", default=320, type=int,
                         help="Target square size (px) to preview a letterbox resize to (uniform scale + "
                              "centered padding, no crop/distortion) — reports each object's pixel size "
                              "before and after. Default: 320 (320x320).")
    parser.add_argument("--tiny-px-threshold", default=4.0, type=float,
                         help="An object is flagged 'tiny after resize' if its width or height drops below "
                              "this many pixels post-resize. Default: 4.0.")
    parser.add_argument("--sample", default=0, type=int,
                         help="If >0, only analyze this many images per top-level folder (fast preview run).")
    parser.add_argument("--no-plots", action="store_true", help="Skip PNG chart generation.")
    parser.add_argument("--no-html", action="store_true", help="Skip HTML report generation.")
    parser.add_argument("--no-checkpoint", action="store_true",
                         help="Disable periodic checkpointing (checkpointing is on by default so a crash "
                              "on a long run doesn't lose progress).")
    parser.add_argument("--resume", action="store_true",
                         help="Resume from the checkpoint left in --out by a previous interrupted run.")
    args = parser.parse_args()

    root = Path(args.root).expanduser().resolve()
    out_dir = Path(args.out).expanduser().resolve()
    if not root.exists():
        print(f"ERROR: root path does not exist: {root}")
        sys.exit(1)

    cfg = AnalyzerConfig(
        root=root, out_dir=out_dir, bbox_format=args.bbox_format,
        hash_threshold=args.hash_threshold, sample=args.sample,
        make_plots=not args.no_plots, make_html=not args.no_html,
        checkpoint=not args.no_checkpoint, resume=args.resume,
        target_size=args.target_size, tiny_px_threshold=args.tiny_px_threshold,
    )

    t0 = datetime.now()
    analyzer = DatasetAnalyzer(cfg)

    # Safety net: per-image failures are already caught inside the scan (see
    # _process_one_image), but if something still crashes the run (Ctrl+C, disk full,
    # an error in an unusual code path), we still want to export whatever was gathered
    # so far rather than losing everything - the checkpoint on disk also lets --resume
    # pick this back up.
    crashed = False
    try:
        analyzer.run()
    except KeyboardInterrupt:
        print("\nInterrupted by user - saving checkpoint and exporting partial results...")
        analyzer._save_checkpoint()
        crashed = True
    except Exception as e:
        print(f"\nERROR: scan stopped early: {e}")
        traceback.print_exc()
        print("Saving checkpoint and exporting whatever was collected so far. "
              "Re-run with --resume to continue from here.")
        analyzer._save_checkpoint()
        crashed = True

    rf = ResultFrames(analyzer)
    out_dir.mkdir(parents=True, exist_ok=True)
    export_excel_and_csv(rf, out_dir)

    plot_paths = []
    if cfg.make_plots:
        try:
            plot_paths = generate_plots(rf, out_dir)
        except Exception as e:
            print(f"WARNING: plot generation failed: {e}")
            traceback.print_exc()

    if cfg.make_html:
        try:
            generate_html_report(rf, out_dir, root, plot_paths)
        except Exception as e:
            print(f"WARNING: HTML report generation failed: {e}")
            traceback.print_exc()

    elapsed = (datetime.now() - t0).total_seconds()
    if crashed:
        print(f"\nFinished with partial results after {elapsed:.1f}s (run stopped early - see above). "
              f"Results in: {out_dir}\nRe-run the same command with --resume to continue.")
        sys.exit(1)
    else:
        print(f"\nAll done in {elapsed:.1f}s. Results in: {out_dir}")


if __name__ == "__main__":
    main()

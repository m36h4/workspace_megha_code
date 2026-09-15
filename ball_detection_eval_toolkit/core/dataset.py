"""
Dataset loaders for ball detection ground truth, in THREE annotation formats:

    1. "client" (Nikon)  -- folder-per-sport, one JSON per image
    2. "coco"            -- single COCO-style instances json for the whole set
    3. "yolo"             -- one .txt per image, normalized cxcywh, + a data.yaml

All three converge on the same internal representation so every downstream
component (metrics, report, backends) is format-agnostic:

    ImageSample.gt_boxes_xywh : (N, 4) float32, [x, y, w, h] TOP-LEFT based,
                                in ORIGINAL image pixel coordinates.

This was fixed as the internal convention per client decision, regardless of
which annotation format the input data arrives in.

---------------------------------------------------------------------------
FORMAT 1 -- "client" (Nikon), the format actually used by the client:

    basic_train_caseA/
        basketball/
            imgs/xxx.jpg
            labels/xxx.json
        soccer/...
        volleyball/...

    Label JSON:
    {
        "data": {
            "ball": [
                {"entire": {"rect": [x, y, w, h], "attr": "None"}},
                ...
            ]
        }
    }

---------------------------------------------------------------------------
FORMAT 2 -- "coco":

    One instances-style json (COCO detection format):
    {
      "images": [{"id": 1, "file_name": "...", "width":.., "height":..}, ...],
      "annotations": [{"image_id":1, "category_id":1, "bbox":[x,y,w,h], ...}, ...],
      "categories": [{"id":1, "name":"ball"}, ...]
    }
    bbox is already [x, y, w, h] top-left in COCO -- no conversion needed for
    box format, only need to join images<->annotations by image_id and locate
    image files on disk (image_dir + file_name).
    Sport is not part of COCO's schema; derived from an optional custom field
    (`images[i]["sport"]`) if present, else from the file_name's parent dir,
    else "unknown".

---------------------------------------------------------------------------
FORMAT 3 -- "yolo":

    images/xxx.jpg
    labels/xxx.txt      # one line per box: "<class> <cx> <cy> <w> <h>", all
                         # normalized 0-1 relative to image width/height
    data.yaml           # optional, for class names / sport subfolder mapping

    Requires reading each image's actual W,H to denormalize into pixel
    coordinates (cv2 image read, or a fast header-only size read).
---------------------------------------------------------------------------
"""

import json
import os
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Dict, Tuple

import cv2
import numpy as np


@dataclass
class ImageSample:
    image_path: str
    sport: str                      # "basketball" | "soccer" | "volleyball" | "unknown"
    gt_boxes_xywh: np.ndarray       # (N, 4) float32, [x, y, w, h] top-left, original-image pixels
    image_id: str                   # stable id, derived from filename (stem) or source id


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def _infer_sport_from_name(name: str) -> str:
    name = name.lower()
    for known in ("basketball", "soccer", "volleyball"):
        if known in name:
            return known
    return "unknown"


def _read_image_size_fast(path: str) -> Tuple[int, int]:
    """Return (width, height) without fully decoding the image, when possible."""
    try:
        with open(path, "rb") as f:
            head = f.read(32)
        if head[:2] == b"\xff\xd8":  # JPEG: fall back to cv2, JPEG header parsing is fiddly
            img = cv2.imread(path)
            h, w = img.shape[:2]
            return w, h
        if head[:8] == b"\x89PNG\r\n\x1a\n":
            w, h = struct.unpack(">II", head[16:24])
            return w, h
    except Exception:
        pass
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    h, w = img.shape[:2]
    return w, h


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class BaseBallDataset:
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> ImageSample:
        return self.samples[idx]

    def stats(self) -> dict:
        by_sport = {}
        total_boxes = 0
        for s in self.samples:
            by_sport.setdefault(s.sport, {"images": 0, "boxes": 0})
            by_sport[s.sport]["images"] += 1
            by_sport[s.sport]["boxes"] += len(s.gt_boxes_xywh)
            total_boxes += len(s.gt_boxes_xywh)
        return {
            "total_images": len(self.samples),
            "total_boxes": total_boxes,
            "by_sport": by_sport,
        }


# ---------------------------------------------------------------------------
# Format 1: client / Nikon
# ---------------------------------------------------------------------------

class BallDataset(BaseBallDataset):
    """
    Walks a root directory of the form:
        root/<sport_or_other_subdir>/imgs/*.{jpg,png,...}
        root/<sport_or_other_subdir>/labels/<same_stem>.json

    Any subdirectory name is treated as the "sport" label unless it matches
    ignore_dirs (e.g. "other").
    """

    def __init__(self, root: str, sport_filter: Optional[List[str]] = None,
                 ignore_dirs: Optional[List[str]] = None):
        self.root = Path(root)
        self.sport_filter = set(s.lower() for s in sport_filter) if sport_filter else None
        self.ignore_dirs = set(d.lower() for d in (ignore_dirs or ["other"]))
        self.samples: List[ImageSample] = []
        self._scan()

    def _scan(self):
        if not self.root.exists():
            raise FileNotFoundError(f"Dataset root does not exist: {self.root}")

        for subdir in sorted(self.root.iterdir()):
            if not subdir.is_dir():
                continue
            if subdir.name.lower() in self.ignore_dirs:
                continue

            imgs_dir = subdir / "imgs"
            labels_dir = subdir / "labels"
            if not imgs_dir.exists() or not labels_dir.exists():
                continue

            sport = _infer_sport_from_name(subdir.name)
            if self.sport_filter and sport not in self.sport_filter and sport != "unknown":
                continue
            if self.sport_filter and sport == "unknown":
                if "unknown" not in self.sport_filter and subdir.name.lower() not in self.sport_filter:
                    continue

            for img_path in sorted(imgs_dir.iterdir()):
                if img_path.suffix.lower() not in IMG_EXTS:
                    continue
                label_path = labels_dir / f"{img_path.stem}.json"
                if not label_path.exists():
                    continue

                gt_boxes = self._load_label(label_path)
                self.samples.append(ImageSample(
                    image_path=str(img_path),
                    sport=sport,
                    gt_boxes_xywh=gt_boxes,
                    image_id=f"{subdir.name}/{img_path.stem}",
                ))

    @staticmethod
    def _load_label(label_path: Path) -> np.ndarray:
        with open(label_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        boxes = []
        ball_entries = data.get("data", {}).get("ball", [])
        for entry in ball_entries:
            entire = entry.get("entire")
            if entire is None:
                continue
            rect = entire.get("rect")
            if rect is None or len(rect) != 4:
                continue
            x, y, w, h = rect
            if w <= 0 or h <= 0:
                continue
            boxes.append([float(x), float(y), float(w), float(h)])

        if len(boxes) == 0:
            return np.zeros((0, 4), dtype=np.float32)
        return np.array(boxes, dtype=np.float32)


# ---------------------------------------------------------------------------
# Format 2: COCO
# ---------------------------------------------------------------------------

class CocoBallDataset(BaseBallDataset):
    """
    Reads a single COCO-format instances json + an image directory.

    Sport is resolved in this order:
        1. images[i]["sport"] custom field, if present
        2. parent directory name of file_name (e.g. "basketball/img001.jpg")
        3. "unknown"

    Category filtering: if the json has multiple categories, only annotations
    whose category name contains "ball" (case-insensitive) are kept, unless
    `category_name` is given explicitly. If there is exactly one category,
    it is used regardless of name.
    """

    def __init__(self, ann_file: str, image_dir: str,
                 category_name: Optional[str] = None,
                 sport_filter: Optional[List[str]] = None):
        self.ann_file = Path(ann_file)
        self.image_dir = Path(image_dir)
        self.sport_filter = set(s.lower() for s in sport_filter) if sport_filter else None
        self.samples: List[ImageSample] = []
        self._load(category_name)

    def _load(self, category_name: Optional[str]):
        with open(self.ann_file, "r", encoding="utf-8") as f:
            coco = json.load(f)

        categories = coco.get("categories", [])
        if category_name:
            keep_cat_ids = {c["id"] for c in categories if c["name"].lower() == category_name.lower()}
        elif len(categories) == 1:
            keep_cat_ids = {categories[0]["id"]}
        else:
            keep_cat_ids = {c["id"] for c in categories if "ball" in c["name"].lower()}
            if not keep_cat_ids:
                keep_cat_ids = {c["id"] for c in categories}  # fallback: keep everything

        images_by_id: Dict[int, dict] = {img["id"]: img for img in coco.get("images", [])}
        boxes_by_image: Dict[int, list] = {img_id: [] for img_id in images_by_id}

        for ann in coco.get("annotations", []):
            if ann.get("category_id") not in keep_cat_ids:
                continue
            img_id = ann["image_id"]
            if img_id not in boxes_by_image:
                continue
            x, y, w, h = ann["bbox"]  # COCO bbox is already top-left [x, y, w, h]
            if w <= 0 or h <= 0:
                continue
            boxes_by_image[img_id].append([float(x), float(y), float(w), float(h)])

        for img_id, img_info in sorted(images_by_id.items()):
            file_name = img_info["file_name"]
            sport = img_info.get("sport")
            if not sport:
                parent = Path(file_name).parent.name
                sport = _infer_sport_from_name(parent) if parent else "unknown"
            sport = sport.lower()

            if self.sport_filter and sport not in self.sport_filter and sport != "unknown":
                continue

            image_path = self.image_dir / file_name
            if not image_path.exists():
                # try matching by basename only, in case file_name includes a
                # subdir that doesn't match the actual on-disk layout
                candidates = list(self.image_dir.rglob(Path(file_name).name))
                if not candidates:
                    continue
                image_path = candidates[0]

            boxes = boxes_by_image.get(img_id, [])
            gt_boxes = np.array(boxes, dtype=np.float32) if boxes else np.zeros((0, 4), dtype=np.float32)

            self.samples.append(ImageSample(
                image_path=str(image_path),
                sport=sport,
                gt_boxes_xywh=gt_boxes,
                image_id=str(img_id),
            ))


# ---------------------------------------------------------------------------
# Format 3: YOLO
# ---------------------------------------------------------------------------

class YoloBallDataset(BaseBallDataset):
    """
    Reads YOLO-format data:
        images_dir/*.jpg
        labels_dir/<same_stem>.txt   -- lines: "<class_id> <cx> <cy> <w> <h>"
                                         all normalized to [0, 1]

    If images_dir has sport subfolders (images_dir/basketball/xxx.jpg), sport
    is taken from the immediate parent folder name; otherwise "unknown"
    (or from `sport_override` if given, useful for single-sport YOLO exports).

    ball_class_id: which class index in the YOLO labels corresponds to "ball".
                   Defaults to 0 (single-class ball datasets are the norm here,
                   per client spec: "class is always ball").
    """

    def __init__(self, images_dir: str, labels_dir: str,
                 ball_class_id: int = 0,
                 sport_override: Optional[str] = None,
                 sport_filter: Optional[List[str]] = None):
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)
        self.ball_class_id = ball_class_id
        self.sport_override = sport_override.lower() if sport_override else None
        self.sport_filter = set(s.lower() for s in sport_filter) if sport_filter else None
        self.samples: List[ImageSample] = []
        self._scan()

    def _resolve_sport(self, img_path: Path) -> str:
        if self.sport_override:
            return self.sport_override
        parent = img_path.parent.name
        return _infer_sport_from_name(parent)

    def _scan(self):
        if not self.images_dir.exists():
            raise FileNotFoundError(f"YOLO images dir does not exist: {self.images_dir}")
        if not self.labels_dir.exists():
            raise FileNotFoundError(f"YOLO labels dir does not exist: {self.labels_dir}")

        img_paths = [p for p in self.images_dir.rglob("*") if p.suffix.lower() in IMG_EXTS]

        for img_path in sorted(img_paths):
            sport = self._resolve_sport(img_path)
            if self.sport_filter and sport not in self.sport_filter and sport != "unknown":
                continue

            rel = img_path.relative_to(self.images_dir).with_suffix(".txt")
            label_path = self.labels_dir / rel
            if not label_path.exists():
                # also try flat layout: labels_dir/<stem>.txt regardless of subdirs
                label_path = self.labels_dir / f"{img_path.stem}.txt"
                if not label_path.exists():
                    continue

            img_w, img_h = _read_image_size_fast(str(img_path))
            gt_boxes = self._load_label(label_path, img_w, img_h)

            self.samples.append(ImageSample(
                image_path=str(img_path),
                sport=sport,
                gt_boxes_xywh=gt_boxes,
                image_id=str(img_path.relative_to(self.images_dir).with_suffix("")),
            ))

    def _load_label(self, label_path: Path, img_w: int, img_h: int) -> np.ndarray:
        boxes = []
        with open(label_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                cls_id = int(float(parts[0]))
                if cls_id != self.ball_class_id:
                    continue
                cx, cy, w, h = [float(v) for v in parts[1:5]]
                # denormalize
                cx_px, cy_px = cx * img_w, cy * img_h
                w_px, h_px = w * img_w, h * img_h
                if w_px <= 0 or h_px <= 0:
                    continue
                x = cx_px - w_px / 2.0
                y = cy_px - h_px / 2.0
                boxes.append([x, y, w_px, h_px])

        if len(boxes) == 0:
            return np.zeros((0, 4), dtype=np.float32)
        return np.array(boxes, dtype=np.float32)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_dataset(format: str, **kwargs) -> BaseBallDataset:
    """
    format: "client" | "coco" | "yolo"
    kwargs are forwarded to the matching dataset class constructor:
        client -> BallDataset(root, sport_filter=None, ignore_dirs=None)
        coco   -> CocoBallDataset(ann_file, image_dir, category_name=None, sport_filter=None)
        yolo   -> YoloBallDataset(images_dir, labels_dir, ball_class_id=0,
                                   sport_override=None, sport_filter=None)
    """
    format = format.lower()
    if format == "client":
        return BallDataset(**kwargs)
    elif format == "coco":
        return CocoBallDataset(**kwargs)
    elif format == "yolo":
        return YoloBallDataset(**kwargs)
    else:
        raise ValueError(f"Unknown annotation format: {format!r}. Use 'client', 'coco', or 'yolo'.")


def load_image_rgb(path: str) -> np.ndarray:
    """Load image as RGB uint8 HWC array."""
    img_bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

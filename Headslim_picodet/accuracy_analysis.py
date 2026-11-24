"""Detailed accuracy analysis of ONNX ball detectors (Ohata-san's request, 11/18):
where each model misses, broken down by sport, ball size and image corruptions.

Runs any number of models side by side on the SAME eval set, e.g. the fast and the
accurate candidate:

    python accuracy_analysis.py --data dataset_eval.yaml \
        --model "Exp-2.6=ONNX/nanodet_resnet_20261007.onnx:nanodetplus" \
        --model "Exp-2.1.1=ONNX/picodet_leakyrelu_2026_10_13.onnx:picodet" \
        --out analysis_20261124

--model NAME=PATH:PREPROC  PREPROC is a picodet_eval_onnx.py preset (picodet | nanodetplus);
                           use the one that gives your reported recall for that model.
--group-by                 how the sport is found for each image:
    name    a ball/sport name found ANYWHERE in the file name, e.g. "img_tennis_0042.jpg" -> tennis
            (default; names from --sports, longest match wins; unmatched files are listed)
    dir     sub-folder in the COCO file_name, e.g. "football/img_001.jpg" -> football
    prefix  leading letters of the file name, e.g. "cricket_0042.jpg" -> cricket
    csv     a CSV with columns file_name,sport (--groups-csv)
    none    no sport breakdown

Metrics (all at IoU=0.4, the same matching code as picodet_eval.py / picodet_eval_onnx.py):
  * Each model's operating threshold is taken from its CLEAN recall@FPPI=0.05 -- one
    threshold, as it would be deployed. Every slice and every corruption is then scored at
    that same threshold: recall (TP/GT) and FP per image.
  * Corruptions additionally report recall@FPPI=0.05 re-computed on the corrupted set (what
    the model could reach if the threshold were re-tuned for that condition).
  * Corruptions are applied to the image region of the 320x480 letterboxed canvas, so their
    strength is the same for every source resolution. Noise is seeded per image.

Outputs in --out: summary.md (all tables), one CSV per table, and PNG charts.
"""
import argparse
import contextlib
import csv
import io
import json
import os
import re
from collections import OrderedDict, defaultdict

import cv2
import numpy as np
import onnxruntime as ort

from picodet_eval import greedy_match, load_ground_truth, recall_at_fppi
from picodet_eval_onnx import PREPROC_PRESETS, decode_onnx_output
from picodet_letterbox import letterbox_np

IOU = 0.4
FPPI = 0.05
SIZE_BINS = [(0, 8), (8, 16), (16, 24), (24, 32), (32, 64), (64, 1e9)]   # same as ball_size_report.py
EDGE_PX = 10        # a ball within this many ORIGINAL-image pixels of the border counts as "at edge"

# ------------------------------------------------------------------ corruptions (RGB uint8 in/out)
def _gauss_blur(im, s):
    return cv2.GaussianBlur(im, (0, 0), sigmaX=s)


def _motion_blur(im, k):          # horizontal streak, like a fast-moving ball / panning camera
    ker = np.zeros((k, k), np.float32)
    ker[k // 2, :] = 1.0 / k
    return cv2.filter2D(im, -1, ker)


def _noise(im, std, seed):
    rng = np.random.default_rng(seed)
    return np.clip(im.astype(np.float32) + rng.normal(0, std, im.shape), 0, 255).astype(np.uint8)


def _brightness(im, f):
    return np.clip(im.astype(np.float32) * f, 0, 255).astype(np.uint8)


def _contrast(im, f):
    m = im.reshape(-1, 3).mean(0)
    return np.clip((im.astype(np.float32) - m) * f + m, 0, 255).astype(np.uint8)


def _hue(im, deg):
    hsv = cv2.cvtColor(im, cv2.COLOR_RGB2HSV)
    hsv[..., 0] = ((hsv[..., 0].astype(np.int32) + int(round(deg / 2))) % 180).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


def _saturation(im, f):
    hsv = cv2.cvtColor(im, cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[..., 1] = np.clip(hsv[..., 1] * f, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)


def _gray(im, _):
    g = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
    return np.stack([g] * 3, -1)


# key -> (title, fn(img, seed, value), default levels, breaking-point sweep mild -> severe)
FAMILIES = OrderedDict([
    ("gaussian_blur", ("Blur (Gaussian, sigma px)", lambda im, _, v: _gauss_blur(im, v), [1, 2, 3],
                       [0.5, 1, 1.5, 2, 2.5, 3, 4, 5, 6])),
    ("motion_blur", ("Blur (motion, length px)", lambda im, _, v: _motion_blur(im, int(v)), [5, 9, 15],
                     [3, 5, 7, 9, 13, 17, 23, 31, 45])),
    ("noise", ("Noise (Gaussian, std)", lambda im, seed, v: _noise(im, v, seed), [5, 10, 20],
               [5, 10, 15, 20, 30, 40, 50, 60])),
    ("brightness", ("Brightness (x)", lambda im, _, v: _brightness(im, v), [0.5, 0.75, 1.25, 1.5], None)),
    ("contrast", ("Contrast (x)", lambda im, _, v: _contrast(im, v), [0.5, 1.5], None)),
    ("hue", ("Color: hue shift (deg)", lambda im, _, v: _hue(im, v), [-20, -10, 10, 20],
             [10, 20, 30, 45, 60, 90, 120, 180])),
    ("saturation", ("Color: saturation (x)", lambda im, _, v: _saturation(im, v), [0.0, 0.5, 1.5], None)),
])
# two-sided families get one sweep per direction
SPLIT_SWEEPS = OrderedDict([
    ("darker", ("brightness", [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2])),
    ("brighter", ("brightness", [1.1, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0])),
    ("lower contrast", ("contrast", [0.9, 0.75, 0.6, 0.5, 0.4, 0.3, 0.2])),
    ("higher contrast", ("contrast", [1.25, 1.5, 2.0, 2.5, 3.0])),
    ("less saturated", ("saturation", [0.75, 0.5, 0.25, 0.0])),
    ("more saturated", ("saturation", [1.25, 1.5, 2.0, 2.5, 3.0])),
])


def build_corruptions(levels_file=None):
    """family title -> [(label, fn(img, seed))]. Levels from levels.json (measure_conditions.py) or defaults."""
    given = {}
    if levels_file:
        with open(levels_file) as f:
            given = {k: v for k, v in json.load(f).items() if not k.startswith("_")}
        unknown = set(given) - set(FAMILIES)
        if unknown:
            raise SystemExit(f"unknown corruption(s) in {levels_file}: {sorted(unknown)}; known: {list(FAMILIES)}")
    out = OrderedDict()
    for key, (title, fn, default, _) in FAMILIES.items():
        items = []
        for x in given.get(key, default):
            v, note = (x["value"], x.get("note", "")) if isinstance(x, dict) else (x, "")
            label = f"{v:g}" + (f" ({note})" if note else "")
            items.append((label, (lambda im, seed, v=v, fn=fn: fn(im, seed, v))))
        out[title] = items
    return out


def sweeps():
    """[(name, title, [(value, fn)])] ordered mild -> severe, for the breaking-point search."""
    out = []
    for key, (title, fn, _, sweep) in FAMILIES.items():
        if sweep is not None:
            out.append((title, [(v, (lambda im, seed, v=v, fn=fn: fn(im, seed, v))) for v in sweep]))
    for name, (key, vals) in SPLIT_SWEEPS.items():
        title, fn = FAMILIES[key][0], FAMILIES[key][1]
        out.append((f"{title.split(' (')[0].replace('Color: ', '').capitalize()}: {name} (x)", [(v, (lambda im, seed, v=v, fn=fn: fn(im, seed, v))) for v in vals]))
    return out


# ------------------------------------------------------------------ inference
def _priors(imgsz, strides, offset):
    """anchor centres, level by level, row-major -> (N, 2) x,y and (N,) stride"""
    H, W = imgsz
    xy, st = [], []
    for s in strides:
        h, w = int(np.ceil(H / s)), int(np.ceil(W / s))
        ys, xs = np.meshgrid((np.arange(h) + offset) * s, (np.arange(w) + offset) * s, indexing="ij")
        xy.append(np.stack([xs.ravel(), ys.ravel()], 1)); st.append(np.full(h * w, s, np.float32))
    return np.concatenate(xy).astype(np.float32), np.concatenate(st)


class OnnxModel:
    """Accepts these single-output layouts (detected on the first image and printed):
      decoded      (N, 4+nc)  [x1,y1,x2,y2 canvas px | scores]   (our export_onnx.py; also (4+nc, N))
      decoded-norm same, boxes normalised to 0..1                (scaled to the canvas)
      nanodet-raw  (N, nc+4*(reg_max+1)) [sigmoid scores | box distribution]  (stock NanoDet-Plus export;
                   decoded here: softmax-integral x stride from the anchor centre)
    plus the multi-output --head-style raw PicoDet export."""

    def __init__(self, path, preproc, imgsz, nc=1, reg_max=7, nano_strides=None, nano_offset=0.0, name=""):
        self.cfg = PREPROC_PRESETS[preproc]
        self.mean = np.array(self.cfg["mean"], np.float32)
        self.std = np.array(self.cfg["std"], np.float32)
        self.sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        self.inp = self.sess.get_inputs()[0].name
        self.imgsz, self.nc, self.reg_max = imgsz, nc, reg_max
        self.nano_strides, self.nano_offset, self.name = nano_strides, nano_offset, name
        self.layout = None

    def _detect(self, raw):
        C_dec, C_raw = 4 + self.nc, self.nc + 4 * (self.reg_max + 1)
        if raw.shape[-1] not in (C_dec, C_raw) and raw.shape[0] in (C_dec, C_raw):
            self.transpose = True
            raw = raw.T
        else:
            self.transpose = False
        C = raw.shape[-1]
        if C == C_raw:
            N = raw.shape[0]
            cands = [self.nano_strides] if self.nano_strides else [(8, 16, 32, 64), (8, 16, 32), (8, 16), (16, 32, 64)]
            for strides in cands:
                if len(_priors(self.imgsz, strides, 0)[1]) == N:
                    self.strides = strides
                    self.priors = _priors(self.imgsz, strides, self.nano_offset)
                    self.layout = "nanodet-raw"
                    break
            else:
                raise SystemExit(f"[{self.name}] output has {N} anchors x {C} channels (NanoDet-Plus raw layout) "
                                 f"but no stride set matches {self.imgsz}; pass --nanodet-strides")
        elif C == C_dec:
            self.layout = "decoded-norm" if np.abs(raw[:, :4]).max() <= 1.5 else "decoded"
        else:
            raise SystemExit(f"[{self.name}] unrecognised output shape {raw.shape} for nc={self.nc}: expected "
                             f"(N,{C_dec}) decoded or (N,{C_raw}) NanoDet-Plus raw")
        extra = f", strides {self.strides}, centre offset {self.nano_offset}" if self.layout == "nanodet-raw" else ""
        print(f"[{self.name}] output layout: {self.layout}{' (transposed)' if self.transpose else ''}{extra}")

    def _to_decoded(self, raw):
        if self.transpose:
            raw = raw.T
        if self.layout == "decoded":
            return raw
        if self.layout == "decoded-norm":
            out = raw.copy()
            out[:, [0, 2]] *= self.imgsz[1]
            out[:, [1, 3]] *= self.imgsz[0]
            return out
        nb = self.reg_max + 1                                   # nanodet-raw
        scores, dist = raw[:, :self.nc], raw[:, self.nc:].reshape(-1, 4, nb)
        e = np.exp(dist - dist.max(-1, keepdims=True))
        d = (e / e.sum(-1, keepdims=True) * np.arange(nb, dtype=np.float32)).sum(-1) * self.priors[1][:, None]
        (cx, cy) = self.priors[0][:, 0], self.priors[0][:, 1]
        boxes = np.stack([cx - d[:, 0], cy - d[:, 1], cx + d[:, 2], cy + d[:, 3]], 1)
        return np.concatenate([boxes, scores], 1).astype(np.float32)

    def __call__(self, canvas_rgb, ratio, orig_hw):
        arr = canvas_rgb[:, :, ::-1] if self.cfg["bgr"] else canvas_rgb
        x = ((arr.astype(np.float32) - self.mean) / self.std).transpose(2, 0, 1)[None]
        outs = self.sess.run(None, {self.inp: np.ascontiguousarray(x, dtype=np.float32)})
        if len(outs) == 1:
            raw = outs[0][0]
            if self.layout is None:
                self._detect(raw)
            raw = self._to_decoded(raw)
        else:                                                   # export_onnx.py --head-style raw
            from picodet_raw_decode import decode_levels
            strides = [round(self.imgsz[0] / o.shape[2]) for o in outs]
            raw = decode_levels(outs, strides, outs[0].shape[1] - 32)
        d = decode_onnx_output(raw, 0.001, 0.45, 300)
        if len(d):
            h, w = orig_hw
            d[:, :4] /= ratio
            d[:, [0, 2]] = d[:, [0, 2]].clip(0, w)
            d[:, [1, 3]] = d[:, [1, 3]].clip(0, h)
        return d


def load_canvases(images, imgsz):
    """Letterbox once, exactly as picodet_eval_onnx.py does. Returns id -> (canvas, ratio, (h,w), (ch,cw))."""
    from libreyolo.utils.image_loader import ImageLoader
    out = {}
    for img_id, path in images.items():
        img = np.array(ImageLoader.load(path, color_format="auto"))          # RGB
        canvas, ratio, _, _ = letterbox_np(img, imgsz)
        h, w = img.shape[:2]
        out[img_id] = (canvas, ratio, (h, w), (int(round(h * ratio)), int(round(w * ratio))))
    return out


def predict(model, canv, corrupt=None):
    preds = {}
    for img_id, (canvas, ratio, hw, (ch, cw)) in canv.items():
        if corrupt is not None:
            canvas = canvas.copy()
            canvas[:ch, :cw] = corrupt(np.ascontiguousarray(canvas[:ch, :cw]), int(img_id) % (2 ** 31))
        preds[img_id] = model(canvas, ratio, hw)
    return preds


# ------------------------------------------------------------------ metrics
def fppi_recall(preds, gts):
    with contextlib.redirect_stdout(io.StringIO()):
        r, thr = recall_at_fppi(preds, gts, IOU, FPPI)
    return float(r), (float(thr) if thr is not None else 1.0)


def match_at(preds, gts, thr):
    """Per image at a fixed threshold: (matched-GT flags, number of FP)."""
    res = {}
    for img_id, gt in gts.items():
        d = preds.get(img_id, np.zeros((0, 6), np.float32))
        d = d[d[:, 4] >= thr]
        d = d[np.argsort(-d[:, 4])]
        is_tp, matched = greedy_match(d, gt, IOU)
        res[img_id] = (matched, int((~is_tp).sum()))
    return res


def slice_table(matches, gts, ball_key, image_key=None):
    """ball_key(img_id, j) -> slice of ball j; image_key(img_id) -> slice of the image (for FP/img)."""
    gt_n, tp_n = defaultdict(int), defaultdict(int)
    fp_n, im_n = defaultdict(int), defaultdict(int)
    for img_id, (matched, fp) in matches.items():
        for j in range(len(gts[img_id])):
            k = ball_key(img_id, j)
            gt_n[k] += 1
            tp_n[k] += int(matched[j])
        if image_key is not None:
            k = image_key(img_id)
            fp_n[k] += fp
            im_n[k] += 1
    keys = list(gt_n) + [k for k in im_n if k not in gt_n]
    return {k: dict(gt=gt_n[k], tp=tp_n[k], recall=tp_n[k] / gt_n[k] if gt_n[k] else float("nan"),
                    images=im_n.get(k, 0), fp_per_img=fp_n[k] / im_n[k] if im_n.get(k) else float("nan"))
            for k in keys}


# ------------------------------------------------------------------ grouping
DEFAULT_SPORTS = ("football,soccer,futsal,basketball,volleyball,tennis,tabletennis,table_tennis,pingpong,"
                  "cricket,baseball,softball,rugby,golf,handball,hockey,badminton,shuttle,netball,"
                  "dodgeball,kickball,squash,water_polo,waterpolo,polo,bowling,billiards,pool")


def sport_of(images, coco_names, mode, csv_path, sports=DEFAULT_SPORTS):
    if mode == "name":
        names = sorted({s.strip().lower() for s in sports.split(",") if s.strip()}, key=len, reverse=True)
        out, unmatched = {}, []
        for i in images:
            base = os.path.basename(coco_names[i].replace("\\", "/")).lower()
            hit = next((n for n in names if n in base), None)        # longest name first: softball before ball
            out[i] = hit or "unknown"
            if hit is None:
                unmatched.append(base)
        if unmatched:
            print(f"{len(unmatched)} file name(s) matched no sport, e.g. {unmatched[:5]} -- "
                  f"add their ball name to --sports")
        return out
    if mode == "none":
        return {i: "all" for i in images}
    if mode == "csv":
        m = {}
        with open(csv_path) as f:
            for row in csv.DictReader(f):
                m[os.path.normpath(row["file_name"])] = row["sport"]
        return {i: m.get(os.path.normpath(coco_names[i]), m.get(os.path.basename(coco_names[i]), "unknown"))
                for i in images}
    out = {}
    for i in images:
        fn = coco_names[i].replace("\\", "/")
        if mode == "dir":
            parts = fn.split("/")
            out[i] = parts[-2] if len(parts) > 1 else "unknown"
        else:                                                   # prefix
            mm = re.match(r"([A-Za-z]+)", os.path.basename(fn))
            out[i] = mm.group(1).lower() if mm else "unknown"
    return out


def size_bin(s):
    for lo, hi in SIZE_BINS:
        if lo <= s < hi:
            return f"{lo:.0f}-{hi:.0f}px" if hi < 1e9 else f"{lo:.0f}+px"
    return "?"


# ------------------------------------------------------------------ report
def write_csv(path, header, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def md_table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def pct(x):
    return "-" if x != x else f"{100 * x:.1f}%"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--split", default="val")
    ap.add_argument("--model", action="append", required=True, metavar="NAME=PATH:PREPROC")
    ap.add_argument("--imgsz", type=int, nargs=2, default=[320, 480], metavar=("H", "W"))
    ap.add_argument("--group-by", default="name", choices=["name", "dir", "prefix", "csv", "none"])
    ap.add_argument("--sports", default=DEFAULT_SPORTS,
                    help="comma-separated ball/sport names to look for in file names (--group-by name)")
    ap.add_argument("--groups-csv", default=None)
    ap.add_argument("--nc", type=int, default=1, help="number of classes in the models")
    ap.add_argument("--nanodet-strides", type=int, nargs="+", default=None,
                    help="strides of a NanoDet-Plus raw output (auto-detected from the anchor count if omitted)")
    ap.add_argument("--nanodet-center-offset", type=float, default=0.0,
                    help="anchor centre offset for NanoDet-Plus raw output: 0.0 = NanoDet-Plus (x*stride), "
                         "0.5 = PicoDet ((x+0.5)*stride)")
    ap.add_argument("--no-corruptions", action="store_true", help="skip the corruption sweep (faster)")
    ap.add_argument("--levels", default=None, help="levels.json from measure_conditions.py (default: built-in levels)")
    ap.add_argument("--breaking-point", action="store_true",
                    help="also sweep each corruption from mild to severe and report where recall@FPPI "
                         "falls below --target-recall (slower)")
    ap.add_argument("--target-recall", type=float, default=0.70)
    ap.add_argument("--out", default="accuracy_analysis")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    imgsz = tuple(a.imgsz)

    models = OrderedDict()
    for spec in a.model:
        name, rest = spec.split("=", 1)
        path, _, preproc = rest.rpartition(":")
        if not path:
            path, preproc = rest, "picodet"
        models[name] = OnnxModel(path, preproc, imgsz, nc=a.nc,
                                 nano_strides=tuple(a.nanodet_strides) if a.nanodet_strides else None,
                                 nano_offset=a.nanodet_center_offset, name=name)
        print(f"model {name}: {path} (preproc {preproc})")

    images, gts, _ = load_ground_truth(a.data, a.split)
    from pycocotools.coco import COCO
    from libreyolo.data import load_data_config
    with contextlib.redirect_stdout(io.StringIO()):
        coco = COCO(load_data_config(a.data)[f"{a.split}_annotation_file"])
    coco_names = {i: coco.loadImgs(i)[0]["file_name"] for i in images}
    sport = sport_of(images, coco_names, a.group_by, a.groups_csv, a.sports)
    n_sports = len(set(sport.values()))
    print(f"{len(images)} images, {sum(len(g) for g in gts.values())} balls, {n_sports} sport group(s): "
          + ", ".join(f"{k}={v}" for k, v in sorted(_count(sport.values()).items())))
    if n_sports > 25:
        print("WARNING: many groups -- --group-by is probably not matching how your sport is stored")

    canv = load_canvases(images, imgsz)
    csize = {i: (np.sqrt((g[:, 2] - g[:, 0]) * (g[:, 3] - g[:, 1])) * canv[i][1]) if len(g) else np.zeros(0)
             for i, g in gts.items()}
    at_edge = {i: ((g[:, 0] < EDGE_PX) | (g[:, 1] < EDGE_PX) | (g[:, 2] > canv[i][2][1] - EDGE_PX)
                   | (g[:, 3] > canv[i][2][0] - EDGE_PX)) if len(g) else np.zeros(0, bool)
               for i, g in gts.items()}
    n_balls = {i: len(g) for i, g in gts.items()}

    md = [f"# Accuracy analysis\n\nEval set: {len(images)} images, {sum(n_balls.values())} balls. "
          f"IoU={IOU}. Each model is scored at its own clean operating threshold (recall@FPPI={FPPI}).\n"]
    clean, thr_of, match_of = {}, {}, {}
    for name, m in models.items():
        print(f"[{name}] clean ...")
        p = predict(m, canv)
        r, thr = fppi_recall(p, gts)
        clean[name], thr_of[name], match_of[name] = r, thr, match_at(p, gts, thr)
    md.append("## Overall (clean)\n\n" + md_table(
        ["Model", f"Recall@FPPI={FPPI}", "Operating threshold"],
        [[n, pct(clean[n]), f"{thr_of[n]:.3f}"] for n in models]))

    # --- slices: sport, size, edge, balls per image
    slices = [
        ("Sport", lambda i, j: sport[i], lambda i: sport[i], None),
        ("Ball size (px on 320x480 canvas)", lambda i, j: size_bin(csize[i][j]), None,
         [size_bin((lo + min(hi, 1e4)) / 2) for lo, hi in SIZE_BINS]),
        ("Ball position", lambda i, j: "at image edge" if at_edge[i][j] else "inside", None,
         ["inside", "at image edge"]),
        ("Balls in image", lambda i, j: "1" if n_balls[i] == 1 else "2+",
         lambda i: "0" if n_balls[i] == 0 else ("1" if n_balls[i] == 1 else "2+"), ["0", "1", "2+"]),
    ]
    charts = []
    for title, bk, ik, order in slices:
        tabs = {n: slice_table(match_of[n], gts, bk, ik) for n in models}
        keys = order or sorted(set().union(*[t.keys() for t in tabs.values()]))
        keys = [k for k in keys if any(k in t for t in tabs.values())]
        header = [title, "Balls"] + [f"{n} recall" for n in models] + (
            ["Images"] + [f"{n} FP/img" for n in models] if ik else [])
        rows = []
        for k in keys:
            t0 = next(t[k] for t in tabs.values() if k in t)
            row = [k, t0["gt"]] + [pct(tabs[n].get(k, {}).get("recall", float("nan"))) for n in models]
            if ik:
                row += [t0["images"]] + [f"{tabs[n].get(k, {}).get('fp_per_img', float('nan')):.3f}" for n in models]
            rows.append(row)
        md.append(f"## By {title.lower()}\n\n" + md_table(header, rows))
        write_csv(os.path.join(a.out, f"by_{_slug(title)}.csv"), header, rows)
        charts.append((title, keys, {n: [tabs[n].get(k, {}).get("recall", np.nan) for k in keys] for n in models}))

    # --- corruptions
    corr_rows = []
    CORRUPTIONS = build_corruptions(a.levels)
    md.append("Corruption levels: " + (f"measured from real images ({a.levels})" if a.levels
                                       else "built-in defaults (run measure_conditions.py for data-driven levels)"))
    if not a.no_corruptions:
        for fam, levels in CORRUPTIONS.items():
            for lab, fn in levels:
                row = [fam, lab]
                for name, m in models.items():
                    print(f"[{name}] {fam} {lab} ...")
                    p = predict(m, canv, fn)
                    mt = match_at(p, gts, thr_of[name])
                    tp = sum(int(v[0].sum()) for v in mt.values())
                    fp = sum(v[1] for v in mt.values())
                    r_retuned, _ = fppi_recall(p, gts)
                    row += [tp / max(1, sum(n_balls.values())), fp / len(images), r_retuned]
                corr_rows.append(row)
        header = ["Corruption", "Level"]
        for n in models:
            header += [f"{n} recall@clean thr", f"{n} FP/img@clean thr", f"{n} recall@FPPI={FPPI} re-tuned"]
        nice = [[r[0], r[1]] + [pct(v) if (k % 3) != 1 else f"{v:.3f}" for k, v in enumerate(r[2:])] for r in corr_rows]
        clean_row = ["(clean)", "-"]
        for n in models:
            mt = match_of[n]
            clean_row += [pct(sum(int(v[0].sum()) for v in mt.values()) / max(1, sum(n_balls.values()))),
                          f"{sum(v[1] for v in mt.values()) / len(images):.3f}", pct(clean[n])]
        md.append("## Robustness to image corruptions\n\nAt the clean operating threshold (as deployed), "
                  "and with the threshold re-tuned to FPPI=0.05 on the corrupted set.\n\n"
                  + md_table(header, [clean_row] + nice))
        write_csv(os.path.join(a.out, "corruptions.csv"), header, [clean_row] + nice)

    bp_curves = {}
    if a.breaking_point:
        rows = []
        for title, steps in sweeps():
            row = [title]
            for name, m in models.items():
                xs, ys = [None], [clean[name]]
                for v, fn in steps:
                    print(f"[{name}] breaking point: {title} {v} ...")
                    r, _ = fppi_recall(predict(m, canv, fn), gts)
                    xs.append(v); ys.append(r)
                    if r < a.target_recall - 0.10:          # well past the target: stop early
                        break
                bp_curves.setdefault(title, {})[name] = (xs, ys)
                row.append(_crossing(xs, ys, a.target_recall))
            rows.append(row)
        header = ["Corruption"] + [f"{n}: level where recall@FPPI={FPPI} < {a.target_recall:.0%}" for n in models]
        md.append(f"## Breaking point (recall@FPPI={FPPI} re-tuned, target {a.target_recall:.0%})\n\n"
                  "Interpolated level from which recall stays below the target (a single noisy dip that recovers is ignored). "
                  "'> x' = never dropped within the tested range (up to x).\n\n" + md_table(header, rows))
        write_csv(os.path.join(a.out, "breaking_point.csv"), header, rows)

    with open(os.path.join(a.out, "summary.md"), "w") as f:
        f.write("\n\n".join(md) + "\n")
    _plots(a.out, list(models), charts, corr_rows, clean)
    if bp_curves:
        _bp_plot(a.out, list(models), bp_curves, a.target_recall)
    print(f"\nwrote {a.out}/summary.md, CSVs and charts")


def _crossing(xs, ys, target):
    """Level from which recall STAYS below target (robust to a single noisy dip): interpolate
    between the last level still at/above target and the next one."""
    if ys[0] < target:
        return f"below target when clean ({ys[0]:.1%})"
    last_ok = max(k for k in range(len(ys)) if ys[k] >= target)
    if last_ok == len(ys) - 1:
        return f"> {xs[-1]:g}"
    k = last_ok + 1
    if xs[last_ok] is None:
        return f"already at mildest level ({xs[k]:g})"
    f = (ys[last_ok] - target) / max(ys[last_ok] - ys[k], 1e-9)
    return f"~{xs[last_ok] + f * (xs[k] - xs[last_ok]):.2f}"


def _bp_plot(out, names, curves, target):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    titles = list(curves)
    cols = 5
    rows_n = int(np.ceil(len(titles) / cols))
    fig, axes = plt.subplots(rows_n, cols, figsize=(3.3 * cols, 3.0 * rows_n), squeeze=False)
    for ax, t in zip(axes.ravel(), titles):
        for n in names:
            xs, ys = curves[t][n]
            ax.plot(["clean"] + [f"{x:g}" for x in xs[1:]], [100 * y for y in ys], "o-", ms=3, label=n)
        ax.axhline(100 * target, color="#c0392b", ls="--", lw=1)
        ax.set_title(t, fontsize=8); ax.tick_params(labelsize=6); ax.set_ylim(0, 100); ax.grid(alpha=0.3)
    for ax in axes.ravel()[len(titles):]:
        ax.axis("off")
    axes[0, 0].legend(fontsize=7); axes[0, 0].set_ylabel(f"recall@FPPI={FPPI} (%)")
    fig.tight_layout(); fig.savefig(os.path.join(out, "breaking_point.png"), dpi=130); plt.close(fig)


def _count(vals):
    c = defaultdict(int)
    for v in vals:
        c[v] += 1
    return c


def _slug(s):
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")[:40]


def _plots(out, names, charts, corr_rows, clean):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    w = 0.8 / max(1, len(names))
    for title, keys, vals in charts:
        fig, ax = plt.subplots(figsize=(max(5, 1.2 * len(keys) + 2), 3.6))
        x = np.arange(len(keys))
        for i, n in enumerate(names):
            ax.bar(x + (i - (len(names) - 1) / 2) * w, 100 * np.array(vals[n], float), w, label=n)
        ax.set_xticks(x, keys, rotation=20 if len(keys) > 5 else 0)
        ax.set_ylabel("recall (%)"); ax.set_ylim(0, 100); ax.set_title(f"Recall by {title.lower()}")
        ax.grid(axis="y", alpha=0.3); ax.legend()
        fig.tight_layout(); fig.savefig(os.path.join(out, f"by_{_slug(title)}.png"), dpi=130); plt.close(fig)
    if not corr_rows:
        return
    fams = list(OrderedDict.fromkeys(r[0] for r in corr_rows))
    fig, axes = plt.subplots(1, len(fams), figsize=(3.2 * len(fams), 3.4), sharey=True)
    for ax, fam in zip(np.atleast_1d(axes), fams):
        rows = [r for r in corr_rows if r[0] == fam]
        for i, n in enumerate(names):
            ax.plot(["clean"] + [r[1].split(" (")[0] for r in rows], [100 * clean[n]] + [100 * r[2 + 3 * i + 2] for r in rows],
                    "o-", label=n)
        ax.set_title(fam, fontsize=8); ax.grid(alpha=0.3); ax.tick_params(labelsize=7)
    np.atleast_1d(axes)[0].set_ylabel(f"recall@FPPI={FPPI} (%)")
    np.atleast_1d(axes)[0].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(out, "corruptions.png"), dpi=130); plt.close(fig)


if __name__ == "__main__":
    main()

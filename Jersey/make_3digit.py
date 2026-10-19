"""
Build 3-digit jersey images using ONLY the validation set (images + .txt labels).

No fonts, no external digit images. Every glyph that gets pasted is a real digit
cut out of one of your own val images:
  1. locate the number (no coords needed): PaddleOCR if installed, else a classical fallback
  2. segment digit pixels (2-cluster k-means in Lab) -> per-digit masks = the coordinates you lack
  3. estimate colour, angle, height, spacing, blur from the original number
  4. erase the original number (inpaint), then re-lay ALL digits (old + borrowed) in one row
  5. rotate / scale to fit, recolour with local shading, blur to match, composite

Usage:
  python make_3digit.py --images val/images --labels val/labels --out synth3 --per-image 2

Label format written: "1 7 7" (space separated), same stem as the image.
"""
import argparse, math, random
from pathlib import Path
import cv2
import numpy as np

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}


def read_label(p):
    try:
        return [int(t) for t in Path(p).read_text().split()]
    except Exception:
        return None


# ---------------------------------------------------------------- localisation
def detect_paddle(img, n):
    try:
        from paddleocr import PaddleOCR
        if not hasattr(detect_paddle, "ocr"):
            detect_paddle.ocr = PaddleOCR(use_angle_cls=False, lang="en", show_log=False)
        res = detect_paddle.ocr.ocr(img, cls=False)
        best = None
        for line in (res[0] or []):
            pts, (txt, conf) = line
            digits = "".join(c for c in txt if c.isdigit())
            if len(digits) == n:
                x, y, w, h = cv2.boundingRect(np.array(pts, dtype=np.int32))
                if best is None or conf > best[1]:
                    best = ((x, y, w, h), conf)
        return best[0] if best else None
    except Exception:
        return None


def detect_classic(img, n):
    H, W = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    best, best_score = None, -1
    for m in (th, 255 - th):
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k)
        cnt, lab, st, _ = cv2.connectedComponentsWithStats(m)
        comps = [st[i] for i in range(1, cnt)
                 if 0.04 * H < st[i][3] < 0.45 * H and 0.12 < st[i][2] / st[i][3] < 1.3]
        for a in comps:
            grp = [c for c in comps
                   if abs(c[3] - a[3]) < 0.25 * a[3]
                   and abs((c[1] + c[3] / 2) - (a[1] + a[3] / 2)) < 0.5 * a[3]
                   and abs((c[0] + c[2] / 2) - (a[0] + a[2] / 2)) < (n + 0.5) * a[3]]
            if len(grp) != n:
                continue
            x0 = min(c[0] for c in grp); y0 = min(c[1] for c in grp)
            x1 = max(c[0] + c[2] for c in grp); y1 = max(c[1] + c[3] for c in grp)
            pad = max(3, int(0.15 * (y1 - y0)))
            xa, ya, xb, yb = max(0, x0 - pad), max(0, y0 - pad), min(W, x1 + pad), min(H, y1 + pad)
            inner = gray[y0:y1, x0:x1].astype(float).mean()
            ring = np.concatenate([gray[ya:yb, xa:x0].ravel(), gray[ya:yb, x1:xb].ravel(),
                                   gray[ya:y0, xa:xb].ravel(), gray[y1:yb, xa:xb].ravel()])
            if ring.size == 0:
                continue
            score = abs(inner - ring.mean()) * math.sqrt(sum(c[4] for c in grp))
            if score > best_score:
                best, best_score = (xa, ya, xb - xa, yb - ya), score
    return best


# ---------------------------------------------------------------- segmentation
def segment(img, box, n):
    """Return full-size mask + list of per-digit stats (x,y,w,h,area) sorted left->right."""
    H, W = img.shape[:2]
    x, y, w, h = box
    pad = int(0.1 * h)
    xa, ya, xb, yb = max(0, x - pad), max(0, y - pad), min(W, x + w + pad), min(H, y + h + pad)
    crop = img[ya:yb, xa:xb]
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB).astype(np.float32)
    flat = lab.reshape(-1, 3)
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.5)
    _, lbl, cen = cv2.kmeans(flat, 2, None, crit, 3, cv2.KMEANS_PP_CENTERS)
    lbl = lbl.reshape(crop.shape[:2])
    border = np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]])
    bmed = np.median(border, axis=0)
    digit_c = int(np.argmax(np.linalg.norm(cen - bmed, axis=1)))
    m = (lbl == digit_c).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    cnt, cl, st, _ = cv2.connectedComponentsWithStats(m)
    keep = [i for i in range(1, cnt) if st[i][3] >= 0.5 * h]
    keep.sort(key=lambda i: st[i][0])
    if len(keep) != n:
        return None, None
    full = np.zeros((H, W), np.uint8)
    stats = []
    for i in keep:
        comp = (cl == i).astype(np.uint8)
        full[ya:yb, xa:xb] |= comp
        sx, sy, sw, sh, sa = st[i]
        stats.append((sx + xa, sy + ya, sw, sh, sa, comp, (xa, ya)))
    return full, stats


def stroke_ratio(alpha):
    dt = cv2.distanceTransform((alpha > 127).astype(np.uint8), cv2.DIST_L2, 3)
    v = dt[dt > 0]
    return 2 * float(np.percentile(v, 75)) / alpha.shape[0] if v.size else 0.1


def estimate_angle(stats):
    if len(stats) < 2:
        return 0.0
    xs = np.array([s[0] + s[2] / 2 for s in stats])
    ys = np.array([s[1] + s[3] for s in stats])      # bottoms
    a = math.degrees(math.atan(np.polyfit(xs, ys, 1)[0]))
    return float(np.clip(a, -25, 25))


def estimate_blur(img, mask, inp, color, box):
    x, y, w, h = box
    g = lambda im: cv2.Laplacian(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY), cv2.CV_64F)[y:y + h, x:x + w].var()
    target = g(img)
    best, bd = 0.0, 1e18
    for s in [0, .5, .8, 1.1, 1.5, 2, 2.5, 3, 4]:
        a = cv2.GaussianBlur(mask.astype(np.float32), (0, 0), s) if s else mask.astype(np.float32)
        r = inp * (1 - a[..., None]) + np.array(color, np.float32) * a[..., None]
        d = abs(g(r.astype(np.uint8)) - target)
        if d < bd:
            best, bd = s, d
    return best


# ---------------------------------------------------------------- analysis pass
def analyse(img, label, use_paddle):
    n = len(label)
    box = (detect_paddle(img, n) if use_paddle else None) or detect_classic(img, n)
    if box is None:
        return None
    mask, stats = segment(img, box, n)
    if mask is None:
        return None
    glyphs = []
    for s in stats:
        sx, sy, sw, sh, _, comp, (ox, oy) = s
        crop = comp[sy - oy:sy - oy + sh, sx - ox:sx - ox + sw] * 255
        crop = cv2.GaussianBlur(crop, (0, 0), 0.7)
        glyphs.append(crop)
    ux0 = min(s[0] for s in stats); uy0 = min(s[1] for s in stats)
    ux1 = max(s[0] + s[2] for s in stats); uy1 = max(s[1] + s[3] for s in stats)
    gaps = [stats[i + 1][0] - (stats[i][0] + stats[i][2]) for i in range(n - 1)]
    hmed = float(np.median([s[3] for s in stats]))
    return dict(box=box, mask=mask, stats=stats, glyphs=glyphs, label=label,
                angle=estimate_angle(stats), hmed=hmed,
                gap=float(np.median(gaps)) if gaps else 0.15 * hmed,
                center=((ux0 + ux1) / 2, (uy0 + uy1) / 2),
                stroke=float(np.mean([stroke_ratio(g) for g in glyphs])),
                color=np.median(img[mask > 0], axis=0))


# ---------------------------------------------------------------- synthesis
def build_bank(infos):
    bank = {d: [] for d in range(10)}
    for k, inf in infos.items():
        if abs(inf["angle"]) > 8:
            continue
        for d, g in zip(inf["label"], inf["glyphs"]):
            bank[d].append((k, g, stroke_ratio(g)))
    return bank


def resize_h(g, h):
    s = h / g.shape[0]
    return cv2.resize(g, (max(2, int(round(g.shape[1] * s))), int(round(h))), interpolation=cv2.INTER_AREA)


def synth(img, key, inf, bank, rng, max_w_ratio, min_scale, hundreds):
    H, W = img.shape[:2]
    n = len(inf["label"])
    need = 3 - n
    if need <= 0:
        return None
    new_digits = [rng.choice(hundreds) if (i == 0 and need >= 1) else rng.randint(0, 9) for i in range(need)]
    front = True
    new_glyphs = []
    for d in new_digits:
        cands = [c for c in bank[d] if c[0] != key]
        if not cands:
            return None
        cands.sort(key=lambda c: abs(c[2] - inf["stroke"]))
        new_glyphs.append(rng.choice(cands[:3])[1])
    hm = inf["hmed"]
    old = [resize_h(g, hm) for g in inf["glyphs"]]
    new = [resize_h(g, hm) for g in new_glyphs]
    seq = new + old if front else old + new
    labels = new_digits + inf["label"] if front else inf["label"] + new_digits
    gap = max(1, int(inf["gap"]))
    cw = sum(g.shape[1] for g in seq) + gap * (len(seq) - 1)
    ch = int(round(hm))
    canvas = np.zeros((ch, cw), np.uint8)
    x = 0
    for g in seq:
        canvas[:, x:x + g.shape[1]] = np.maximum(canvas[:, x:x + g.shape[1]], g)
        x += g.shape[1] + gap
    # size fit
    s = min(1.0, (max_w_ratio * W) / cw)
    if s < min_scale:
        return None
    # erase original
    k = max(5, int(0.2 * hm)) | 1
    big = cv2.dilate(inf["mask"], np.ones((k, k), np.uint8))
    inp = cv2.inpaint(img, big * 255, 3, cv2.INPAINT_TELEA)
    # shading map from inpainted fabric
    lum = cv2.GaussianBlur(cv2.cvtColor(inp, cv2.COLOR_BGR2GRAY).astype(np.float32), (0, 0), 12)
    ref = np.median(lum[inf["mask"] > 0])
    shade = np.clip(lum / max(ref, 1), 0.7, 1.3)[..., None]
    # rotate + scale canvas, place at original centre
    cx, cy = inf["center"]
    M = cv2.getRotationMatrix2D((cw / 2, ch / 2), -inf["angle"], s)
    M[0, 2] += cx - cw / 2
    M[1, 2] += cy - ch / 2
    alpha = cv2.warpAffine(canvas, M, (W, H), flags=cv2.INTER_LINEAR).astype(np.float32) / 255
    ys, xs = np.where(alpha > 0.5)
    if xs.size == 0 or xs.min() < 2 or ys.min() < 2 or xs.max() > W - 3 or ys.max() > H - 3:
        return None
    sigma = estimate_blur(img, inf["mask"], inp, inf["color"], inf["box"])
    if sigma:
        alpha = cv2.GaussianBlur(alpha, (0, 0), sigma)
    fill = inf["color"].astype(np.float32)[None, None, :] * shade
    out = inp.astype(np.float32) * (1 - alpha[..., None]) + fill * alpha[..., None]
    return np.clip(out, 0, 255).astype(np.uint8), labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-image", type=int, default=1)
    ap.add_argument("--max-width-ratio", type=float, default=0.6, help="max number width / image width")
    ap.add_argument("--min-scale", type=float, default=0.6, help="reject if number must shrink below this")
    ap.add_argument("--hundreds", default="1", help="allowed leading digits, e.g. 1 or 123456789")
    ap.add_argument("--no-paddle", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    out_i = Path(a.out) / "images"; out_l = Path(a.out) / "labels"
    out_i.mkdir(parents=True, exist_ok=True); out_l.mkdir(parents=True, exist_ok=True)

    infos, imgs = {}, {}
    files = sorted(p for p in Path(a.images).iterdir() if p.suffix.lower() in IMG_EXT)
    for p in files:
        lab = read_label(Path(a.labels) / (p.stem + ".txt"))
        img = cv2.imread(str(p))
        if img is None or not lab or len(lab) > 2:
            continue
        inf = analyse(img, lab, not a.no_paddle)
        if inf:
            infos[p.stem], imgs[p.stem] = inf, img
    print(f"analysed ok: {len(infos)}/{len(files)}")
    bank = build_bank(infos)
    print("glyphs per digit:", {d: len(v) for d, v in bank.items()})

    made = 0
    for k, inf in infos.items():
        for j in range(a.per_image):
            r = synth(imgs[k], k, inf, bank, rng, a.max_width_ratio, a.min_scale, [int(c) for c in a.hundreds])
            if r is None:
                continue
            im, labels = r
            cv2.imwrite(str(out_i / f"{k}_3d{j}.jpg"), im)
            (out_l / f"{k}_3d{j}.txt").write_text(" ".join(map(str, labels)))
            made += 1
    print(f"written {made} images to {a.out}")


if __name__ == "__main__":
    main()

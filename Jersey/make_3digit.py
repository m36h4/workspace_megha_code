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
import argparse, math, random, time
from collections import Counter
from pathlib import Path
import cv2
import numpy as np

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}
GPU = False
REASONS = Counter()
LAST = {"r": ""}


def no(reason):
    LAST["r"] = reason
    REASONS[reason] += 1
    return None


def read_label(p):
    try:
        return [int(t) for t in Path(p).read_text().split()]
    except Exception:
        return None


# ---------------------------------------------------------------- localisation
def get_ocr(gpu):
    if not hasattr(get_ocr, "o"):
        from paddleocr import PaddleOCR
        get_ocr.o = PaddleOCR(use_angle_cls=False, lang="en", use_gpu=gpu, show_log=False)
    return get_ocr.o


def paddle_candidates(img, n, gpu):
    """Candidate number boxes, best first: det+rec boxes whose digit count == n, then other text, then det-only."""
    try:
        ocr = get_ocr(gpu)
    except Exception as e:
        if not getattr(paddle_candidates, "warned", False):
            print("PaddleOCR unavailable, using classic detector only:", repr(e)[:150])
            paddle_candidates.warned = True
        return []
    out = []
    try:
        res = ocr.ocr(img, cls=False)
        for line in (res[0] or []):
            pts, (txt, conf) = line
            d = "".join(c for c in txt if c.isdigit())
            box = cv2.boundingRect(np.array(pts, dtype=np.int32))
            out.append((0 if len(d) == n else 1, -conf, box))
        det = ocr.ocr(img, rec=False, cls=False)
        for pts in (det[0] or []):
            out.append((2, 0, cv2.boundingRect(np.array(pts, dtype=np.int32))))
    except Exception:
        pass
    out.sort(key=lambda t: t[:2])
    return [t[2] for t in out]


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
        comps = sorted(comps, key=lambda c: -c[4])[:60]
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
        return None, "digit count mismatch"
    hs = [st[i][3] for i in keep]
    if max(hs) > 1.35 * min(hs) or h > 0.5 * H:
        return None, "inconsistent digit heights"
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
    x0, y0 = max(0, x - 10), max(0, y - 10)
    x1, y1 = min(img.shape[1], x + w + 10), min(img.shape[0], y + h + 10)
    g = lambda im: cv2.Laplacian(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
    img_c, inp_c, mask_c = img[y0:y1, x0:x1], inp[y0:y1, x0:x1], mask[y0:y1, x0:x1]
    target = g(img_c)
    best, bd = 0.0, 1e18
    for s in [0, .5, .8, 1.1, 1.5, 2, 2.5, 3, 4]:
        a = cv2.GaussianBlur(mask_c.astype(np.float32), (0, 0), s) if s else mask_c.astype(np.float32)
        r = inp_c * (1 - a[..., None]) + np.array(color, np.float32) * a[..., None]
        d = abs(g(np.clip(r, 0, 255).astype(np.uint8)) - target)
        if d < bd:
            best, bd = s, d
    return best


# ---------------------------------------------------------------- analysis pass
def analyse(img, label, use_paddle, gpu):
    n = len(label)
    boxes = paddle_candidates(img, n, gpu)[:6] if use_paddle else []
    cb = detect_classic(img, n)
    if cb is not None:
        boxes.append(cb)
    if not boxes:
        return None, "no number found"
    reason = "digit count mismatch"
    for box in boxes:
        mask, stats = segment(img, box, n)
        if mask is not None:
            break
        reason = stats
    else:
        return None, reason
    glyphs = []
    for s_ in stats:
        sx, sy, sw, sh, _, comp, (ox, oy) = s_
        crop = comp[sy - oy:sy - oy + sh, sx - ox:sx - ox + sw] * 255
        glyphs.append(cv2.GaussianBlur(crop, (0, 0), 0.7))
    ux0 = min(s_[0] for s_ in stats); uy0 = min(s_[1] for s_ in stats)
    ux1 = max(s_[0] + s_[2] for s_ in stats); uy1 = max(s_[1] + s_[3] for s_ in stats)
    gaps = [stats[i + 1][0] - (stats[i][0] + stats[i][2]) for i in range(n - 1)]
    hmed = float(np.median([s_[3] for s_ in stats]))
    return dict(box=box, mask=mask, stats=stats, glyphs=glyphs, label=label,
                angle=estimate_angle(stats), hmed=hmed,
                gap=float(np.median(gaps)) if gaps else 0.15 * hmed,
                center=((ux0 + ux1) / 2, (uy0 + uy1) / 2),
                stroke=float(np.mean([stroke_ratio(g) for g in glyphs])),
                color=np.median(img[mask > 0], axis=0)), None


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
        return no("synth: already 3 digits")
    new_digits = [rng.choice(hundreds) if (i == 0 and need >= 1) else rng.randint(0, 9) for i in range(need)]
    front = True
    new_glyphs = []
    for d in new_digits:
        cands = [c for c in bank[d] if c[0] != key]
        if not cands:
            return no("synth: no glyph for borrowed digit in bank")
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
        return no("synth: 3-digit number too wide for image")
    if "cache" not in inf:   # computed once per source image
        k = max(5, int(0.2 * hm)) | 1
        big = cv2.dilate(inf["mask"], np.ones((k, k), np.uint8))
        inp = cv2.inpaint(img, big * 255, 3, cv2.INPAINT_TELEA)
        lum = cv2.GaussianBlur(cv2.cvtColor(inp, cv2.COLOR_BGR2GRAY).astype(np.float32), (0, 0), 12)
        ref = np.median(lum[inf["mask"] > 0])
        shade = np.clip(lum / max(ref, 1), 0.7, 1.3)[..., None]
        sg = estimate_blur(img, inf["mask"], inp, inf["color"], inf["box"])
        inf["cache"] = (inp, shade, sg)
    inp, shade, sigma = inf["cache"]
    # rotate + scale canvas, place at original centre
    cx, cy = inf["center"]
    M = cv2.getRotationMatrix2D((cw / 2, ch / 2), -inf["angle"], s)
    M[0, 2] += cx - cw / 2
    M[1, 2] += cy - ch / 2
    alpha = cv2.warpAffine(canvas, M, (W, H), flags=cv2.INTER_LINEAR).astype(np.float32) / 255
    ys, xs = np.where(alpha > 0.5)
    if xs.size == 0 or xs.min() < 2 or ys.min() < 2 or xs.max() > W - 3 or ys.max() > H - 3:
        return no("synth: new number falls outside image")
    if sigma:
        alpha = cv2.GaussianBlur(alpha, (0, 0), sigma)
    fill = inf["color"].astype(np.float32)[None, None, :] * shade
    out = inp.astype(np.float32) * (1 - alpha[..., None]) + fill * alpha[..., None]
    return np.clip(out, 0, 255).astype(np.uint8), labels


def panel(img, inf, out, labels, note):
    a = img.copy()
    if inf is not None:
        ov = a.copy(); ov[inf["mask"] > 0] = (0, 0, 255)
        a = cv2.addWeighted(ov, 0.5, a, 0.5, 0)
        x, y, w, h = inf["box"]; cv2.rectangle(a, (x, y), (x + w, y + h), (0, 255, 0), 1)
        for st in inf["stats"]:
            cv2.rectangle(a, (st[0], st[1]), (st[0] + st[2], st[1] + st[3]), (255, 255, 0), 1)
    b = out if out is not None else np.full_like(img, 40)
    if out is None:
        cv2.putText(b, "REJECTED", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    tgt = 320
    f = lambda im: cv2.resize(im, (int(im.shape[1] * tgt / im.shape[0]), tgt))
    p = np.hstack([f(img), f(a), f(b)])
    bar = np.full((24, p.shape[1], 3), 255, np.uint8)
    cv2.putText(bar, note[:110], (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
    return np.vstack([bar, p])


def overlay(img, inf):
    o = img.copy()
    o[inf["mask"] > 0] = (0.5 * o[inf["mask"] > 0] + 0.5 * np.array([0, 0, 255])).astype(np.uint8)
    x, y, w, h = inf["box"]
    cv2.rectangle(o, (x, y), (x + w, y + h), (0, 255, 0), 1)
    return o


def sheet_row(a, b, tw=360):
    def fit(im):
        s = tw / im.shape[1]
        return cv2.resize(im, (tw, max(1, int(im.shape[0] * s))))
    a, b = fit(a), fit(b)
    h = max(a.shape[0], b.shape[0])
    pad = lambda im: cv2.copyMakeBorder(im, 0, h - im.shape[0], 0, 4, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    return np.hstack([pad(a), pad(b)])


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
    ap.add_argument("--cpu", action="store_true", help="run PaddleOCR on CPU (default: GPU)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0, help="only process first N images")
    ap.add_argument("--dry-run", type=int, default=0, metavar="N",
                    help="analyse a random pool (--pool), save N before/after pairs as contact sheets in OUT/dryrun, write no dataset")
    ap.add_argument("--pool", type=int, default=150, help="dry-run pool size (needed to build the digit bank)")
    a = ap.parse_args()
    rng = random.Random(a.seed)
    out_i = Path(a.out) / "images"; out_l = Path(a.out) / "labels"
    if not a.dry_run:
        out_i.mkdir(parents=True, exist_ok=True); out_l.mkdir(parents=True, exist_ok=True)

    files = sorted(p for p in Path(a.images).iterdir() if p.suffix.lower() in IMG_EXT)
    if a.dry_run:
        files = rng.sample(files, min(a.pool, len(files)))
    elif a.limit:
        files = files[:a.limit]
    infos, imgs, reasons = {}, {}, {}
    t0 = time.time()
    for ii, p in enumerate(files):
        if ii % 20 == 0:
            print(f"[{ii}/{len(files)}] analysed={len(infos)} {time.time() - t0:.0f}s", flush=True)
        lab = read_label(Path(a.labels) / (p.stem + ".txt"))
        img = cv2.imread(str(p))
        if img is None or not lab:
            reasons["bad image/label"] = reasons.get("bad image/label", 0) + 1
            continue
        if len(lab) > 2:
            reasons["already 3+ digits (skipped)"] = reasons.get("already 3+ digits (skipped)", 0) + 1
            continue
        inf, why = analyse(img, lab, not a.no_paddle, not a.cpu)
        if inf:
            infos[p.stem], imgs[p.stem] = inf, img
        else:
            reasons[why] = reasons.get(why, 0) + 1
    print(f"analysed ok: {len(infos)}/{len(files)}  failures: {reasons}")
    bank = build_bank(infos)
    print("glyphs per digit:", {d: len(v) for d, v in bank.items()})

    hund = [int(c) for c in a.hundreds]
    if a.dry_run:
        d = Path(a.out) / "dryrun"; d.mkdir(parents=True, exist_ok=True)
        rows, nsheet, shown, rej = [], 0, 0, 0
        keys = list(infos)
        rng.shuffle(keys)
        for k in keys:
            if shown >= a.dry_run:
                break
            r = synth(imgs[k], k, infos[k], bank, rng, a.max_width_ratio, a.min_scale, hund)
            if r is None:
                rej += 1
                continue
            im, labels = r
            cv2.putText(im, " ".join(map(str, labels)), (4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            rows.append(sheet_row(overlay(imgs[k], infos[k]), im))
            shown += 1
            if len(rows) == 4 or shown == a.dry_run:
                w = max(r_.shape[1] for r_ in rows)
                rows = [cv2.copyMakeBorder(r_, 0, 0, 0, w - r_.shape[1], cv2.BORDER_CONSTANT, value=(255, 255, 255)) for r_ in rows]
                cv2.imwrite(str(d / f"sheet_{nsheet}.jpg"), np.vstack(rows)); nsheet += 1; rows = []
        print(f"dry run: {shown} pairs in {nsheet} sheet(s) at {d}  (left = detected box/mask, right = result; {rej} rejected at synth)")
        return

    made = 0
    for k, inf in infos.items():
        for j in range(a.per_image):
            r = synth(imgs[k], k, inf, bank, rng, a.max_width_ratio, a.min_scale, hund)
            if r is None:
                continue
            im, labels = r
            cv2.imwrite(str(out_i / f"{k}_3d{j}.jpg"), im)
            (out_l / f"{k}_3d{j}.txt").write_text(" ".join(map(str, labels)))
            made += 1
    print(f"written {made} images to {a.out}")


if __name__ == "__main__":
    main()

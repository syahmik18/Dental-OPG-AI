"""
NERVE POST-PROCESSING  –  v3
Correctly detects the inferior alveolar canal as a dark band INSIDE
the bright mandible bone, not at the jaw border.

Anatomical facts encoded here:
  • Canal runs from the lingula (top of ramus) downward through the mandible body
  • Path is U-shaped: high on sides (~30–38 % image height),
    lowest near premolars (~65–75 % image height)
  • LEFT  side occupies roughly x = 4 %–44 % of image width
  • RIGHT side occupies roughly x = 56 %–96 % of image width
  • Gap in the middle (mental symphysis) – canals do NOT connect
  • Canal appears as a DARK band bordered by TWO BRIGHT cortical lines
"""

import cv2
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# DEBUG HELPER
# ─────────────────────────────────────────────────────────────────────────────

def _save_debug(name, img):
    vis = (img > 0).astype(np.uint8) * 255 if img.max() <= 1 else img
    cv2.imwrite(f"debug_nerve_{name}.jpg", vis)
    print(f"  [debug] {name}: {int(np.sum(vis > 0))} px")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1  –  find the mandible bone body per column
# The mandible body is the brightest continuous horizontal region in the
# lower part of the panoramic image.
# ─────────────────────────────────────────────────────────────────────────────

def _find_bone_band_per_column(img, x0, x1, y_top_frac=0.30, y_bot_frac=0.85):
    """
    For each column in [x0, x1], find the vertical extent of the bright bone.
    Returns arrays (bone_top[col], bone_bot[col]) in global pixel coordinates.
    """
    h, w = img.shape
    y_top = int(y_top_frac * h)
    y_bot = int(y_bot_frac * h)

    clahe    = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(img)

    bone_top = np.full(w, -1, dtype=int)
    bone_bot = np.full(w, -1, dtype=int)

    for col in range(x0, x1):
        col_strip = enhanced[y_top:y_bot, col].astype(float)

        # Smooth vertically to reduce noise
        col_smooth = np.convolve(col_strip, np.ones(5) / 5, mode='same')

        # Adaptive threshold: bone is above-average brightness in this strip
        thresh = np.percentile(col_smooth, 55)
        bone_pixels = np.where(col_smooth > thresh)[0]

        if len(bone_pixels) < 10:
            continue

        # Find the largest continuous bone region
        gaps   = np.where(np.diff(bone_pixels) > 5)[0]
        segs   = np.split(bone_pixels, gaps + 1)
        largest = max(segs, key=len)

        bone_top[col] = y_top + int(largest[0])
        bone_bot[col] = y_top + int(largest[-1])

    return bone_top, bone_bot


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2  –  find canal centre inside the bone band per column
# Canal = darkest local minimum between two bright ridges within the bone band
# ─────────────────────────────────────────────────────────────────────────────

def _find_canal_centre_per_column(img, x0, x1, bone_top, bone_bot,
                                  min_canal_px=3, max_canal_px=35):
    """
    For each column find the y-centre of the dark canal band inside the bone.
    Returns array canal_y[col] in global pixel coordinates (-1 = not found).
    """
    h, w   = img.shape
    canal_y = np.full(w, -1, dtype=int)

    clahe    = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(img)

    for col in range(x0, x1):
        bt = bone_top[col]
        bb = bone_bot[col]
        if bt < 0 or bb < 0 or (bb - bt) < 15:
            continue

        strip = enhanced[bt:bb, col].astype(float)

        # Smooth to suppress noise
        strip_s = np.convolve(strip, np.ones(3) / 3, mode='same')

        # Canal is the DARKEST region inside the bone
        # Look for a dip: local min flanked by higher values
        min_idx  = int(np.argmin(strip_s))
        min_val  = strip_s[min_idx]
        mean_val = np.mean(strip_s)

        # Canal must be meaningfully darker than the mean bone brightness
        if min_val > mean_val * 0.82:
            continue   # no obvious dark band

        # Expand from min to find canal width
        threshold = min_val + 0.4 * (mean_val - min_val)
        top_edge  = min_idx
        bot_edge  = min_idx

        while top_edge > 0 and strip_s[top_edge - 1] < threshold:
            top_edge -= 1
        while bot_edge < len(strip_s) - 1 and strip_s[bot_edge + 1] < threshold:
            bot_edge += 1

        canal_width = bot_edge - top_edge
        if not (min_canal_px <= canal_width <= max_canal_px):
            continue

        centre = bt + (top_edge + bot_edge) // 2
        if 0 <= centre < h:
            canal_y[col] = centre

    return canal_y


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3  –  robustly fit a smooth curve through canal centres
# ─────────────────────────────────────────────────────────────────────────────

def _fit_canal_curve(canal_y, x0, x1, img_shape, poly_degree=4):
    """
    Fit a polynomial y = f(x) through valid canal_y points and
    return a binary mask of the smooth centreline.
    """
    h, w = img_shape[:2]

    xs = np.array([c for c in range(x0, x1) if canal_y[c] >= 0])
    ys = np.array([canal_y[c] for c in xs])

    if len(xs) < 15:
        print(f"  [nerve] only {len(xs)} valid canal points in [{x0},{x1}] – skipping fit")
        # Fall back: draw raw points
        mask = np.zeros((h, w), dtype=np.uint8)
        for c in range(x0, x1):
            if 0 <= canal_y[c] < h:
                mask[canal_y[c], c] = 255
        return mask

    # RANSAC-style: remove outliers via IQR before fitting
    q25, q75 = np.percentile(ys, [25, 75])
    iqr      = q75 - q25
    valid    = (ys >= q25 - 1.5 * iqr) & (ys <= q75 + 1.5 * iqr)
    xs, ys   = xs[valid], ys[valid]

    if len(xs) < 10:
        mask = np.zeros((h, w), dtype=np.uint8)
        return mask

    try:
        coeffs = np.polyfit(xs, ys, poly_degree)
    except Exception:
        mask = np.zeros((h, w), dtype=np.uint8)
        return mask

    mask  = np.zeros((h, w), dtype=np.uint8)
    x_arr = np.arange(x0, x1)
    y_arr = np.clip(np.polyval(coeffs, x_arr).astype(int), 0, h - 1)
    for x, y in zip(x_arr, y_arr):
        mask[y, x] = 255

    return mask


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4  –  process one side (left or right)
# ─────────────────────────────────────────────────────────────────────────────

def _detect_canal_one_side(img, x0, x1, side_label=""):
    h, w = img.shape

    print(f"  [nerve] processing {side_label} side  x=[{x0},{x1}]")

    # 1. Locate bone band per column
    bone_top, bone_bot = _find_bone_band_per_column(img, x0, x1)
    bone_found = np.sum(bone_top[x0:x1] >= 0)
    print(f"  [nerve]   bone found in {bone_found}/{x1-x0} columns")

    # 2. Find canal dark-band centre per column
    canal_y = _find_canal_centre_per_column(img, x0, x1, bone_top, bone_bot)
    canal_found = np.sum(canal_y[x0:x1] >= 0)
    print(f"  [nerve]   canal found in {canal_found}/{x1-x0} columns")

    # 3. Fit smooth curve
    canal_mask = _fit_canal_curve(canal_y, x0, x1, img.shape)
    print(f"  [nerve]   curve pixels: {int(np.sum(canal_mask > 0))}")

    return canal_mask


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def post_process_nerve_mask(raw_mask, img=None):
    """
    Lightweight post-processing designed for U-Net output.
    U-Net already predicts a good mask — we just need to:
      1. Clip to anatomical ROI
      2. Close small gaps
      3. Remove small noise blobs
      4. Dilate to visible thickness
    Skeletonization and poly-fit are intentionally removed —
    they were destroying the U-Net mask.
    """
    h, w = raw_mask.shape
    print(f"  [nerve] raw_mask pixels  : {int(np.sum(raw_mask > 0))}")

    # ── STAGE 1: ROI clip ───────────────────────────────────────
    # Canal lives between 30% and 85% height on panoramic X-ray
    roi = np.zeros_like(raw_mask)
    roi[int(0.25 * h):int(0.90 * h), int(0.02 * w):int(0.98 * w)] = 255  # wider ROI
    mask = cv2.bitwise_and(raw_mask, roi)
    print(f"  [nerve] after ROI clip   : {int(np.sum(mask > 0))}")
    _save_debug("1_roi_clip", mask)

    # ── STAGE 2: Close small gaps ───────────────────────────────
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_close, iterations=2)
    print(f"  [nerve] after close      : {int(np.sum(mask > 0))}")
    _save_debug("2_closed", mask)

    # ── STAGE 3: Remove small noise blobs ──────────────────────
    # Keep only components larger than 1% of image area
    num, labels, stats, _ = cv2.connectedComponentsWithStats(
        (mask > 0).astype(np.uint8), connectivity=8)
    min_area = int(0.001 * h * w)  # 0.1% of image – much less aggressive
    cleaned  = np.zeros_like(mask)
    for i in range(1, num):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            cleaned[labels == i] = 255
    mask = cleaned
    print(f"  [nerve] after denoise    : {int(np.sum(mask > 0))}")
    _save_debug("3_denoised", mask)

    # ── STAGE 4: Fallback if mask still empty ──────────────────
    if int(np.sum(mask > 0)) < 500 and img is not None:
        print("  [nerve] mask too sparse – running image-based fallback")
        left_mask  = _detect_canal_one_side(img,
                        int(0.04*w), int(0.44*w), "LEFT")
        right_mask = _detect_canal_one_side(img,
                        int(0.56*w), int(0.96*w), "RIGHT")
        mask = cv2.bitwise_or(left_mask, right_mask)
        _save_debug("4_fallback", mask)

    # ── STAGE 5: Dilate to visible line thickness ───────────────
    k_draw = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    final  = cv2.dilate(mask, k_draw, iterations=1)
    print(f"  [nerve] FINAL pixels     : {int(np.sum(final > 0))}")
    _save_debug("5_final", final)

    return final


# ─────────────────────────────────────────────────────────────────────────────
# INTERNAL UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def _skeletonize(mask):
    binary = (mask > 0).astype(np.uint8) * 255
    try:
        import cv2.ximgproc as xip
        return xip.thinning(binary, thinningType=xip.THINNING_ZHANGSUEN)
    except (ImportError, AttributeError):
        pass
    skel   = np.zeros_like(binary)
    temp   = binary.copy()
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    for _ in range(300):
        eroded = cv2.erode(temp, kernel)
        opened = cv2.dilate(eroded, kernel)
        skel   = cv2.bitwise_or(skel, cv2.subtract(temp, opened))
        temp   = eroded
        if cv2.countNonZero(temp) == 0:
            break
    return skel


def _keep_largest(mask):
    num, labels, stats, _ = cv2.connectedComponentsWithStats(
        (mask > 0).astype(np.uint8), connectivity=8)
    if num <= 1:
        return mask
    best = max(range(1, num), key=lambda i: stats[i, cv2.CC_STAT_AREA])
    result = np.zeros_like(mask)
    result[labels == best] = 255
    return result


def _fit_poly_mask(skeleton, deg=3):
    ys, xs = np.where(skeleton > 0)
    if len(xs) < 15:
        return skeleton
    h, w = skeleton.shape
    try:
        coeffs = np.polyfit(ys, xs, deg)
    except Exception:
        return skeleton
    y_arr = np.arange(int(ys.min()), int(ys.max()) + 1)
    x_arr = np.clip(np.polyval(coeffs, y_arr).astype(int), 0, w - 1)
    out = np.zeros_like(skeleton)
    for y, x in zip(y_arr, x_arr):
        if 0 <= y < h:
            out[int(y), int(x)] = 255
    return out
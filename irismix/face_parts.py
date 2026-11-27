"""Landmark part weights on a 512 x 512 face crop (decomposer part weights, renderer source conditions and captions):
  landmark_part_weights(p)  5 soft part weights (lips, eyes, brows, face, forehead), priority lips > eyes > brows > forehead > face,
                 summing to 1 per pixel; p = 478 MediaPipe landmarks in px of the 512 crop
plus the MediaPipe index groups of the lips, eyes, brows and face oval."""
import cv2, numpy as np

S = 512
PARTS = ('lips', 'eyes', 'brows', 'face', 'forehead')

LIP_OUT = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185]
LIP_IN = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 415, 310, 311, 312, 13, 82, 81, 80, 191]
EYE_L = [33, 246, 161, 160, 159, 158, 157, 173, 133, 155, 154, 153, 145, 144, 163, 7]
EYE_R = [263, 466, 388, 387, 386, 385, 384, 398, 362, 382, 381, 380, 374, 373, 390, 249]
BROW_L = [70, 63, 105, 66, 107, 55, 65, 52, 53, 46]
BROW_R = [300, 293, 334, 296, 336, 285, 295, 282, 283, 276]
BROWLINE_L = [[70, 63, 105, 66, 107], [46, 53, 52, 65, 55]]
BROWLINE_R = [[300, 293, 334, 296, 336], [276, 283, 282, 295, 285]]
FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149,
             150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
UPPER_OVAL = [10, 338, 297, 332, 284, 251, 389, 162, 21, 54, 103, 67, 109]


def polygon_mask(pts, k=0, blur=0):
    m = np.zeros((S, S), np.uint8); cv2.fillPoly(m, [np.round(pts).astype(np.int32)], 1)
    if k > 0: m = cv2.dilate(m, np.ones((2 * k + 1, 2 * k + 1), np.uint8))
    m = m.astype(np.float32)
    return cv2.GaussianBlur(m, (0, 0), blur) if blur > 0 else m


def scale_about_centre(pts, s):
    c = pts.mean(0); return c + s * (pts - c)


def landmark_part_weights(p):
    """soft 5-part weights from landmarks, same parts as the LayerBeauty-Creative part weights."""
    lips = polygon_mask(p[LIP_OUT], 3, 2)
    eyes = np.zeros((S, S), np.float32); brows = np.zeros((S, S), np.float32)
    for eye, bl, br in ((EYE_L, BROWLINE_L, BROW_L), (EYE_R, BROWLINE_R, BROW_R)):
        e = p[eye]
        # eye region: eye contour widened (liner wings) up to the lower brow line (lid / shadow area)
        reg = np.concatenate([scale_about_centre(e, np.array([1.45, 1.9], np.float32)), p[bl[1]]], 0)
        eyes = np.maximum(eyes, polygon_mask(cv2.convexHull(reg.astype(np.float32))[:, 0], 4, 4))
        brows = np.maximum(brows, polygon_mask(cv2.convexHull(p[br])[:, 0], 3, 2))
    c = p[FACE_OVAL].mean(0)
    up = p[UPPER_OVAL]
    oval_ext = np.concatenate([p[FACE_OVAL], c + 1.75 * (up - c)], 0)
    face_all = polygon_mask(cv2.convexHull(oval_ext.astype(np.float32))[:, 0], 0, 3)
    brow_y = p[BROW_L + BROW_R][:, 1].min()
    yy = np.arange(S, dtype=np.float32)[:, None].repeat(S, 1)
    fore = face_all * np.clip((brow_y - yy) / 12 + 0.5, 0, 1)
    out = {}; rest = np.ones((S, S), np.float32)
    for k, m in (('lips', lips), ('eyes', eyes), ('brows', brows), ('forehead', fore)):
        out[k] = m * rest; rest = rest * (1 - m)
    out['face'] = rest                                       # remainder (skin, background) -> face, as in Creative w
    return np.stack([out[k] for k in PARTS])


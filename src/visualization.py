import numpy as np
import cv2


def overlay_mask(rgb, mask, alpha=0.45):
    color = np.zeros_like(rgb)
    color[:, :, 2] = 255  # red overlay
    mask3 = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    overlay = np.where(mask3 > 0, color, 0)
    out = cv2.addWeighted(rgb, 1.0, overlay, alpha, 0)
    return out


def stack_h(*imgs):
    w = imgs[0].shape[1]
    h = imgs[0].shape[0]
    imgs2 = [cv2.resize(im, (w, h)) for im in imgs]
    return np.hstack(imgs2)


def add_legend(img, lines):
    """Draw simple text legend at top-left."""
    out = img.copy()
    x, y = 10, 22
    for i, t in enumerate(lines):
        cv2.putText(out, t, (x, y + 22*i), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(out, t, (x, y + 22*i), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (20, 20, 20), 1, cv2.LINE_AA)
    return out

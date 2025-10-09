import argparse
import numpy as np
import cv2
from preprocessing import read_raster, to_uint8, compute_ndvi, align_by_resize, compute_ndwi, uint8_norm
from detection import delta_ndvi_mask, ssim_change, apply_water_mask, change_stats
from visualization import overlay_mask, stack_h, add_legend


def to_rgb(img):
    if img.shape[0] >= 3:
        rgb = np.stack([img[0], img[1], img[2]], axis=-1)
        rgb = uint8_norm(rgb)
    else:
        g = uint8_norm(img[0])
        rgb = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
    return rgb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--t1", required=True)
    ap.add_argument("--t2", required=True)
    ap.add_argument("--out", default="demo/output.png")
    ap.add_argument("--red", type=int, default=None, help="0-based RED index")
    ap.add_argument("--nir", type=int, default=None, help="0-based NIR index")
    ap.add_argument("--green", type=int, default=1,
                    help="0-based GREEN index (for NDWI). Default=1")
    ap.add_argument("--alpha", type=float, default=0.45, help="overlay transparency 0..1")
    ap.add_argument("--ndwi_thresh", type=float, default=0.15, help="water mask threshold")
    args = ap.parse_args()

    img1, _ = read_raster(args.t1)   # (C,H,W)
    img2, _ = read_raster(args.t2)
    img2 = align_by_resize(img1, img2)

    C1, C2 = img1.shape[0], img2.shape[0]

    # Band selection
    use_ndvi = False
    if args.red is not None and args.nir is not None:
        assert 0 <= args.red < C1 and 0 <= args.red < C2 and 0 <= args.nir < C1 and 0 <= args.nir < C2
        red1, nir1 = img1[args.red], img1[args.nir]
        red2, nir2 = img2[args.red], img2[args.nir]
        use_ndvi = True
    elif C1 >= 4 and C2 >= 4:
        red1, nir1 = img1[2], img1[3]
        red2, nir2 = img2[2], img2[3]
        use_ndvi = True

    rgb1 = to_rgb(img1)
    rgb2 = to_rgb(img2)

    panels = []
    legends = []

    if use_ndvi:
        ndvi1 = compute_ndvi(nir1, red1)
        ndvi2 = compute_ndvi(nir2, red2)

        # NDWI for water masking (needs GREEN + NIR)
        green1 = img1[args.green] if args.green < C1 else img1[1]
        green2 = img2[args.green] if args.green < C2 else img2[1]
        ndwi1 = compute_ndwi(green1, nir1)
        ndwi2 = compute_ndwi(green2, nir2)
        ndwi = (ndwi1 + ndwi2) / 2.0

        _, mask_ndvi = delta_ndvi_mask(ndvi1, ndvi2)
        mask_ndvi = apply_water_mask(mask_ndvi, ndwi, thresh=args.ndwi_thresh)
        over1 = overlay_mask(rgb1, mask_ndvi, alpha=args.alpha)
        over2 = overlay_mask(rgb2, mask_ndvi, alpha=args.alpha)
        ch, tot, pct = change_stats(mask_ndvi)
        legends.append(f"NDVI change: {ch}/{tot} px = {pct:.2f}% (water-masked)")
        panels.extend([over1, over2])

        # SSIM on RED
        g1, g2 = uint8_norm(red1), uint8_norm(red2)
    else:
        # SSIM-only fallback on band 0
        g1, g2 = uint8_norm(img1[0]), uint8_norm(img2[0])

    _, mask_ssim = ssim_change(g1, g2)
    if use_ndvi:
        # also water-mask SSIM when NDWI available
        mask_ssim = apply_water_mask(mask_ssim, ndwi, thresh=args.ndwi_thresh)

    over_ssim = overlay_mask(rgb2, mask_ssim, alpha=args.alpha)
    ch2, tot2, pct2 = change_stats(mask_ssim)
    legends.append(f"SSIM change: {ch2}/{tot2} px = {pct2:.2f}%")

    # Compose and annotate
    panels.append(over_ssim)
    comp = panels[0] if len(panels) == 1 else stack_h(*panels)
    comp = add_legend(comp, legends)
    cv2.imwrite(args.out, comp)
    print(f"[OK] Wrote: {args.out}")
    if not use_ndvi:
        print("[INFO] NDVI disabled (no NIR/indices). Provide --red/--nir for NDVI mode.")


if __name__ == "__main__":
    main()

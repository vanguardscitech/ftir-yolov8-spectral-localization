from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
IMAGE_PATH = ROOT / "sample.png"
OUT_DIR = ROOT / "outputs" / "real_spectrum_validation"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    img_bgr = cv2.imread(str(IMAGE_PATH))
    if img_bgr is None:
        raise FileNotFoundError(f"Cannot read {IMAGE_PATH}")

    # Isolate the Matplotlib blue curve while ignoring grey gridlines and black text.
    b, g, r = cv2.split(img_bgr)
    mask = (b > 95) & (g > 60) & (r < 95) & ((b.astype(np.int16) - r.astype(np.int16)) > 45)
    ys, xs = np.where(mask)
    if len(xs) < 50:
        raise RuntimeError("Could not isolate enough curve pixels from sample.png")

    # Use the curve extent as a robust proxy for the plot area.
    x_min, x_max = int(np.percentile(xs, 0.5)), int(np.percentile(xs, 99.5))
    y_min, y_max = int(np.percentile(ys, 0.5)), int(np.percentile(ys, 99.5))

    curve_x = []
    curve_y = []
    for x in range(x_min, x_max + 1):
        col_y = ys[xs == x]
        if len(col_y) == 0:
            continue
        curve_x.append(x)
        curve_y.append(float(np.median(col_y)))

    curve_x = np.asarray(curve_x, dtype=float)
    curve_y = np.asarray(curve_y, dtype=float)

    # Approximate axis calibration from the visible plot: x is 4000 -> 400 cm-1.
    wavenumber = 4000 + (curve_x - x_min) / max(x_max - x_min, 1) * (400 - 4000)

    # Normalize curve height into the same approximate transmittance scale used in training.
    transmittance = 100 - (curve_y - y_min) / max(y_max - y_min, 1) * 50
    transmittance = np.clip(transmittance, 0, 110)

    order = np.argsort(wavenumber)[::-1]
    wavenumber = wavenumber[order]
    transmittance = transmittance[order]

    data_path = OUT_DIR / "sample_digitized_curve.csv"
    np.savetxt(
        data_path,
        np.column_stack([wavenumber, transmittance]),
        delimiter=",",
        header="wavenumber_cm-1,transmittance_percent",
        comments="",
    )

    fig, ax = plt.subplots(figsize=(10, 6), dpi=100)
    ax.plot(wavenumber, transmittance, color="black", linewidth=1.0)
    ax.set_xlim(4000, 400)
    ax.set_ylim(0, 110)
    ax.set_xlabel("Wavenumber (1/cm)", fontsize=10, fontweight="bold")
    ax.set_ylabel("%T", fontsize=10, fontweight="bold")
    ax.minorticks_on()
    ax.tick_params(which="major", length=6, width=1.5)
    ax.tick_params(which="minor", length=3, width=1)
    for axis in ["top", "bottom", "left", "right"]:
        ax.spines[axis].set_linewidth(1.5)
    plt.tight_layout()

    standardized_path = OUT_DIR / "sample_real_ftir_standardized.png"
    fig.savefig(standardized_path, dpi=100)
    plt.close(fig)

    print(f"curve_points={len(wavenumber)}")
    print(f"csv={data_path}")
    print(f"image={standardized_path}")


if __name__ == "__main__":
    main()

from pathlib import Path
import csv
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / "Ultralytics"))

from ultralytics import YOLO

MODEL_PATH = ROOT / "models" / "best.pt"
IMAGE_PATH = ROOT / "data" / "FTIR_Dataset_Base" / "train" / "images" / "ftir_sample_007.png"
OUT_DIR = ROOT / "outputs" / "figure9_clean_prediction"
CONF_THRESHOLD = 0.25


COLORS = {
    "O-H Alcohol": "#f97316",
    "N-H Amine": "#7c3aed",
    "C-H Alkane": "#16a34a",
    "C-H Aldehyde": "#ca8a04",
    "S-H Thiol": "#0f766e",
    "C-N Nitrile": "#2563eb",
    "C-C Alkyne": "#9333ea",
    "C=O Carbonyl": "#dc2626",
    "C=C Alkene": "#2563eb",
    "N-O Nitro": "#0f766e",
    "C-O Ether": "#0891b2",
}


def estimate_wavenumber(center_x, width):
    plot_left = int(width * 0.071)
    plot_right = int(width * 0.985)
    x = min(max(center_x, plot_left), plot_right)
    fraction = (x - plot_left) / max(plot_right - plot_left, 1)
    return 4000.0 + fraction * (400.0 - 4000.0)


def extract_curve(image_path):
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Cannot read {image_path}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect dark spectral curve while excluding the frame and text-heavy margins.
    h, w = gray.shape
    plot_left, plot_right = int(w * 0.079), int(w * 0.978)
    plot_top, plot_bottom = int(h * 0.045), int(h * 0.885)
    roi = gray[plot_top:plot_bottom, plot_left:plot_right]
    mask = roi < 95

    xs, ys = [], []
    for x in range(6, mask.shape[1] - 6):
        y_idx = np.where(mask[:, x])[0]
        if len(y_idx) == 0:
            continue
        # Select the central dark trace; avoid axis/tick artifacts near the bottom.
        y_idx = y_idx[(y_idx > 4) & (y_idx < int(mask.shape[0] * 0.92))]
        if len(y_idx) == 0:
            continue
        xs.append(plot_left + x)
        ys.append(plot_top + float(np.median(y_idx)))

    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    wn = np.asarray([estimate_wavenumber(x, w) for x in xs])

    # Axis scale is 0-110 %T in the generated spectra.
    transmittance = 110.0 - (ys - plot_top) / max(plot_bottom - plot_top, 1) * 110.0
    transmittance = np.clip(transmittance, 0, 110)
    order = np.argsort(wn)[::-1]
    return wn[order], transmittance[order]


def predict():
    model = YOLO(str(MODEL_PATH))
    result = model.predict(str(IMAGE_PATH), conf=CONF_THRESHOLD, verbose=False)[0]
    img = cv2.imread(str(IMAGE_PATH))
    h, w = img.shape[:2]
    preds = []
    for box in result.boxes:
        x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].tolist()]
        cx = (x1 + x2) / 2.0
        preds.append(
            {
                "label": result.names[int(box.cls[0])],
                "confidence": float(box.conf[0]),
                "center_x_px": cx,
                "approx_wavenumber_cm-1": estimate_wavenumber(cx, w),
            }
        )
    return sorted(preds, key=lambda item: item["approx_wavenumber_cm-1"], reverse=True)


def merge_close(preds, min_gap=55):
    kept = []
    for pred in sorted(preds, key=lambda item: item["confidence"], reverse=True):
        if all(abs(pred["approx_wavenumber_cm-1"] - other["approx_wavenumber_cm-1"]) >= min_gap for other in kept):
            kept.append(pred)
    return sorted(kept, key=lambda item: item["approx_wavenumber_cm-1"], reverse=True)


def y_at(wn, x, y):
    order = np.argsort(x)
    return float(np.interp(wn, x[order], y[order]))


def base_axis(ax, x, y):
    ax.plot(x, y, color="black", linewidth=1.15)
    ax.set_xlim(4000, 400)
    ax.set_ylim(0, 108)
    ax.set_xlabel("Wavenumber (cm$^{-1}$)", fontweight="bold")
    ax.set_ylabel("Transmittance (%)", fontweight="bold")
    ax.grid(True, linestyle="--", linewidth=0.55, alpha=0.30)
    for spine in ax.spines.values():
        spine.set_linewidth(1.2)


def render_labeled(x, y, preds):
    selected = merge_close(preds)
    fig, ax = plt.subplots(figsize=(12.5, 6.7), dpi=180)
    base_axis(ax, x, y)

    # Fixed label lanes reduce collisions in the dense 2300-1500 cm-1 region.
    lanes = [103, 96, 89, 82]
    for idx, pred in enumerate(selected):
        wn = pred["approx_wavenumber_cm-1"]
        yy = y_at(wn, x, y)
        color = COLORS.get(pred["label"], "#16a34a")
        ax.vlines(wn, ymin=2, ymax=yy, color=color, linewidth=1.9, alpha=0.94)
        ax.scatter([wn], [yy], s=28, color=color, edgecolor="white", linewidth=0.8, zorder=5)
        text = f"{pred['label']}\n{wn:.0f} cm$^{{-1}}$ | {pred['confidence']:.2f}"
        ax.annotate(
            text,
            xy=(wn, yy),
            xytext=(wn, lanes[idx % len(lanes)]),
            ha="center",
            va="center",
            fontsize=7.8,
            color="#111827",
            bbox=dict(boxstyle="round,pad=0.23", fc="white", ec=color, lw=1.0),
            arrowprops=dict(arrowstyle="-", color=color, lw=0.9, shrinkA=4, shrinkB=4),
        )

    ax.set_title("YOLOv8n Inference Output with Clean Vertical Spectral Markers", fontweight="bold")
    fig.tight_layout()
    out = OUT_DIR / "figure9_clean_yolo_prediction.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out, selected


def render_numbered_table(x, y, preds):
    selected = merge_close(preds)
    fig = plt.figure(figsize=(14.2, 7.0), dpi=180)
    gs = fig.add_gridspec(1, 2, width_ratios=[3.2, 1.28], wspace=0.04)
    ax = fig.add_subplot(gs[0, 0])
    tax = fig.add_subplot(gs[0, 1])
    base_axis(ax, x, y)

    for idx, pred in enumerate(selected, start=1):
        wn = pred["approx_wavenumber_cm-1"]
        yy = y_at(wn, x, y)
        color = COLORS.get(pred["label"], "#16a34a")
        ax.vlines(wn, ymin=2, ymax=yy, color=color, linewidth=1.35, alpha=0.85)
        ax.scatter([wn], [yy], s=26, color=color, edgecolor="white", linewidth=0.7, zorder=5)
        ax.text(
            wn,
            104 - (idx % 3) * 3.2,
            str(idx),
            ha="center",
            va="center",
            fontsize=7.4,
            color="white",
            bbox=dict(boxstyle="circle,pad=0.22", fc=color, ec="white", lw=0.7),
        )

    rows = [
        [
            str(i),
            p["label"],
            f"{p['approx_wavenumber_cm-1']:.0f}",
            f"{p['confidence']:.2f}",
        ]
        for i, p in enumerate(selected, start=1)
    ]
    tax.axis("off")
    table = tax.table(
        cellText=rows,
        colLabels=["#", "Predicted band", "cm$^{-1}$", "Conf."],
        colLoc="left",
        cellLoc="left",
        loc="center",
        colWidths=[0.12, 0.50, 0.21, 0.17],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7.5)
    table.scale(1.0, 1.35)
    for (row, col), cell in table.get_celld().items():
        cell.set_linewidth(0.35)
        if row == 0:
            cell.set_facecolor("#f3f4f6")
            cell.set_text_props(weight="bold")
        elif col == 0:
            color = COLORS.get(selected[row - 1]["label"], "#16a34a")
            cell.set_facecolor(color)
            cell.set_text_props(color="white", weight="bold")

    ax.set_title("YOLOv8n Inference Output - Numbered Spectral Markers", fontweight="bold")
    fig.tight_layout()
    out = OUT_DIR / "figure9_clean_yolo_prediction_numbered_table.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def write_csv(preds):
    path = OUT_DIR / "figure9_clean_yolo_predictions.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["label", "confidence", "center_x_px", "approx_wavenumber_cm-1"],
        )
        writer.writeheader()
        writer.writerows(preds)
    return path


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    x, y = extract_curve(IMAGE_PATH)
    preds = predict()
    csv_path = write_csv(preds)
    labeled_path, selected = render_labeled(x, y, preds)
    table_path = render_numbered_table(x, y, preds)
    print(f"predictions={len(preds)}")
    print(f"selected={len(selected)}")
    print(f"csv={csv_path}")
    print(f"labeled={labeled_path}")
    print(f"numbered_table={table_path}")
    for pred in selected:
        print(f"{pred['label']}\t{pred['confidence']:.3f}\t{pred['approx_wavenumber_cm-1']:.1f}")


if __name__ == "__main__":
    main()

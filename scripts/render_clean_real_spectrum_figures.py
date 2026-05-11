from pathlib import Path
import csv

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs" / "real_spectrum_validation"
CURVE_CSV = OUT_DIR / "sample_digitized_curve.csv"
PRED_CSV = OUT_DIR / "sample_real_ftir_standardized_predictions.csv"


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


def read_curve():
    data = np.genfromtxt(CURVE_CSV, delimiter=",", names=True)
    return data["wavenumber_cm1"], data["transmittance_percent"]


def read_predictions():
    preds = []
    with PRED_CSV.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            preds.append(
                {
                    "label": row["label"],
                    "confidence": float(row["confidence"]),
                    "wn": float(row["approx_wavenumber_cm-1"]),
                }
            )
    return sorted(preds, key=lambda item: item["wn"], reverse=True)


def merge_close_predictions(preds, min_gap=65):
    kept = []
    for pred in sorted(preds, key=lambda item: item["confidence"], reverse=True):
        if all(abs(pred["wn"] - other["wn"]) >= min_gap for other in kept):
            kept.append(pred)
    return sorted(kept, key=lambda item: item["wn"], reverse=True)


def transmittance_at(wn, x, y):
    order = np.argsort(x)
    return float(np.interp(wn, x[order], y[order]))


def base_axis(ax, x, y):
    ax.plot(x, y, color="black", linewidth=1.15)
    ax.set_xlim(4000, 400)
    ax.set_ylim(45, 105)
    ax.set_xlabel("Wavenumber (cm$^{-1}$)", fontweight="bold")
    ax.set_ylabel("Transmittance (%)", fontweight="bold")
    ax.grid(True, linestyle="--", linewidth=0.55, alpha=0.35)
    for spine in ax.spines.values():
        spine.set_linewidth(1.2)


def render_high_confidence(x, y, preds):
    selected = [p for p in preds if p["confidence"] >= 0.40]
    selected = merge_close_predictions(selected, min_gap=75)

    fig, ax = plt.subplots(figsize=(12, 6.6), dpi=180)
    base_axis(ax, x, y)
    lanes = [102.5, 98.8, 95.1, 91.4]

    for idx, pred in enumerate(selected):
        color = COLORS.get(pred["label"], "#2563eb")
        yy = transmittance_at(pred["wn"], x, y)
        ax.vlines(pred["wn"], ymin=47, ymax=yy, color=color, linewidth=2.0, alpha=0.92)
        ax.scatter([pred["wn"]], [yy], s=32, color=color, edgecolor="white", linewidth=0.8, zorder=5)
        label = f"{pred['label']}\n{pred['wn']:.0f} cm$^{{-1}}$ | {pred['confidence']:.2f}"
        ax.annotate(
            label,
            xy=(pred["wn"], yy),
            xytext=(pred["wn"], lanes[idx % len(lanes)]),
            ha="center",
            va="center",
            fontsize=8.2,
            color="#111827",
            bbox=dict(boxstyle="round,pad=0.24", fc="white", ec=color, lw=1.1),
            arrowprops=dict(arrowstyle="-", color=color, lw=1.0, shrinkA=4, shrinkB=4),
        )

    ax.set_title("Real FTIR Spectrum - High-Confidence YOLOv8n Predictions", fontweight="bold")
    fig.tight_layout()
    out = OUT_DIR / "sample_real_ftir_clean_high_confidence.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out, selected


def render_numbered_table(x, y, preds):
    selected = merge_close_predictions(preds, min_gap=65)

    fig = plt.figure(figsize=(14.5, 7.2), dpi=180)
    gs = fig.add_gridspec(1, 2, width_ratios=[3.35, 1.25], wspace=0.04)
    ax = fig.add_subplot(gs[0, 0])
    tax = fig.add_subplot(gs[0, 1])
    base_axis(ax, x, y)

    for idx, pred in enumerate(selected, start=1):
        color = COLORS.get(pred["label"], "#2563eb")
        yy = transmittance_at(pred["wn"], x, y)
        ax.vlines(pred["wn"], ymin=47, ymax=yy, color=color, linewidth=1.35, alpha=0.82)
        ax.scatter([pred["wn"]], [yy], s=26, color=color, edgecolor="white", linewidth=0.7, zorder=5)
        ax.text(
            pred["wn"],
            103.0 - (idx % 3) * 2.2,
            str(idx),
            ha="center",
            va="center",
            fontsize=7.6,
            color="white",
            bbox=dict(boxstyle="circle,pad=0.23", fc=color, ec="white", lw=0.7),
        )

    ax.set_title("Real FTIR Spectrum - Numbered YOLOv8n Predictions", fontweight="bold")
    tax.axis("off")
    rows = [[str(i), p["label"], f"{p['wn']:.0f}", f"{p['confidence']:.2f}"] for i, p in enumerate(selected, start=1)]
    table = tax.table(
        cellText=rows,
        colLabels=["#", "Predicted band", "cm$^{-1}$", "Conf."],
        colLoc="left",
        cellLoc="left",
        loc="center",
        colWidths=[0.12, 0.48, 0.22, 0.18],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7.6)
    table.scale(1.0, 1.32)
    for (row, col), cell in table.get_celld().items():
        cell.set_linewidth(0.35)
        if row == 0:
            cell.set_facecolor("#f3f4f6")
            cell.set_text_props(weight="bold")
        elif col == 0:
            color = COLORS.get(selected[row - 1]["label"], "#2563eb")
            cell.set_facecolor(color)
            cell.set_text_props(color="white", weight="bold")

    fig.tight_layout()
    out = OUT_DIR / "sample_real_ftir_clean_numbered_table.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out, selected


def render_numbered_table_no_confidence(x, y, preds):
    selected = merge_close_predictions(preds, min_gap=65)

    fig = plt.figure(figsize=(13.7, 7.2), dpi=180)
    gs = fig.add_gridspec(1, 2, width_ratios=[3.35, 1.05], wspace=0.04)
    ax = fig.add_subplot(gs[0, 0])
    tax = fig.add_subplot(gs[0, 1])
    base_axis(ax, x, y)

    for idx, pred in enumerate(selected, start=1):
        color = COLORS.get(pred["label"], "#2563eb")
        yy = transmittance_at(pred["wn"], x, y)
        ax.vlines(pred["wn"], ymin=47, ymax=yy, color=color, linewidth=1.35, alpha=0.82)
        ax.scatter([pred["wn"]], [yy], s=26, color=color, edgecolor="white", linewidth=0.7, zorder=5)
        ax.text(
            pred["wn"],
            103.0 - (idx % 3) * 2.2,
            str(idx),
            ha="center",
            va="center",
            fontsize=7.6,
            color="white",
            bbox=dict(boxstyle="circle,pad=0.23", fc=color, ec="white", lw=0.7),
        )

    ax.set_title("Real FTIR Spectrum - Numbered YOLOv8n Predictions", fontweight="bold")
    tax.axis("off")
    rows = [[str(i), p["label"], f"{p['wn']:.0f}"] for i, p in enumerate(selected, start=1)]
    table = tax.table(
        cellText=rows,
        colLabels=["#", "Predicted band", "cm$^{-1}$"],
        colLoc="left",
        cellLoc="left",
        loc="center",
        colWidths=[0.14, 0.60, 0.26],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7.6)
    table.scale(1.0, 1.32)
    for (row, col), cell in table.get_celld().items():
        cell.set_linewidth(0.35)
        if row == 0:
            cell.set_facecolor("#f3f4f6")
            cell.set_text_props(weight="bold")
        elif col == 0:
            color = COLORS.get(selected[row - 1]["label"], "#2563eb")
            cell.set_facecolor(color)
            cell.set_text_props(color="white", weight="bold")

    fig.tight_layout()
    out = OUT_DIR / "sample_real_ftir_clean_numbered_table_no_confidence.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out, selected


def main():
    x, y = read_curve()
    preds = read_predictions()
    high_out, high_selected = render_high_confidence(x, y, preds)
    table_out, table_selected = render_numbered_table(x, y, preds)
    table_no_conf_out, table_no_conf_selected = render_numbered_table_no_confidence(x, y, preds)
    print(f"high_confidence={high_out}")
    print(f"high_confidence_count={len(high_selected)}")
    print(f"numbered_table={table_out}")
    print(f"numbered_table_count={len(table_selected)}")
    print(f"numbered_table_no_confidence={table_no_conf_out}")
    print(f"numbered_table_no_confidence_count={len(table_no_conf_selected)}")


if __name__ == "__main__":
    main()

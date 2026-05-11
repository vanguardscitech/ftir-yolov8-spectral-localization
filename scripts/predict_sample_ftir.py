from pathlib import Path
import csv
import os

import cv2


ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / "Ultralytics"))

from ultralytics import YOLO

MODEL_PATH = ROOT / "models" / "best.pt"
IMAGE_PATH = ROOT / "outputs" / "real_spectrum_validation" / "sample_real_ftir_standardized.png"
OUT_DIR = ROOT / "outputs" / "real_spectrum_validation"
CONF_THRESHOLD = 0.15


def estimate_wavenumber(center_x: float, width: int) -> float:
    """Approximate mapping for the plotted sample image, assuming 4000 -> 400 cm-1."""
    plot_left = int(width * 0.071)
    plot_right = int(width * 0.951)
    x = min(max(center_x, plot_left), plot_right)
    fraction = (x - plot_left) / max(plot_right - plot_left, 1)
    return 4000.0 + fraction * (400.0 - 4000.0)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(MODEL_PATH))
    results = model.predict(str(IMAGE_PATH), conf=CONF_THRESHOLD, verbose=False)
    result = results[0]

    img = cv2.imread(str(IMAGE_PATH))
    if img is None:
        raise FileNotFoundError(f"Cannot read {IMAGE_PATH}")
    h, w = img.shape[:2]

    detections = []
    for box in result.boxes:
        x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].tolist()]
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        cx = (x1 + x2) / 2.0
        label = result.names[cls_id]
        detections.append(
            {
                "label": label,
                "confidence": conf,
                "center_x_px": cx,
                "approx_wavenumber_cm-1": estimate_wavenumber(cx, w),
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
            }
        )

    detections.sort(key=lambda item: item["center_x_px"])

    csv_path = OUT_DIR / "sample_real_ftir_standardized_predictions.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "label",
                "confidence",
                "center_x_px",
                "approx_wavenumber_cm-1",
                "x1",
                "y1",
                "x2",
                "y2",
            ],
        )
        writer.writeheader()
        writer.writerows(detections)

    colors = {
        "O-H Alcohol": (28, 115, 255),
        "N-H Amine": (140, 70, 255),
        "C-H Alkane": (0, 180, 0),
        "C-H Aldehyde": (0, 150, 210),
        "S-H Thiol": (40, 170, 170),
        "C-N Nitrile": (220, 80, 40),
        "C-C Alkyne": (190, 60, 120),
        "C=O Carbonyl": (0, 0, 220),
        "C=C Alkene": (200, 90, 0),
        "N-O Nitro": (120, 120, 0),
        "C-O Ether": (0, 120, 120),
    }

    for idx, item in enumerate(detections):
        cx = int(round(item["center_x_px"]))
        color = colors.get(item["label"], (0, 128, 0))
        text_y = 44 + (idx % 3) * 28
        cv2.line(img, (cx, 35), (cx, h - 44), color, 2, cv2.LINE_AA)
        label = f"{item['label']} {item['confidence']:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.43, 1)
        x0 = max(4, min(cx - tw // 2, w - tw - 8))
        cv2.rectangle(img, (x0 - 3, text_y - th - 5), (x0 + tw + 3, text_y + 5), (255, 255, 255), -1)
        cv2.putText(img, label, (x0, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (0, 0, 0), 1, cv2.LINE_AA)

    out_img_path = OUT_DIR / "sample_real_ftir_standardized_yolo_prediction.png"
    cv2.imwrite(str(out_img_path), img)

    print(f"detections={len(detections)}")
    print(f"csv={csv_path}")
    print(f"image={out_img_path}")
    for item in detections:
        print(
            f"{item['label']}\tconf={item['confidence']:.3f}\t"
            f"approx_cm-1={item['approx_wavenumber_cm-1']:.1f}"
        )


if __name__ == "__main__":
    main()

# YOLOv8-Based FTIR Spectral Band Localization

This repository contains the code, model artifacts, generated FTIR dataset resources, and representative outputs for:

**An Efficient Deep Learning Approach for Automated FTIR Spectral Interpretation and Simultaneous Molecular Bond Localization**

The project reformulates FTIR spectral interpretation as an object-detection problem. A YOLOv8n detector is trained to localize and classify representative absorption-band manifestations in FTIR spectral plots.

Repository URL: https://github.com/vanguardscitech/ftir-yolov8-spectral-localization

## Contents

- `scripts/`: data augmentation, inference, and clean figure-rendering scripts.
- `data/FTIR_Dataset_Base/`: 200 base synthetic FTIR spectra with YOLO-format labels.
- `data/data_augmented_1000.yaml`: YOLO dataset configuration for the 1000-image augmented dataset.
- `models/best.pt`: trained YOLOv8n model weights.
- `outputs/`: representative prediction figures and CSV outputs.
- `sample.png`: real FTIR spectrum used as a preliminary qualitative workflow demonstration.

## Dataset

The manuscript uses 200 base synthetic FTIR spectra expanded fivefold through geometry-preserving augmentation, yielding:

- 800 training images
- 200 validation images
- 1000 total image-level examples

The base dataset is included in `data/FTIR_Dataset_Base/`. The 1000-image dataset can be regenerated with:

```bash
python scripts/create_augmented_ftir_1000.py
```

The augmentation preserves label geometry and includes Gaussian noise, brightness/contrast variation, mild blur, and simulated baseline drift.

## Classes

The final 11 YOLO classes are:

0. O-H Alcohol
1. N-H Amine
2. C-H Alkane
3. C-H Aldehyde
4. S-H Thiol
5. C-N Nitrile
6. C-C Alkyne
7. C=O Carbonyl
8. C=C Alkene
9. N-O Nitro
10. C-O Ether

## Model Inference

To run the trained model on the real FTIR sample:

```bash
python scripts/standardize_sample_ftir.py
python scripts/predict_sample_ftir.py
python scripts/render_clean_real_spectrum_figures.py
```

To reproduce the clean Figure 9 prediction:

```bash
python scripts/render_clean_figure9_prediction.py
```

## Requirements

Install dependencies with:

```bash
pip install -r requirements.txt
```

Tested with Python 3.10 and Ultralytics YOLOv8.

## Notes on Real-Spectrum Demonstration

The real FTIR spectrum in `sample.png` corresponds to a wound dressing containing *Liquidambar orientalis* resin. It is included as a qualitative workflow demonstration and domain-shift check. It is not included in the quantitative validation metrics because independent expert region-level annotations are not yet available.

## Citation

If this repository supports your work, please cite the associated manuscript once published.

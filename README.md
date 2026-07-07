# Brain Tumor Segmentation with a Deep-Learning CNN

Multi-class brain-tumor segmentation on the **BraTS** MRI dataset using a custom
TensorFlow/Keras U-Net with **Twin Squeeze-and-Excitation Attention** (`BC-TSEA-UNet`).
The network takes the four standard MRI modalities and predicts the three clinical
tumor sub-regions per slice.

- **Inputs (4 channels):** T1, T2, T1ce, FLAIR — each slice resized to `160 × 160`
- **Outputs (3 channels):** ET (Enhancing Tumor), TC (Tumor Core), WT (Whole Tumor)
- **Evaluation metrics:** Dice, IoU, Sensitivity, Specificity, HD95, ASSD

## Model

A U-Net encoder–decoder built in [`src/train_generator.py`](src/train_generator.py):

- **Squeeze-and-Excitation (SE) blocks** recalibrate channel responses; a "twin SE"
  block applies two BN → ReLU → SE stages in sequence at every level.
- **Encoder:** four `conv_block` (double 3×3 conv + BN + ReLU) stages with filters
  32 → 64 → 128 → 256, each followed by a twin-SE block and max-pooling.
- **Bottleneck:** 512-filter conv block with an SE block.
- **Decoder:** upsampling with skip connections back to the matching encoder stage,
  mirrored twin-SE blocks, and a final `1×1` conv with sigmoid activation (3 outputs).

### Custom loss — `hybrid_boundary_et_loss`

The tumor sub-regions are heavily imbalanced (ET is tiny), so the loss combines
overlap and boundary terms, with extra weight on the enhancing tumor:

```
0.35 · BCE
0.20 · Dice (all regions)
0.20 · Dice (ET only)
0.15 · Boundary loss (Sobel-edge, all regions)
0.10 · Boundary loss (ET only)
```

Trained with Adam (`lr=5e-4`), `ModelCheckpoint` on `val_loss`, and early stopping
(patience 3). ET-rich slices are oversampled during training to counter class imbalance.

## Project structure

```
.
├── src/                       # pipeline scripts
│   ├── check_files.py         # inspect the raw dataset folder
│   ├── test_dataset.py        # list patients / files in the dataset
│   ├── preprocess_and_cache.py# NIfTI → normalized 160×160 slices, cached as .npz
│   ├── train_generator.py     # build & train the BC-TSEA-UNet
│   ├── predict_and_evaluate.py# run inference, compute Dice/IoU/HD95/ASSD → CSV
│   ├── multi_slice_evaluation.py # best / average / worst-case slice analysis
│   ├── visualize_results.py   # qualitative multi-modal overlays of GT vs. prediction
│   └── plot_all_graphs.py     # metric / training-curve plots
├── results/
│   ├── figures/               # qualitative segmentation figures
│   ├── advanced_visuals/      # per-sample multi-modal overlays
│   ├── multi_analysis/        # best / average / worst case comparisons
│   └── plots/                 # accuracy, loss, Dice/IoU/HD95/ASSD bar charts
├── requirements.txt
└── README.md
```

> **Note:** `data/` (raw BraTS + `cache_npz/`) and `results/models/` are **not**
> committed — the `.gitignore` excludes model weights (`*.keras`, `*.h5`), cached
> arrays (`*.npz`, `*.npy`), and datasets. You supply your own BraTS data locally.

## Data layout

Place the BraTS training data under `data/`. Each patient folder contains the four
modality volumes and the segmentation label as NIfTI files:

```
data/BraTS_Dataset/training_data1_v2/<patient_id>/
    *_t1.nii(.gz)  *_t2.nii(.gz)  *_t1ce.nii(.gz)  *_flair.nii(.gz)  *_seg.nii(.gz)
```

## Setup

```bash
python -m pip install -r requirements.txt
```

The pipeline uses **TensorFlow/Keras** (model), plus **SimpleITK**, **OpenCV**,
**nibabel**, and **SciPy** for preprocessing and metrics. Make sure these are
installed (see [`requirements.txt`](requirements.txt)).

## Usage

Run the scripts from the project root, in order:

```bash
# 0. (optional) sanity-check the dataset folder
python src/check_files.py
python src/test_dataset.py

# 1. Preprocess raw NIfTI volumes into cached .npz slices
python src/preprocess_and_cache.py

# 2. Train the BC-TSEA-UNet (saves best_model.keras + training history)
python src/train_generator.py

# 3. Run inference on the held-out test split and compute metrics
python src/predict_and_evaluate.py       # → results/metrics/final_metrics.csv

# 4. Analyze and visualize results
python src/multi_slice_evaluation.py     # best/average/worst case slices
python src/visualize_results.py          # multi-modal GT vs. prediction overlays
python src/plot_all_graphs.py            # metric & training-curve plots
```

Data is split **80 / 10 / 10** (train / val / test) at the patient level with a fixed
seed (`random_state=42`) so patients never leak across splits.

## Results

Sample qualitative outputs (in [`results/`](results/)):

| Whole Tumor | Tumor Core | Enhancing Tumor |
|:---:|:---:|:---:|
| ![WT](results/figures/wt_result.png) | ![TC](results/figures/tc_result.png) | ![ET](results/figures/et_result.png) |

Best / average / worst-case slices are saved under `results/multi_analysis/`, and
Dice / IoU / HD95 / ASSD bar charts under `results/plots/`.

## Preprocessing details

`preprocess_and_cache.py` performs, per patient:

- z-score intensity normalization of each modality
- resize to `160 × 160` (configurable via `IMG_SIZE`)
- optional N4 bias-field correction (`USE_N4`, off by default for speed) and
  histogram matching via SimpleITK
- stacks the 4 modalities and 3 label channels, caching each patient as a single
  `.npz` file for fast loading during training

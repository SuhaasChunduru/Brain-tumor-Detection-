"""Shared paths, shapes, and constants for the whole pipeline.

Every script resolves paths relative to the repository root, so run them from
there (see README).
"""

# Raw BraTS volumes: one sub-directory per patient.
DATASET_PATH = "data/BraTS_Dataset/training_data1_v2"

# Preprocessed slices: one .npz per patient, keys "X" (slices) and "Y" (masks).
CACHE_DIR = "data/cache_npz"

MODEL_DIR = "results/models"
METRICS_DIR = "results/metrics"
PLOTS_DIR = "results/plots"

IMG_SIZE = 160
BATCH_SIZE = 4

# Input channels, in the order preprocess.py stacks them.
MODALITY_NAMES = ["T1", "T1ce", "T2", "FLAIR"]

# Output channels, in the order preprocess.py builds them.
REGION_NAMES = ["ET", "TC", "WT"]

N_MODALITIES = len(MODALITY_NAMES)
N_REGIONS = len(REGION_NAMES)

# Fixed seed for the patient-level split, so train/val/test membership is
# identical across preprocessing, training, and evaluation runs.
SPLIT_SEED = 42

"""Convert raw BraTS NIfTI volumes into cached 2D slice arrays.

Per patient: normalize each modality, harmonize intensities, resize to
IMG_SIZE, keep only tumour-bearing slices, and write one .npz holding
X (N, IMG_SIZE, IMG_SIZE, 4) and Y (N, IMG_SIZE, IMG_SIZE, 3).

Run:  python -m src.data.preprocess
"""

import os

import cv2
import nibabel as nib
import numpy as np
import SimpleITK as sitk

from src.config import CACHE_DIR, DATASET_PATH, IMG_SIZE

# N4 is slow and made little difference on this data; histogram matching alone
# handled most of the cross-scanner intensity drift.
USE_N4 = False

# Label values in the segmentation volume. BraTS 2021 and earlier encode
# enhancing tumour as 4; BraTS 2023 re-encodes it as 3. Set LABEL_ET to match
# the release you downloaded, or the ET channel will come out empty.
LABEL_NCR = 1  # necrotic / non-enhancing core
LABEL_ED = 2   # peritumoural edema
LABEL_ET = 4   # enhancing tumour


def load_nii(path):
    return nib.load(path).get_fdata()


def find_file(files, tag):
    matches = [f for f in files if f.endswith(tag)]
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected one file ending with {tag}, got {matches}")
    return matches[0]


def zscore(vol):
    vol = vol.astype(np.float32)
    return (vol - vol.mean()) / (vol.std() + 1e-8)


def n4_bias_correction_fast(vol_np):
    """N4 bias-field correction with a shortened iteration schedule."""
    img = sitk.GetImageFromArray(vol_np.astype(np.float32))
    mask = sitk.OtsuThreshold(img, 0, 1, 200)
    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    corrector.SetMaximumNumberOfIterations([10, 5, 3, 2])
    return sitk.GetArrayFromImage(corrector.Execute(img, mask))


def histogram_match(source_np, ref_np):
    """Match a modality's intensity histogram onto a reference modality."""
    source = sitk.GetImageFromArray(source_np.astype(np.float32))
    ref = sitk.GetImageFromArray(ref_np.astype(np.float32))
    matcher = sitk.HistogramMatchingImageFilter()
    matcher.SetNumberOfHistogramLevels(128)
    matcher.SetNumberOfMatchPoints(10)
    matcher.ThresholdAtMeanIntensityOn()
    return sitk.GetArrayFromImage(matcher.Execute(source, ref))


def build_region_masks(seg_slice):
    """Turn a label slice into the three nested BraTS evaluation regions."""
    et = seg_slice == LABEL_ET
    tc = (seg_slice == LABEL_NCR) | (seg_slice == LABEL_ET)
    wt = (seg_slice == LABEL_NCR) | (seg_slice == LABEL_ED) | (seg_slice == LABEL_ET)

    resized = [
        cv2.resize(r.astype(np.float32), (IMG_SIZE, IMG_SIZE),
                   interpolation=cv2.INTER_NEAREST)
        for r in (et, tc, wt)
    ]
    return np.stack(resized, axis=-1)


def process_patient(ppath):
    """Return (X, Y) arrays of tumour-bearing slices for one patient folder."""
    files = [f for f in os.listdir(ppath) if not f.startswith(".")]

    t1 = load_nii(os.path.join(ppath, find_file(files, "-t1n.nii.gz")))
    t1ce = load_nii(os.path.join(ppath, find_file(files, "-t1c.nii.gz")))
    t2 = load_nii(os.path.join(ppath, find_file(files, "-t2w.nii.gz")))
    flair = load_nii(os.path.join(ppath, find_file(files, "-t2f.nii.gz")))
    seg = load_nii(os.path.join(ppath, find_file(files, "-seg.nii.gz")))

    t1, t1ce, t2, flair = zscore(t1), zscore(t1ce), zscore(t2), zscore(flair)

    if USE_N4:
        t1 = n4_bias_correction_fast(t1)
        t1ce = n4_bias_correction_fast(t1ce)
        t2 = n4_bias_correction_fast(t2)
        flair = n4_bias_correction_fast(flair)

    # T1 is the reference; the other three are matched onto it so a channel
    # means the same thing across patients and scanners.
    ref = t1
    t1ce = histogram_match(t1ce, ref)
    t2 = histogram_match(t2, ref)
    flair = histogram_match(flair, ref)

    # Channel order here defines MODALITY_NAMES in src/config.py.
    combined = np.stack([ref, t1ce, t2, flair], axis=-1).astype(np.float32)

    X_list, Y_list = [], []
    for z in range(combined.shape[2]):
        m = seg[:, :, z]

        # Slices with no annotated tumour are dropped entirely.
        if np.sum(m) == 0:
            continue

        X_list.append(cv2.resize(combined[:, :, z, :], (IMG_SIZE, IMG_SIZE)))
        Y_list.append(build_region_masks(m))

    return np.array(X_list, dtype=np.float32), np.array(Y_list, dtype=np.float32)


def main():
    os.makedirs(CACHE_DIR, exist_ok=True)

    patients = sorted(
        p for p in os.listdir(DATASET_PATH)
        if os.path.isdir(os.path.join(DATASET_PATH, p)) and not p.startswith(".")
    )
    print("Total patients found:", len(patients))

    for i, pid in enumerate(patients):
        save_path = os.path.join(CACHE_DIR, f"{pid}.npz")

        if os.path.exists(save_path):
            print(f"[{i + 1}/{len(patients)}] Skipping cached: {pid}")
            continue

        try:
            Xp, Yp = process_patient(os.path.join(DATASET_PATH, pid))
            np.savez_compressed(save_path, X=Xp, Y=Yp)
            print(f"[{i + 1}/{len(patients)}] Saved {pid} | slices={len(Xp)}")
        except Exception as e:
            print(f"[{i + 1}/{len(patients)}] Error in {pid}: {e}")

    print("Preprocessing and caching complete.")


if __name__ == "__main__":
    main()

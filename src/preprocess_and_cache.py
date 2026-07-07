import os
import numpy as np
import nibabel as nib
import cv2
import SimpleITK as sitk

# Real dataset path
dataset_path = "data/BraTS_Dataset/training_data1_v2"
cache_dir = "data/cache_npz"
os.makedirs(cache_dir, exist_ok=True)

# Increased resolution for better ET visibility
IMG_SIZE = 160
USE_N4 = False   # keep False first for speed on Mac

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
    img = sitk.GetImageFromArray(vol_np.astype(np.float32))
    mask = sitk.OtsuThreshold(img, 0, 1, 200)
    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    corrector.SetMaximumNumberOfIterations([10, 5, 3, 2])
    corrected = corrector.Execute(img, mask)
    return sitk.GetArrayFromImage(corrected)

def histogram_match(source_np, ref_np):
    source = sitk.GetImageFromArray(source_np.astype(np.float32))
    ref = sitk.GetImageFromArray(ref_np.astype(np.float32))
    matcher = sitk.HistogramMatchingImageFilter()
    matcher.SetNumberOfHistogramLevels(128)
    matcher.SetNumberOfMatchPoints(10)
    matcher.ThresholdAtMeanIntensityOn()
    out = matcher.Execute(source, ref)
    return sitk.GetArrayFromImage(out)

# Choose patient folders only
patients = sorted([
    p for p in os.listdir(dataset_path)
    if os.path.isdir(os.path.join(dataset_path, p)) and not p.startswith(".")
])

# Optional: use a subset first if needed
# patients = patients[:20]
# patients = patients[:50]

print("Total patients found:", len(patients))

for i, pid in enumerate(patients):
    save_path = os.path.join(cache_dir, f"{pid}.npz")

    if os.path.exists(save_path):
        print(f"[{i+1}/{len(patients)}] Skipping cached: {pid}")
        continue

    try:
        ppath = os.path.join(dataset_path, pid)
        files = [f for f in os.listdir(ppath) if not f.startswith(".")]

        t1 = load_nii(os.path.join(ppath, find_file(files, "-t1n.nii.gz")))
        t1ce = load_nii(os.path.join(ppath, find_file(files, "-t1c.nii.gz")))
        t2 = load_nii(os.path.join(ppath, find_file(files, "-t2w.nii.gz")))
        flair = load_nii(os.path.join(ppath, find_file(files, "-t2f.nii.gz")))
        seg = load_nii(os.path.join(ppath, find_file(files, "-seg.nii.gz")))

        # Normalize
        t1, t1ce, t2, flair = zscore(t1), zscore(t1ce), zscore(t2), zscore(flair)

        # Optional N4
        if USE_N4:
            t1 = n4_bias_correction_fast(t1)
            t1ce = n4_bias_correction_fast(t1ce)
            t2 = n4_bias_correction_fast(t2)
            flair = n4_bias_correction_fast(flair)

        # Histogram matching
        ref = t1
        t1ce = histogram_match(t1ce, ref)
        t2 = histogram_match(t2, ref)
        flair = histogram_match(flair, ref)

        # Stack 4 modalities
        combined = np.stack([ref, t1ce, t2, flair], axis=-1).astype(np.float32)

        X_list, Y_list = [], []

        for z in range(combined.shape[2]):
            m = seg[:, :, z]

            # Slice selection: keep only tumor slices
            if np.sum(m) == 0:
                continue

            img2d = cv2.resize(combined[:, :, z, :], (IMG_SIZE, IMG_SIZE))

            # Multi-region masks
            et = (m == 4)
            tc = (m == 1) | (m == 4)
            wt = (m == 1) | (m == 2) | (m == 4)

            et = cv2.resize(et.astype(np.float32), (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST)
            tc = cv2.resize(tc.astype(np.float32), (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST)
            wt = cv2.resize(wt.astype(np.float32), (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST)

            mask3 = np.stack([et, tc, wt], axis=-1)

            X_list.append(img2d)
            Y_list.append(mask3)

        Xp = np.array(X_list, dtype=np.float32)
        Yp = np.array(Y_list, dtype=np.float32)

        np.savez_compressed(save_path, X=Xp, Y=Yp)

        print(f"[{i+1}/{len(patients)}] Saved {pid} | slices={len(Xp)}")

    except Exception as e:
        print(f"[{i+1}/{len(patients)}] Error in {pid}: {e}")

print("✅ Preprocessing and caching complete.")
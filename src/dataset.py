"""Patient-level splits and tf.data input pipelines over the cached .npz slices.

Training and evaluation both import from here so they cannot drift apart on the
split definition.
"""

import os

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

from src.config import BATCH_SIZE, CACHE_DIR, IMG_SIZE, N_MODALITIES, N_REGIONS, SPLIT_SEED

OUTPUT_SIGNATURE = (
    tf.TensorSpec(shape=(IMG_SIZE, IMG_SIZE, N_MODALITIES), dtype=tf.float32),
    tf.TensorSpec(shape=(IMG_SIZE, IMG_SIZE, N_REGIONS), dtype=tf.float32),
)


def list_cached_patients(cache_dir=CACHE_DIR):
    """Sorted list of cached patient .npz paths."""
    if not os.path.isdir(cache_dir):
        raise FileNotFoundError(
            f"No cache at {cache_dir!r}. Run `python -m src.data.preprocess` first."
        )
    files = sorted(
        os.path.join(cache_dir, f) for f in os.listdir(cache_dir) if f.endswith(".npz")
    )
    if not files:
        raise FileNotFoundError(f"{cache_dir!r} contains no .npz files.")
    return files


def patient_splits(cache_dir=CACHE_DIR):
    """Split patients 80/10/10 into (train, val, test).

    The split is at the patient level, so slices from one patient never appear
    in more than one split.
    """
    all_files = list_cached_patients(cache_dir)
    train_files, temp_files = train_test_split(
        all_files, test_size=0.2, random_state=SPLIT_SEED
    )
    val_files, test_files = train_test_split(
        temp_files, test_size=0.5, random_state=SPLIT_SEED
    )
    return train_files, val_files, test_files


def count_slices(file_list):
    """Total number of cached slices across the given patients."""
    return sum(len(np.load(path)["X"]) for path in file_list)


def npz_generator(file_list, oversample_et=False):
    """Yield (slice, mask) pairs one patient file at a time.

    With `oversample_et`, slices holding more than 30 enhancing-tumor pixels are
    yielded twice. ET is by far the smallest region, so this roughly doubles its
    weight in a training epoch without changing the loss.
    """
    for path in file_list:
        data = np.load(path)
        X, Y = data["X"], data["Y"]
        for i in range(len(X)):
            yield X[i], Y[i]
            if oversample_et and np.sum(Y[i][:, :, 0]) > 30:
                yield X[i], Y[i]


def make_dataset(file_list, batch_size=BATCH_SIZE, shuffle=False, repeat=False,
                 oversample_et=False):
    """Build a batched tf.data.Dataset over the given patient files."""
    ds = tf.data.Dataset.from_generator(
        lambda: npz_generator(file_list, oversample_et=oversample_et),
        output_signature=OUTPUT_SIGNATURE,
    )
    if shuffle:
        ds = ds.shuffle(2000)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    if repeat:
        ds = ds.repeat()
    return ds

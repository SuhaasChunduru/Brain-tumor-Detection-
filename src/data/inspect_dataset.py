"""Sanity-check the raw dataset folder before preprocessing.

Prints how many patient folders were found and what files the first one holds,
so a missing modality or an unexpected naming scheme shows up immediately.

Run:  python -m src.data.inspect_dataset
"""

import os

from src.config import DATASET_PATH


def main():
    if not os.path.isdir(DATASET_PATH):
        print(f"Dataset path does not exist: {DATASET_PATH}")
        print("Point DATASET_PATH in src/config.py at your BraTS download.")
        return

    patients = sorted(
        p for p in os.listdir(DATASET_PATH)
        if os.path.isdir(os.path.join(DATASET_PATH, p)) and not p.startswith(".")
    )

    print("Dataset path:", DATASET_PATH)
    print("Total patient folders:", len(patients))

    if not patients:
        print("No patient folders found. Check the path and the archive layout.")
        return

    first = os.path.join(DATASET_PATH, patients[0])
    print("First patient:", patients[0])
    print("Files:", sorted(os.listdir(first)))


if __name__ == "__main__":
    main()

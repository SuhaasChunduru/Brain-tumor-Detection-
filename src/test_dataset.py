import os

dataset_path = "data/BraTS_Dataset"

patients = sorted([
    p for p in os.listdir(dataset_path)
    if os.path.isdir(os.path.join(dataset_path, p))
])

print("Total patient folders:", len(patients))
print("First patient:", patients[0])
print("Files in first patient:", os.listdir(os.path.join(dataset_path, patients[0])))
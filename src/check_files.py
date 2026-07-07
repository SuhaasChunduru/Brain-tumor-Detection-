import os

dataset_path = "data/BraTS_Dataset"

patients = [
    p for p in os.listdir(dataset_path)
    if os.path.isdir(os.path.join(dataset_path, p))
]

print("Folders found:", patients)

if len(patients) == 0:
    print("No folders found inside data/BraTS_Dataset")
else:
    first = os.path.join(dataset_path, patients[0])
    print("\nInside first folder:")
    print(os.listdir(first))
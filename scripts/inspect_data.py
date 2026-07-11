from pathlib import Path

import numpy as np 
from aeon.datasets import load_classification


DATA_DIR = Path("data/raw")

X_train, y_train, metadata = load_classification(
    name = "FaceDetection",
    split = "train",
    extract_path = DATA_DIR,
    return_metadata = True,
)

labels, counts = np.unique(y_train, return_counts = True)

print("X type:", type(X_train))
print("X shape:", X_train.shape)
print("X dtype:", X_train.dtype)

print("\ny type:", type(y_train))
print("y shape:", y_train.shape)
print("Class counts:", dict(zip(labels, counts)))

print("\nMetadata:", metadata)

X_test, y_test, test_metadata = load_classification(
    name = "FaceDetection",
    split = "test",
    extract_path = DATA_DIR,
    return_metadata= True,
)

test_labels, test_counts = np.unique(y_test, return_counts = True)

print("\n--- Test split ---")
print("X_test shape:", X_test.shape)
print("X_test dtype:", X_test.dtype)
print("y_test shape:", y_test.shape)
print("Test class counts:", dict(zip(test_labels, test_counts)))
print("Test metadata:", test_metadata)

assert X_train.shape[1:] == X_test.shape[1:], "Train/test channel-time shapes differ."
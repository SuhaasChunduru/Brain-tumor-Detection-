"""Train BC-TSEA-UNet on the cached BraTS slices.

Writes best_model.keras (best val_loss), the final weights, and the training
history arrays under results/models/.

Run:  python -m src.train
"""

import math
import os

import numpy as np
import tensorflow as tf

from src.config import BATCH_SIZE, MODEL_DIR
from src.dataset import count_slices, make_dataset, patient_splits
from src.models.losses import hybrid_boundary_et_loss
from src.models.unet import build_model

EPOCHS = 12
LEARNING_RATE = 5e-4
EARLY_STOPPING_PATIENCE = 3


def main():
    train_files, val_files, test_files = patient_splits()
    print("Train patients:", len(train_files))
    print("Val patients:", len(val_files))
    print("Test patients:", len(test_files))

    train_count = count_slices(train_files)
    val_count = count_slices(val_files)
    print("Train slices:", train_count)
    print("Val slices:", val_count)

    # The datasets repeat indefinitely, so Keras needs explicit step counts.
    steps_per_epoch = math.ceil(train_count / BATCH_SIZE)
    validation_steps = math.ceil(val_count / BATCH_SIZE)
    print("Steps per epoch:", steps_per_epoch)
    print("Validation steps:", validation_steps)

    train_ds = make_dataset(
        train_files, shuffle=True, repeat=True, oversample_et=True
    )
    val_ds = make_dataset(val_files, repeat=True)

    model = build_model()
    model.compile(
        optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=LEARNING_RATE),
        loss=hybrid_boundary_et_loss,
        metrics=["accuracy"],
    )

    os.makedirs(MODEL_DIR, exist_ok=True)
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            os.path.join(MODEL_DIR, "best_model.keras"),
            save_best_only=True,
            monitor="val_loss",
            mode="min",
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
        ),
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        callbacks=callbacks,
        verbose=1,
    )

    model.save(os.path.join(MODEL_DIR, "bc_tsea_unet.keras"))
    for key, fname in [
        ("loss", "train_loss.npy"),
        ("val_loss", "val_loss.npy"),
        ("accuracy", "train_acc.npy"),
        ("val_accuracy", "val_acc.npy"),
    ]:
        np.save(os.path.join(MODEL_DIR, fname), np.array(history.history[key]))

    print(f"Model and training history saved to {MODEL_DIR}/")


if __name__ == "__main__":
    main()

import os
import math
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models

CACHE_DIR = "data/cache_npz"
BATCH_SIZE = 4
IMG_SIZE = 160
EPOCHS = 12

all_files = sorted([
    os.path.join(CACHE_DIR, f)
    for f in os.listdir(CACHE_DIR)
    if f.endswith(".npz")
])

print("Total cached patients:", len(all_files))

train_files, temp_files = train_test_split(all_files, test_size=0.2, random_state=42)
val_files, test_files = train_test_split(temp_files, test_size=0.5, random_state=42)

print("Train patients:", len(train_files))
print("Val patients:", len(val_files))
print("Test patients:", len(test_files))

def count_slices(file_list):
    total = 0
    for path in file_list:
        data = np.load(path)
        total += len(data["X"])
    return total

train_count = count_slices(train_files)
val_count = count_slices(val_files)
test_count = count_slices(test_files)

print("Train slices:", train_count)
print("Val slices:", val_count)
print("Test slices:", test_count)

steps_per_epoch = math.ceil(train_count / BATCH_SIZE)
validation_steps = math.ceil(val_count / BATCH_SIZE)

print("Steps per epoch:", steps_per_epoch)
print("Validation steps:", validation_steps)

# ET-rich hard sample oversampling
def npz_generator(file_list, oversample_et=False):
    for path in file_list:
        data = np.load(path)
        X = data["X"]
        Y = data["Y"]

        for i in range(len(X)):
            x = X[i]
            y = Y[i]

            yield x, y

            if oversample_et:
                et_pixels = np.sum(y[:, :, 0])  # ET channel
                if et_pixels > 30:
                    yield x, y

output_signature = (
    tf.TensorSpec(shape=(IMG_SIZE, IMG_SIZE, 4), dtype=tf.float32),
    tf.TensorSpec(shape=(IMG_SIZE, IMG_SIZE, 3), dtype=tf.float32),
)

train_ds = tf.data.Dataset.from_generator(
    lambda: npz_generator(train_files, oversample_et=True),
    output_signature=output_signature
).shuffle(2000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE).repeat()

val_ds = tf.data.Dataset.from_generator(
    lambda: npz_generator(val_files, oversample_et=False),
    output_signature=output_signature
).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE).repeat()

# -----------------------------
# Model blocks
# -----------------------------
def se_block(x, reduction=16):
    ch = x.shape[-1]
    s = layers.GlobalAveragePooling2D()(x)
    s = layers.Dense(max(ch // reduction, 1), activation="relu")(s)
    s = layers.Dense(ch, activation="sigmoid")(s)
    s = layers.Reshape((1, 1, ch))(s)
    return layers.Multiply()([x, s])

def twin_se_block(x):
    y = layers.BatchNormalization()(x)
    y = layers.Activation("relu")(y)
    y = se_block(y)

    y = layers.BatchNormalization()(y)
    y = layers.Activation("relu")(y)
    y = se_block(y)
    return y

def conv_block(x, filters):
    x = layers.Conv2D(filters, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.Conv2D(filters, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    return x

def build_model(input_shape=(160, 160, 4)):
    inputs = layers.Input(input_shape)

    c1 = conv_block(inputs, 32)
    s1 = twin_se_block(c1)
    p1 = layers.MaxPooling2D()(s1)

    c2 = conv_block(p1, 64)
    s2 = twin_se_block(c2)
    p2 = layers.MaxPooling2D()(s2)

    c3 = conv_block(p2, 128)
    s3 = twin_se_block(c3)
    p3 = layers.MaxPooling2D()(s3)

    c4 = conv_block(p3, 256)
    s4 = twin_se_block(c4)
    p4 = layers.MaxPooling2D()(s4)

    b = conv_block(p4, 512)
    b = se_block(b)

    u4 = layers.UpSampling2D()(b)
    u4 = layers.Concatenate()([u4, s4])
    d4 = twin_se_block(conv_block(u4, 256))

    u3 = layers.UpSampling2D()(d4)
    u3 = layers.Concatenate()([u3, s3])
    d3 = twin_se_block(conv_block(u3, 128))

    u2 = layers.UpSampling2D()(d3)
    u2 = layers.Concatenate()([u2, s2])
    d2 = twin_se_block(conv_block(u2, 64))

    u1 = layers.UpSampling2D()(d2)
    u1 = layers.Concatenate()([u1, s1])
    d1 = twin_se_block(conv_block(u1, 32))

    outputs = layers.Conv2D(3, 1, activation="sigmoid")(d1)
    return models.Model(inputs, outputs)

# -----------------------------
# Loss functions
# -----------------------------
def dice_loss(y_true, y_pred, smooth=1e-6):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    intersection = tf.reduce_sum(y_true * y_pred, axis=[1, 2, 3])
    denom = tf.reduce_sum(y_true, axis=[1, 2, 3]) + tf.reduce_sum(y_pred, axis=[1, 2, 3])

    dice = (2.0 * intersection + smooth) / (denom + smooth)
    return 1.0 - tf.reduce_mean(dice)

def et_dice_loss(y_true, y_pred, smooth=1e-6):
    y_true_et = tf.cast(y_true[..., 0:1], tf.float32)
    y_pred_et = tf.cast(y_pred[..., 0:1], tf.float32)

    intersection = tf.reduce_sum(y_true_et * y_pred_et, axis=[1, 2, 3])
    denom = tf.reduce_sum(y_true_et, axis=[1, 2, 3]) + tf.reduce_sum(y_pred_et, axis=[1, 2, 3])

    dice = (2.0 * intersection + smooth) / (denom + smooth)
    return 1.0 - tf.reduce_mean(dice)

def boundary_loss(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    true_edges = tf.image.sobel_edges(y_true)   # [B,H,W,C,2]
    pred_edges = tf.image.sobel_edges(y_pred)

    true_mag = tf.sqrt(tf.reduce_sum(tf.square(true_edges), axis=-1) + 1e-6)
    pred_mag = tf.sqrt(tf.reduce_sum(tf.square(pred_edges), axis=-1) + 1e-6)

    return tf.reduce_mean(tf.abs(true_mag - pred_mag))

def et_boundary_loss(y_true, y_pred):
    y_true_et = y_true[..., 0:1]
    y_pred_et = y_pred[..., 0:1]
    return boundary_loss(y_true_et, y_pred_et)

def hybrid_boundary_et_loss(y_true, y_pred):
    bce = tf.reduce_mean(tf.keras.losses.binary_crossentropy(y_true, y_pred))
    dice_all = dice_loss(y_true, y_pred)
    dice_et = et_dice_loss(y_true, y_pred)
    bnd_all = boundary_loss(y_true, y_pred)
    bnd_et = et_boundary_loss(y_true, y_pred)

    return (
        0.35 * bce +
        0.20 * dice_all +
        0.20 * dice_et +
        0.15 * bnd_all +
        0.10 * bnd_et
    )

model = build_model()

model.compile(
    optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=0.0005),
    loss=hybrid_boundary_et_loss,
    metrics=["accuracy"]
)

os.makedirs("results/models", exist_ok=True)

callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        "results/models/best_model.keras",
        save_best_only=True,
        monitor="val_loss",
        mode="min"
    ),
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=3,
        restore_best_weights=True
    )
]

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    steps_per_epoch=steps_per_epoch,
    validation_steps=validation_steps,
    callbacks=callbacks,
    verbose=1
)

model.save("results/models/bc_tsea_unet.keras")
print("✅ Model saved")

np.save("results/models/train_loss.npy", np.array(history.history["loss"]))
np.save("results/models/val_loss.npy", np.array(history.history["val_loss"]))
np.save("results/models/train_acc.npy", np.array(history.history["accuracy"]))
np.save("results/models/val_acc.npy", np.array(history.history["val_accuracy"]))
print("✅ Training history saved")
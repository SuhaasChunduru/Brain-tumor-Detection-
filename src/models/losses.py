"""Loss terms for the three-region segmentation head.

Channel 0 is ET, the smallest and hardest region, so it gets its own overlap and
boundary terms on top of the all-region ones.
"""

import tensorflow as tf


def dice_loss(y_true, y_pred, smooth=1e-6):
    """1 - Dice over all region channels at once."""
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    intersection = tf.reduce_sum(y_true * y_pred, axis=[1, 2, 3])
    denom = tf.reduce_sum(y_true, axis=[1, 2, 3]) + tf.reduce_sum(y_pred, axis=[1, 2, 3])

    dice = (2.0 * intersection + smooth) / (denom + smooth)
    return 1.0 - tf.reduce_mean(dice)


def et_dice_loss(y_true, y_pred, smooth=1e-6):
    """1 - Dice on the ET channel only."""
    y_true_et = tf.cast(y_true[..., 0:1], tf.float32)
    y_pred_et = tf.cast(y_pred[..., 0:1], tf.float32)

    intersection = tf.reduce_sum(y_true_et * y_pred_et, axis=[1, 2, 3])
    denom = tf.reduce_sum(y_true_et, axis=[1, 2, 3]) + tf.reduce_sum(y_pred_et, axis=[1, 2, 3])

    dice = (2.0 * intersection + smooth) / (denom + smooth)
    return 1.0 - tf.reduce_mean(dice)


def boundary_loss(y_true, y_pred):
    """L1 distance between Sobel edge magnitudes of the target and prediction.

    Dice is dominated by region interiors, so a blurry or slightly displaced
    contour costs almost nothing. Matching edge maps penalizes that directly.
    """
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    true_edges = tf.image.sobel_edges(y_true)  # [B, H, W, C, 2]
    pred_edges = tf.image.sobel_edges(y_pred)

    true_mag = tf.sqrt(tf.reduce_sum(tf.square(true_edges), axis=-1) + 1e-6)
    pred_mag = tf.sqrt(tf.reduce_sum(tf.square(pred_edges), axis=-1) + 1e-6)

    return tf.reduce_mean(tf.abs(true_mag - pred_mag))


def et_boundary_loss(y_true, y_pred):
    """Sobel boundary loss on the ET channel only."""
    return boundary_loss(y_true[..., 0:1], y_pred[..., 0:1])


def hybrid_boundary_et_loss(y_true, y_pred):
    """Weighted sum of pixel, overlap, and boundary terms.

        0.35 BCE + 0.20 Dice(all) + 0.20 Dice(ET) + 0.15 edge(all) + 0.10 edge(ET)

    BCE keeps early training stable when the Dice terms are near-degenerate on
    mostly-empty masks; the Dice terms carry the region overlap; the edge terms
    target HD95/ASSD. Weights were set by hand, not searched.
    """
    bce = tf.reduce_mean(tf.keras.losses.binary_crossentropy(y_true, y_pred))
    dice_all = dice_loss(y_true, y_pred)
    dice_et = et_dice_loss(y_true, y_pred)
    bnd_all = boundary_loss(y_true, y_pred)
    bnd_et = et_boundary_loss(y_true, y_pred)

    return (
        0.35 * bce
        + 0.20 * dice_all
        + 0.20 * dice_et
        + 0.15 * bnd_all
        + 0.10 * bnd_et
    )

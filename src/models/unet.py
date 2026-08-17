"""BC-TSEA-UNet: a U-Net whose every stage is gated by twin squeeze-excite blocks."""

from tensorflow.keras import layers, models

from src.config import IMG_SIZE, N_MODALITIES, N_REGIONS


def se_block(x, reduction=16):
    """Squeeze-and-excitation: global pool -> bottleneck MLP -> per-channel gate."""
    ch = x.shape[-1]
    s = layers.GlobalAveragePooling2D()(x)
    s = layers.Dense(max(ch // reduction, 1), activation="relu")(s)
    s = layers.Dense(ch, activation="sigmoid")(s)
    s = layers.Reshape((1, 1, ch))(s)
    return layers.Multiply()([x, s])


def twin_se_block(x):
    """Two BN -> ReLU -> SE stages in series.

    A single SE gate multiplies the features once; stacking two with a
    normalization between them lets the second gate re-weight channels after the
    first has already suppressed some, which is what "twin" refers to.
    """
    y = layers.BatchNormalization()(x)
    y = layers.Activation("relu")(y)
    y = se_block(y)

    y = layers.BatchNormalization()(y)
    y = layers.Activation("relu")(y)
    y = se_block(y)
    return y


def conv_block(x, filters):
    """Double 3x3 conv + BN + ReLU."""
    x = layers.Conv2D(filters, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.Conv2D(filters, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    return x


def build_model(input_shape=(IMG_SIZE, IMG_SIZE, N_MODALITIES), n_regions=N_REGIONS):
    """4-level U-Net, 32 -> 512 filters, twin-SE at every encoder and decoder stage.

    The skip connections carry the *gated* encoder features (s1..s4) rather than
    the raw conv outputs, so the decoder sees the same channel weighting the
    encoder settled on.

    The output is `n_regions` independent sigmoids, not a softmax: ET, TC, and WT
    are nested regions (ET < TC < WT), so a pixel legitimately belongs to more
    than one of them.
    """
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

    outputs = layers.Conv2D(n_regions, 1, activation="sigmoid")(d1)
    return models.Model(inputs, outputs, name="bc_tsea_unet")

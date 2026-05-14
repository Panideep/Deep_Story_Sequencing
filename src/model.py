import tensorflow as tf
from tensorflow.keras import layers, Model


class TemporalAttention(layers.Layer):

    def __init__(self, units):
        super().__init__()
        self.W = layers.Dense(units, activation="tanh")
        self.V = layers.Dense(1)

    def call(self, features):

        score = self.V(self.W(features))

        weights = tf.nn.softmax(score, axis=1)

        context = weights * features

        context = tf.reduce_sum(context, axis=1)

        return context


def build_model(
    filters=None,
    kernel_size=3,
    dropout=0.3,
    attention_units=128,
    batch_norm=False,
    num_classes=5,
    input_shape=(224, 224, 3),
):
    """Build the CNN-attention network.

    When called with a single config dict (legacy), the values are looked up
    there for backwards compatibility. Otherwise each argument can be passed
    directly, which is more convenient in the notebook experiments.
    """

    # support legacy usage where a config dict is passed
    if filters is None or isinstance(filters, dict):
        config = filters
        filters = config["model"]["conv_filters"]
        kernel_size = config["model"]["kernel_size"]
        dropout = config["model"]["dropout"]
        batch_norm = config["model"]["batch_norm"]
        attention_units = config["model"]["attention_units"]
        num_classes = config["model"]["num_classes"]
        input_shape = tuple(config["model"].get("input_shape", (224, 224, 3)))

    if filters is None:
        filters = [32, 64, 128]

    inputs = layers.Input(shape=input_shape)
    x = inputs

    for f in filters:
        x = layers.Conv2D(f, kernel_size, padding="same")(x)
        if batch_norm:
            x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.MaxPooling2D()(x)

    x = layers.Reshape((-1, x.shape[-1]))(x)
    attention = TemporalAttention(attention_units)(x)
    x = layers.Dense(256, activation="relu")(attention)
    x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = Model(inputs, outputs)
    return model
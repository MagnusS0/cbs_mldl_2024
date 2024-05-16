import keras
from functools import partial

from keras.saving import register_keras_serializable

DefaultConv2D = partial(keras.layers.Conv2D, kernel_size=3, strides=1,
                        padding="same", kernel_initializer="he_normal",
                        use_bias=False)


@register_keras_serializable(package='Custom', name='ResidualUnit')
class ResidualUnit(keras.layers.Layer):
    """A custom residual unit layer for ResNet models.

    This layer implements a residual unit, which is a building block for ResNet models.
    It consists of a series of convolutional layers, batch normalization, and skip connections.

    Args:
        filters (int): The number of filters in the convolutional layers.
        strides (int, optional): The stride value for the convolutional layers. Defaults to 1.
        activation (str or callable, optional): The activation function to use. Defaults to "relu".

    """

    def __init__(self, filters, strides=1, activation="relu", **kwargs):
        super().__init__(**kwargs)
        self.strides = strides
        self.filters = filters
        self.activation = keras.activations.get(activation)
        self.main_layers = [
            DefaultConv2D(filters, strides=strides),
            keras.layers.BatchNormalization(),
            self.activation,
            DefaultConv2D(filters),
            keras.layers.BatchNormalization()
        ]
        self.skip_layers = []
        if strides > 1:
            self.skip_layers = [
                DefaultConv2D(filters, kernel_size=1, strides=strides),
                keras.layers.BatchNormalization()
            ]

    def build(self, input_shape):
        super().build(input_shape)

    def call(self, inputs):
        Z = inputs
        for layer in self.main_layers:
            Z = layer(Z)
        skip_Z = inputs
        for layer in self.skip_layers:
            skip_Z = layer(skip_Z)
        return self.activation(Z + skip_Z)

    def get_config(self):
        config = super().get_config()
        config.update({
                "filters": self.filters,
                "strides": self.strides,
                "activation": keras.activations.serialize(self.activation)
            })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

def build_resnet_model(classes:int):
    model = keras.Sequential([
        keras.layers.Input(shape=(224, 224, 3)),
        DefaultConv2D(64, kernel_size=7, strides=2),
        keras.layers.BatchNormalization(),
        keras.layers.Activation("relu"),
        keras.layers.MaxPool2D(pool_size=3, strides=2, padding="same"),
    ])
    prev_filters = 64
    for filters in [64] * 2 + [128] * 2 + [256] * 2 + [512] * 2:
        strides = 1 if filters == prev_filters else 2
        model.add(ResidualUnit(filters, strides=strides))
        prev_filters = filters

    model.add(keras.layers.GlobalAvgPool2D())
    model.add(keras.layers.Dense(classes, activation="softmax"))
    return model

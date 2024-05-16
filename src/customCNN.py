import keras
from functools import partial


DefaultConv2D = partial(keras.layers.Conv2D, kernel_size=7, strides=1,
                        padding="same", kernel_initializer="he_normal",
                        use_bias=False)

def build_cnn_model(classes:int):
    model = keras.Sequential([
        keras.layers.Input(shape=(224, 224, 3)),
        # First Convolutional Block
        DefaultConv2D(64),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPooling2D((2, 2)),

        # Second Convolutional Block
        DefaultConv2D(128),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPooling2D((2, 2)),

        # Third Convolutional Block
        DefaultConv2D(256),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPooling2D((2, 2)),

        # Global Average Pooling
        keras.layers.GlobalAveragePooling2D(),

        # Dense Layers
        keras.layers.Dense(512, activation='relu', kernel_initializer="he_normal", use_bias=False),
        keras.layers.Dropout(0.3),
        keras.layers.BatchNormalization(),

        keras.layers.Dense(256, activation='relu', kernel_initializer="he_normal", use_bias=False),
        keras.layers.Dropout(0.3), 
        keras.layers.BatchNormalization(),

        # Output Layer
        keras.layers.Dense(classes, activation='softmax')
    ])
    
    return model
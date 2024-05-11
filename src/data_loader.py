import tensorflow as tf
import keras
import keras_cv
import os
import pandas as pd
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
import keras
import keras_cv
import os
import pandas as pd
from sklearn.utils.class_weight import compute_class_weight


class ImageAugmenter:
    def __init__(self):
        self.random_flip = keras_cv.layers.RandomFlip(mode="horizontal_and_vertical")
        self.crop_and_resize = keras_cv.layers.RandomCropAndResize(
            target_size=(224, 224),
            crop_area_factor=(0.8, 1.0),
            aspect_ratio_factor=(0.9, 1.1)
        )
        self.rand_augment = keras_cv.layers.RandAugment(
            augmentations_per_image=3,
            value_range=(0, 1),
            magnitude=0.5,
            magnitude_stddev=0.2,
            rate=1.0
        )

    def augment_train(self, images, labels):
        images = tf.cast(images, tf.float32) / 255.0
        images = self.random_flip(images, training=True)
        images = self.crop_and_resize(images, training=True)
        images = self.rand_augment(images, training=True)
        return images, labels

    def augment_val(self, images, labels):
        images = tf.cast(images, tf.float32) / 255.0
        return images, labels


def load_datasets(train_path, val_path, batch_size):
    augmenter = ImageAugmenter()

    train_ds = keras.utils.image_dataset_from_directory(
        train_path,
        labels="inferred",
        label_mode="categorical",
        class_names=None,
        color_mode="rgb",
        batch_size=batch_size,
        image_size=(224, 224),
        shuffle=True,
        seed=0
    )
    train_ds = train_ds.map(
        augmenter.augment_train, 
        num_parallel_calls=tf.data.AUTOTUNE).prefetch(buffer_size=tf.data.AUTOTUNE)

    val_ds = keras.utils.image_dataset_from_directory(
        val_path,
        labels="inferred",
        label_mode="categorical",
        class_names=None,
        color_mode="rgb",
        batch_size=batch_size,
        image_size=(224, 224),
        shuffle=False
    )
    val_ds = val_ds.map(
        augmenter.augment_val, 
        num_parallel_calls=tf.data.AUTOTUNE).prefetch(buffer_size=tf.data.AUTOTUNE)

    return train_ds, val_ds


def compute_class_weights(dataset_dir):
    # Extract class weights for imbalanced datasets
    data = []
    for class_dir in os.listdir(dataset_dir):
        for img in os.listdir(os.path.join(dataset_dir, class_dir)):
            data.append((os.path.join(dataset_dir, class_dir, img), class_dir))
    df = pd.DataFrame(data, columns=['filepath', 'label'])
    class_labels = df['label'].unique()
    weights = compute_class_weight(class_weight='balanced', classes=class_labels, y=df['label'].values)
    return weights

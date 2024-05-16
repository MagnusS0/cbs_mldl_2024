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
    """
    A class that performs image augmentation operations on input images.

    Attributes:
        random_flip: A `keras_cv.layers.RandomFlip` layer for random flipping of images.
        crop_and_resize: A `keras_cv.layers.RandomCropAndResize` layer for random cropping and resizing of images.

    Methods:
        augment_train(images, labels): Applies image augmentation operations on training images.
        augment_val(images, labels): Applies image normalization on validation images.
    """

    def __init__(self):
        self.random_flip = keras_cv.layers.RandomFlip(mode="horizontal_and_vertical")
        self.crop_and_resize = keras_cv.layers.RandomCropAndResize(
            target_size=(224, 224),
            crop_area_factor=(0.8, 1.0),
            aspect_ratio_factor=(0.9, 1.1)
        )

    def augment_train(self, images, labels):
        images = tf.cast(images, tf.float32) / 255.0
        images = self.random_flip(images, training=True)
        images = self.crop_and_resize(images, training=True)
        return images, labels

    def augment_val(self, images, labels):
        images = tf.cast(images, tf.float32) / 255.0
        return images, labels


def load_datasets(train_path, val_path, batch_size):
    """
    Loads and preprocesses the training and validation datasets.

    Args:
        train_path (str): The path to the training dataset directory.
        val_path (str): The path to the validation dataset directory.
        batch_size (int): The batch size for training and validation.

    Returns:
        train_ds (tf.data.Dataset): The preprocessed training dataset.
        val_ds (tf.data.Dataset): The preprocessed validation dataset.
    """
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
    """ 
    Extract class weights for imbalanced datasets
    """
    data = []
    for class_dir in os.listdir(dataset_dir):
        for img in os.listdir(os.path.join(dataset_dir, class_dir)):
            data.append((os.path.join(dataset_dir, class_dir, img), class_dir))
    df = pd.DataFrame(data, columns=['filepath', 'label'])
    class_labels = df['label'].unique()
    weights = compute_class_weight(class_weight='balanced', classes=class_labels, y=df['label'].values)
    return weights

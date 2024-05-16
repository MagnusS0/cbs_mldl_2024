import tensorflow as tf
from keras.optimizers import SGD
from keras.losses import CategoricalFocalCrossentropy

from data_loader import compute_class_weights
from callbacks import get_callbacks

def compile_and_train(model, train_ds, val_ds, epochs, max_lr=1e-2):
    """
    Compiles the model with specified optimizer, loss function, and metrics.
    Trains the model on the training dataset for the specified number of epochs.
    
    Args:
        model (tf.keras.Model): The model to be compiled and trained.
        train_ds (tf.data.Dataset): The training dataset.
        val_ds (tf.data.Dataset): The validation dataset.
        epochs (int): The number of epochs to train the model.
        max_lr (float, optional): The maximum learning rate for the optimizer. Defaults to 1e-2.
    
    Returns:
        history (tf.keras.callbacks.History): A History object containing the training history.
    """
    class_weights = compute_class_weights("../data/train_directory")
    
    model.compile(
        optimizer=SGD(learning_rate=0.001, momentum=0.9, weight_decay=1e-4),
        loss=CategoricalFocalCrossentropy(class_weights),
        metrics=['accuracy']
    )
    
    callbacks = get_callbacks(train_ds, epochs, max_lr=max_lr)

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks,
    )
    return history

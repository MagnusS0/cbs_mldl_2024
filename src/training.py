import tensorflow as tf
from keras.optimizers import SGD
from keras.losses import CategoricalFocalCrossentropy

from data_loader import compute_class_weights
from callbacks import get_callbacks

def compile_and_train(model, train_ds, val_ds, epochs, max_lr=1e-2):
    class_weights = compute_class_weights("./Dataset/train_directory")
    
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

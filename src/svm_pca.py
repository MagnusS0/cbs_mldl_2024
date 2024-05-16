import numpy as np
from joblib import Parallel, delayed
from skimage.io import imread
from skimage.color import gray2rgb, rgba2rgb
from skimage.transform import resize
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
import os

def process_image(file_path, label, target_size=(64, 64)):
    image = imread(file_path)
    if image.shape[-1] == 4:
        image = rgba2rgb(image)
    elif image.ndim == 2:
        image = gray2rgb(image)
    image = resize(image, target_size, anti_aliasing=True)
    image_flat = image.flatten()
    return image_flat / 255.0, label

def preprocess_images_for_svm(directory, target_size=(64, 64)):
    class_names = sorted(os.listdir(directory))
    # Load images in parallel to speed up the process
    results = Parallel(n_jobs=-1)(delayed(process_image)(os.path.join(directory, class_name, file_name), label, target_size)
                                  for label, class_name in enumerate(class_names)
                                  for file_name in os.listdir(os.path.join(directory, class_name))
                                  if os.path.isdir(os.path.join(directory, class_name)))
    images, labels = zip(*results)
    return np.array(images), np.array(labels)

def load_svm_datasets(train_path, val_path, target_size=(64, 64)):
    X_train, y_train = preprocess_images_for_svm(train_path, target_size)
    X_val, y_val = preprocess_images_for_svm(val_path, target_size)
    
    return X_train, y_train, X_val, y_val

def create_svm_pca_pipeline(n_components=20, C=10.0, gamma='auto', random_state=42):
    pca = PCA(n_components=n_components, whiten=True, random_state=random_state)
    svm = SVC(kernel='rbf', C=C, gamma=gamma, random_state=random_state, class_weight='balanced')
    model = make_pipeline(pca, svm)
    return model
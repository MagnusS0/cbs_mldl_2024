from sklearn.metrics import accuracy_score
from svm_pca import load_svm_datasets
from svm_pca import create_svm_pca_pipeline
import joblib

TRAIN_DATADIR = "../data/train_directory"
VAL_DATADIR = "../data/val_directory"

def train_svm(train_path, val_path, target_size=(64, 64)):
    # Load your data
    print("Loading datasets...")
    X_train, y_train, X_val, y_val = load_svm_datasets(train_path, val_path, target_size)  # Assuming load_data function provides the datasets
    print(f"Datasets loaded \n X_train shape: {X_train.shape}, y_train shape: {y_train.shape}, \n X_val shape: {X_val.shape}, y_val shape: {y_val.shape}")

    # Create SVM with PCA pipeline
    model_svm = create_svm_pca_pipeline()

    # Train the SVM model
    print("Training the SVM model...")
    model_svm.fit(X_train, y_train)

    print("Training completed.")

    # Predict and evaluate the SVM model
    y_pred_svm = model_svm.predict(X_val)
    accuracy_svm = accuracy_score(y_val, y_pred_svm)
    print(f"SVM Model Accuracy: {accuracy_svm * 100:.2f}%")

    # Save the model
    joblib.dump(model_svm, "../models/svm_model.pkl")
    print("SVM model saved successfully.")

if __name__ == '__main__':
    train_svm(TRAIN_DATADIR, VAL_DATADIR)
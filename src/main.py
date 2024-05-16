from data_loader import load_datasets
from resnet18 import build_resnet_model
from customCNN import build_cnn_model
from training import compile_and_train
from utils import set_random_seeds


# Constants for training
TRAIN_DATADIR = "../data/train_directory"
VAL_DATADIR = "../data/val_directory"
BATCH_SIZE = 128
EPOCHS = 35
MAX_LR = 1e-2
CLASSES = 11
MODEL_NAME = "resnet18"

def model(model_name):
    if model_name == "resnet18":
        return build_resnet_model
    elif model_name == "cnn":
        return build_cnn_model
    else:
        raise ValueError("Invalid model name")
    
build_model = model(MODEL_NAME)

def main(build_model=build_model):
    set_random_seeds(42)
    train_ds, val_ds = load_datasets(TRAIN_DATADIR, VAL_DATADIR, BATCH_SIZE) 
    model = build_model(CLASSES) 
    compile_and_train(model, train_ds, val_ds, EPOCHS, MAX_LR)

if __name__ == '__main__':
    main()

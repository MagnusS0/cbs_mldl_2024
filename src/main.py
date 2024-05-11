from data_loader import load_datasets
from resnet18 import build_model
from training import compile_and_train
from utils import set_random_seeds
import tensorflow as tf

# Constants
TRAIN_DATADIR = "./Dataset/train_directory"
VAL_DATADIR = "./Dataset/val_directory"
BATCH_SIZE = 128
EPOCHS = 75

def main():
    set_random_seeds(42)
    train_ds, val_ds = load_datasets(TRAIN_DATADIR, VAL_DATADIR, BATCH_SIZE) 
    model = build_model(classes=11) 
    compile_and_train(model, train_ds, val_ds, EPOCHS)

if __name__ == '__main__':
    main()

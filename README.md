# cbs_mldl_2024


## Training Models with main.py
The [main.py](src/main.py) file in this project is used to train our models. Here's how you can use it:

Prerequisites
Ensure you have the following installed:

- Python 3.10 or later
- TensorFlow 2.16 or later
- Keras 3.2 or later
- Keras-cv 0.8.2

### Steps
1. **Set up your environment**: Make sure you have installed all the necessary packages listed in the prerequisites.

2. **Prepare your data**: Unzip the dataset.zip in the /data directory. The data should be in a format that can be read by the load_datasets function in src/data_loader.py.

3. **Run the script**: Navigate to the src directory and run the [main.py](src/main.py) script. You can do this by opening a terminal in the src directory and running the following command:

```sh
python main.py
```

This will start the training process using the default parameters defined in the [main.py](src/main.py) file:

- `TRAIN_DATADIR`: The directory where the training data is located. Default is `./data/train_directory`.
- `VAL_DATADIR`: The directory where the validation data is located. Default is `./data/val_directory`.
- `BATCH_SIZE`: The batch size for training. Default is 128.
- `EPOCHS`: The number of epochs for training. Default is 35.
- `MAX_LR`: The maximum learning rate. Default is 1e-2.
- `CLASSES`: The number of classes. Default is 11.
- `MODEL_NAME`: The name of the model. Default is 'resnet18'.
  
The main function in `main.py` uses these constants to load the datasets, build the model, and start the training process.

4. **Customize the training process**: If you want to use a different model or change the training parameters, you can modify the constants in the [main.py](src/main.py) file or pass a different model building function to the main function.

### Output
The trained model will be saved in the models directory. The training process also generates logs, which are saved in the logs directory.

## Training SVM with svm_training.py
### Step
1. **Run the script:** Navigate to the directory containing the svm_training.py script in your terminal and run the following command:
```sh
python svm_training.py
```
This will start the training process. The script will first load the datasets from the specified directories. It will then create an SVM model with a PCA pipeline and train it using the loaded data.

2. **Check the output:** After the training is completed, the script will evaluate the model's accuracy on the validation data and print the accuracy score. The trained model will be saved as svm_model.pkl in the ../models directory.

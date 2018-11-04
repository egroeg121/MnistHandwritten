
'''

GENERAL Description:
Use previous calculated weights to create predictions for a dataset. Produces probabilities for each class and saves in hdf5
prints the loss and accuracy of the evaluation

COMMAND LINE ARGUMENTS:
    [0]RUN_DIR [1]MODEL_PATH [2]WEIGHTS_HDF5_DIR [3]DATA_HDF5_DIR [4]OUTPUT_PREDICTS_DIR

        DIR stands for directory
        the HDF5 must have an images and labels dataset
        Model_path is the .py file containg the getModel function
    Or as module calling the different functions with correct parameters

    To change train parameters like number of epochs etc look at the train function
'''


def main(run_dir,model_path,weights_dir,data_dir):

    # Load data from hdf5
    data_images, data_labels = load_data(data_dir)

    # Display Data sizes
    print( "\nShape of data:")
    print("Images: ",data_images.shape,"  Labels: ",data_labels.shape,"\n")

    # Evaluate data
    loss,accuracy = evaluate(model_path,weights_dir,data_images,data_labels)

    print( "\nEvaluation Results:")
    print("Loss: ",loss,"  Accuracy: ",accuracy,"\n")

def evaluate(model_path,weights_dir,data_images,data_labels):
    from keras.optimizers import SGD
    from keras.utils import to_categorical

    # Import model
    model = load_model(model_path)

    # Load weights
    model.load_weights(weights_dir)

    # compile
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    # Create logger object
    #logger = callbacks.TensorBoard(log_dir='logs', write_graph=True,histogram_freq=5)

    # train
    loss,accuracy = model.evaluate(
        data_images,
        data_labels,
        batch_size=500,
        verbose=1
    )

    return loss,accuracy

def save_predictions(predictions,predicts_dir):
    import h5py

    f = h5py.File(predicts_dir, "w")
    predicts_dset = f.create_dataset("predicts", data=predictions)
    f.close()


def load_model(model_path): # Load the model.py file from model_path. then execute getModel()
    import importlib.util

    spec = importlib.util.spec_from_file_location("module.name", model_path)
    model = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(model)
    return model.getModel()


def load_data(input_dir):
    import h5py

    f = h5py.File(input_dir, "r")
    if len( f.keys() ) != 2:
        print(" ERROR: Incorrect number of datasets. There should be 2 datasets: 'images and 'labels' ")
        exit()

    images = f['images']
    labels = f['labels']

    return images,labels

# If running from terminal
import sys
if len(sys.argv) != 4:
    print("ERROR: Incorrect number of Arguments. Input Arguements: ",len(sys.argv))
    print("Sys Arg: [0]RUN_DIR [1]MODEL_PATH [2]WEIGHTS_HDF5_DIR [3]DATA_HDF5_DIR")
    exit()
if __name__ == "__main__": main(sys.argv[0],sys.argv[1],sys.argv[2],sys.argv[3],)
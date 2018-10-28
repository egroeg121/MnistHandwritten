
'''

PREDICT: Use previous calculated weights to create predictions for a dataset. Produces probabilities for each class and saves in hdf5
    Can be used with command line arguements: [0]RUN_DIR [1]MODEL_PATH [2]WEIGHTS_HDF5_DIR [3]DATA_HDF5_DIR [4]OUTPUT_PREDICTS_DIR
        DIR stands for directory
        the HDF5 must have an images and labels dataset
    Or as module calling the different functions with correct parameters

'''


def main(run_dir,model_path,weights_dir,data_dir,predicts_dir):

    # Load data from hdf5
    data_images, data_labels = load_data(data_dir)

    # Display Data sizes
    print( "\nShape of data:")
    print("Images: ",data_images.shape,"   Labels: ",data_labels.shape,"\n")

    # Train data
    predictions = predict(model_path,weights_dir,data_images)

    # Save Predictions
    save_predictions(predictions,predicts_dir)

def predict(model_path,weights_dir,data_images):
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
    predicts = model.predict(
        data_images,
        batch_size=500,
        verbose=1
    )

    return predicts

def save_predictions(predictions,predicts_dir):
    import h5py

    f = h5py.File(predicts_dir, "w")
    predicts_dset = f.create_dataset("predicts", data=predictions)
    f.close()


def load_model(model_path):
    import importlib.util

    spec = importlib.util.spec_from_file_location("module.name", model_path)
    foo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(foo)
    return foo.getModel()


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
if len(sys.argv) != 5:
    print("ERROR: Incorrect number of Arguments. Input Arguements: ",len(sys.argv))
    print("Sys Arg: [0]RUN_DIR [1]MODEL_PATH [2]WEIGHTS_HDF5_DIR [3]DATA_HDF5_DIR [4]OUTPUT_PREDICTS_DIR")
    exit()
if __name__ == "__main__": main(sys.argv[0],sys.argv[1],sys.argv[2],sys.argv[3],sys.argv[4])
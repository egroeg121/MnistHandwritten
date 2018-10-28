def main(run_dir,model_path,weights_dir,data_dir):

    # Load data from hdf5
    data_images, data_labels = load_data(data_dir)

    # Display Data sizes
    print( "\nShape of data:")
    print("Images: ",data_images.shape,"   Labels: ",data_labels.shape,"\n")

    # Train data
    predicts = predict(model_path,weights_dir,data_images)

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
if len(sys.argv) != 4:
    print("ERROR: Incorrect number of Arguments. Input Arguements: ",len(sys.argv))
    print("Sys Arg: [0]FILE.PY [1]MODEL_PATH [2]WEIGHTS_PATH [3]DATA_PATH")
    exit()
if __name__ == "__main__": main(sys.argv[0],sys.argv[1],sys.argv[2],sys.argv[3])
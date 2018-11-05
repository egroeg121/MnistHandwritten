
'''

GENERAL Description:
 trains a chosen network using the input data. It tests against the test dataset on every epoch.
 Then saves weights to a hdf5 at the output directory

COMMAND LINE ARGUMENTS:]
    RUN_DIR MODEL_PATH TRAIN_DIR_HDF5 TEST_DIR_HDF5 OUTPUT_WEIGHTS_DIR

        DIR stands for directory
        the HDF5 must have an images and labels dataset
    Or as module calling the different functions with correct parameters

    To change train parameters like number of epochs etc look at the train function

'''

def main(run_dir,model_path,train_dir,test_dir,weights_dir):

    from FuncLib import hdf5_functions

    # Load data from hdf5
    #train_images,train_labels = load_data(train_dir)
    train_data = hdf5_functions.read(train_dir,("images","labels"))
    train_images = train_data[0]
    train_labels = train_data[1]
    test_data = hdf5_functions.read(test_dir, ("images", "labels"))
    test_images = test_data[0]
    test_labels = test_data[1]

    # Display Data sizes
    print( "Shape of train data:")
    print("Images: ",train_images.shape,"   Labels: ",train_labels.shape)
    print( "Shape of test data:")
    print("Images: ",test_images.shape,"   Labels: ",test_labels.shape)

    # Train data
    model, train_acc, train_losses, val_acc, val_losses = train(model_path,train_images,train_labels,test_images,test_labels)

    # Plot train acc and val acc



    # Save Weights
    save_weights(model,weights_dir)

def train(model_path,train_images,train_labels,test_images,test_labels):
    from keras import callbacks
    from keras.optimizers import SGD
    from keras.utils import to_categorical

    # Import model
    print("Importing Model")
    model = load_model(model_path)

    # compile
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    # Create logger object
    #logger = callbacks.TensorBoard(log_dir='logs', write_graph=True,histogram_freq=5)

    print("Started Training")
    # train
    model_history = model.fit(
        train_images,
        to_categorical(train_labels),
        epochs=2,
        batch_size=100,
        verbose=1,
        shuffle="batch",
        # callbacks=[logger],
        validation_data=[test_images, to_categorical(test_labels)]
    )

    train_accuracy_list = model_history
    val_losses = model_history.history.val_loss
    val_acc = model_history.history.val_loss
    train_losses = model_history.loss
    train_acc = model_history.acc

    return model,train_acc,train_losses,val_acc,val_losses

def save_weights(model,weights_dir):
    # Save weights
    model.save_weights(weights_dir)


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

def plot(y1,y2):
        import matplotlib.pyplot as plt

        x = range(len(y1))
        plt.plot(x,y1)
        plt.plot(x,y2)
        plt.show()

# If running from terminal
import sys
if len(sys.argv) != 5:
    print("ERROR: Incorrect number of Arguments. Input Arguements: ",len(sys.argv))
    print("Sys Arg: [0]FILE.PY [1]MODEL_PATH [2]TRAIN_DIR_HDF5 [3]TEST_DIR_HDF5 [4]OUTPUT_WEIGHTS_DIR")
    exit()
if __name__ == "__main__": main(sys.argv[0],sys.argv[1],sys.argv[2],sys.argv[3],sys.argv[4])
def main(run_dir,train_dir,test_dir,weights_dir):

    # Load data from hdf5
    train_images,train_labels = load_data(train_dir)
    test_images,test_labels = load_data(test_dir)

    # Display Data sizes
    print( "Shape of train data:")
    print("Images: ",train_images.shape,"   Labels: ",train_labels.shape)
    print( "Shape of test data:")
    print("Images: ",test_images.shape,"   Labels: ",test_labels.shape)

    # Train data
    model = train(train_images,train_labels,test_images,test_labels)

    # Save weights
    save_weights(model,weights_dir)

def train(train_images,train_labels,test_images,test_labels):
    from keras import callbacks
    from keras.optimizers import SGD
    from keras.utils import to_categorical
    import handwritemodel as hwmodel

    # Import model
    model = hwmodel.getModel()

    # compile
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    # Create logger object
    #logger = callbacks.TensorBoard(log_dir='logs', write_graph=True,histogram_freq=5)

    # train
    model.predict(
        train_images,
        to_categorical(train_labels),
        epochs=10,
        batch_size=500,
        verbose=1,
        shuffle="batch",
        # callbacks=[logger],
        validation_data=[test_images, to_categorical(test_labels)]
    )

    return model


def load_data(input_dir):
    import h5py

    f = h5py.File(input_dir, "r")
    if len( f.keys() ) != 2:
        print(" ERROR: Incorrect number of datasets. There should be 2 datasets: 'images and 'labels' ")
        exit()

    images = f['images']
    labels = f['labels']

    return images,labels

def save_weights(model,weights_dir):
    import h5py

    weights = model.get_weights()
    f = h5py.File(weights_dir, "w")
    weights_dset = f.create_dataset("weights", data=weights)
    f.close()


# If running from terminal
import sys
if len(sys.argv) != 4:
    print("ERROR: Incorrect number of Arguments. Input Arguements: ",len(sys.argv))
    print("Sys Arg: FILE.PY TRAIN_DIR_HDF5 TEST_DIR_HDF5 OUTPUT_WEIGHTS_DIR")
    exit()
if __name__ == "__main__": main(sys.argv[0],sys.argv[1],sys.argv[2],sys.argv[3])
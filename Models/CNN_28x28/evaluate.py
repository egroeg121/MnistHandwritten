

def main(run_dir,dir):

    # Load data from hdf5
    images,labels = load_data(dir)

    # Display Data sizes
    print( "Shape of train data:")
    print("Images: ",images.shape,"   Labels: ",labels.shape)

    # Train data
    evaluate(images,labels)



def evaluate(images,labels):
    from keras.utils import to_categorical
    import handwritemodel as model

    # Import model
    model = model.getModel()

    # train
    model.evaluate(images,
                   to_categorical(labels),
                   batch_size=100)


def load_data(input_dir):
    import h5py

    f = h5py.File(input_dir, "r")
    if len( f.keys() ) != 2:
        print(" ERROR: Incorrect number of datasets. There should be 2 datasets: 'images and 'labels' ")
        exit()

    images = f['images']
    labels = f['labels']

    return images,labels

# If running from terminal rather than as library
import sys
if len(sys.argv) != 2:
    print("ERROR: Incorrect number of Arguments. Input Arguements: ",len(sys.argv))
    print("Sys Arg: FILE.PY INPUT_HDF5")
    exit()
if __name__ == "__main__": main(sys.argv[0],sys.argv[1])
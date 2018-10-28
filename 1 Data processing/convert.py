import numpy as np

'''
GENERAL DESCRIPTION:
	Each file converts 2 ubytes files into one hdf5 file containg two datasets: 'images' and 'labels'

COMMAND LINE ARGUMENTS:
	FILE_NAME INPUT_IMAGES INPUT_LABELS OUTPUT_DIRECTORY


FILES:
	No_Processing : does not processing on the dataset, just converts the ubytes into hdf5

'''


def main(run_dir,input_dir_images,input_dir_labels,output_dir):

    # get np array from ubytes
    print("Loading Data")
    input_np_images = read_ubyte(input_dir_images)
    input_np_labels = read_ubyte(input_dir_labels)
    print("Images data shape: ",input_np_images.shape)
    print("Labels data shape: ", input_np_labels    .shape)

    # Process data
    output_np_images = np.expand_dims(input_np_images, axis=3)
    output_np_labels = input_np_labels#

    # write nparray to hdf5
    write_hdf5(output_dir,output_np_images,output_np_labels)

    print("DONE")

def write_hdf5(directory,images_np,labels_np):
    import h5py

    f = h5py.File(directory, "w")
    image_dset = f.create_dataset("images", data=images_np)
    label_dset = f.create_dataset("labels", data=labels_np)
    f.close()

def read_ubyte(filename):
    import numpy as np

    # numpy datatypes in relation to the 'magic number'
    dtype_lib = {
        b'\x08': 'uint8',
        b'\x09': 'int8',
        b'\x0B': 'int16',
        b'\x0C': 'int32',
        b'\x0D': 'float32',
        b'\x0E': 'float64'}

    data = open(filename, 'rb')

    # extract the datatype and no. of dimensions of the data from the 'magic number'
    data.seek(2)
    datatype = dtype_lib.get(data.read(1))
    no_dimensions = int.from_bytes(data.read(1), byteorder='big')

    array_dimensions = []
    for n in range(no_dimensions):
        array_dimensions.append(int.from_bytes(data.read(4), byteorder='big'))

    # create data array
    data_array = np.fromstring(data.read(), dtype=datatype).reshape(array_dimensions)
    return data_array

# Runs at the start of the program
import sys
if len(sys.argv) != 4:
    print("ERROR: Incorrect number of Arguments. Input Arguements: ",len(sys.argv))
    print("Sys Arg: FILE.PY INPUT_DIRECTORY_IMAGES INPUT_DIRECTORY_LABELS OUTPUT_DIRECTORY")
    exit()
if __name__ == "__main__": main(sys.argv[0],sys.argv[1],sys.argv[2],sys.argv[3])
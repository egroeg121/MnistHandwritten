import numpy as np
import numpy as np

def main(run_dir,input_dir_images,input_dir_labels,output_dir):

    # get np array from ubytes
    print("Loading Data")
    input_np_images = read_ubyte(input_dir_images)
    input_np_labels = read_ubyte(input_dir_labels)
    print("Images data shape: ",input_np_images.shape)
    print("Labels data shape: ", input_np_labels.shape)

    # Get numbers
    num_examples = input_np_images.shape[0]
    images_height = input_np_images.shape[1]
    images_width = input_np_images.shape[2]

    # Process data
    output_np_images = np.expand_dims( downsample(input_np_images,2), axis=3)
    output_np_labels = input_np_labels#

    # write nparray to hdf5
    write_hdf5(output_dir,output_np_images,output_np_labels)

    print("DONE")

    # extract the dimensions of the data from the next bytes

def downsample(array,factor):
    from PIL import Image

    out_array = np.zeros( (array.shape[0],14,14),dtype= np.uint8 )
    for i in range(array.shape[0]):
        im_resized = Image.fromarray(array[0]).resize( (np.uint(array.shape[1]/factor),np.uint(array.shape[2]/factor)),Image.BILINEAR )
        out_array[i] = np.array(im_resized)

    return out_array

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

    # extract the dimensions of the data from the next bytes
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
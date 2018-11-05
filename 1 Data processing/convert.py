'''
GENERAL DESCRIPTION:
    Does data processing on the the data. What kind of data processing is used by the PROCESSING_FUNCTION.py
	Each file converts 2 ubytes files into one hdf5 file containg two datasets: 'images' and 'labels'.


COMMAND LINE ARGUMENTS:
	FILE_NAME PROCESSING_FUNCTION INPUT_IMAGES INPUT_LABELS OUTPUT_DIRECTORY

PROCESSING_FUNCTION:
    must have a process_images taking arguments of array and process_labels taking arguments of an array

'''


def main(run_dir,processing_func,input_dir_images,input_dir_labels,output_dir):
    from FuncLib import hdf5_functions
    from FuncLib import ubyte_functions

    # get np array from ubytes
    print("Loading Data")
    input_np_images = ubyte_functions.read(input_dir_images)
    input_np_labels = ubyte_functions.read(input_dir_labels)
    print("Images data shape: ",input_np_images.shape)
    print("Labels data shape: ", input_np_labels    .shape)

    # Process images
    output_np_images = im_process(input_np_images,processing_func)
    output_np_labels = input_np_labels

    # write nparray to hdf5
    hdf5_functions.write(output_dir,["images","labels"],[output_np_images,output_np_labels])

    print("DONE")
    return output_np_images,output_np_labels

def im_process(input_np_images,processing_func):
    import importlib.util

    spec = importlib.util.spec_from_file_location("module.name", processing_func)
    foo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(foo)
    return foo.process_images(input_np_images)


# Runs at the start of the program
import sys
if __name__ == "__main__":
    import sys
    if len(sys.argv) != 5:
        print("ERROR: Incorrect number of Arguments. Input Arguements: ", len(sys.argv))
        print("Sys Arg: [0]FILE.PY [1]MODEL_PATH [2]TRAIN_DIR_HDF5 [3]TEST_DIR_HDF5 [4]OUTPUT_WEIGHTS_DIR")
        exit()
    main(sys.argv[0],sys.argv[1],sys.argv[2],sys.argv[3],sys.argv[4])
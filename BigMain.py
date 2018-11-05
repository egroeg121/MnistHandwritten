import importlib
data_process = importlib.import_module('.convert',package='1 Data Processing')
#from _1_Data_Processing import convert
import os

run_dir = os.getcwd()
processing_func = "_1_Data_Processing/no_process.py"
input_dir_images = "0 Data/Raw data/t10k-images.idx3-ubyte"
input_dir_labels = "0 Data/Raw data/t10k-labels.idx1-ubyte"
output_dir = "0 Data/Processed data/NoProcessing_test.hdf5"
output_images, ouput_labels = data_process.main(run_dir,processing_func,input_dir_images,input_dir_labels,output_dir)

print("Done")
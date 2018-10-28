0 Data - contains Processed and Raw data. 
1 Data Processing - contains a convert file which is then calls the different processing methods. Inputs the dataset and outputs .hdf5 files
2 Data Visualisation - Contains functions for visiualising the data pre training, or seeing example inputs
3 Models - Contains folders for each model type. Then train.py, evaluate.py and predict.py import these models and output weights or loss/accuracy.
4 Output Visualisation - For visualisation of the spread of outputs, or for loss/accuracy graphs and other metrics



GIT IGNORE:
	.gitignore can be used to ignore large data files. This stops them taking a long for git to push/pull. It may also be confidential and not be uploaded to the git


WORKING DIRECTORY:
	All functions are coded so this is the working directory. Otherwise pulling in and out folders depends on where you are in the folder structure


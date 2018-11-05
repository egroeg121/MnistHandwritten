import h5py



def write(directory,names,data):
    '''
    Writing from HDF5 files

    :param directory: String directory of the hdf5 to read
    :param names: Array of the names to write the datasets to, e.g. ["images","labels"]
    :param data: Array of arrays of the data to write into each datset. Must be in the same order as the names
    '''

    if names.shape[0] != data.shape[0]:
        print("ERROR: Data size and Names Mismatch. Number of Names must match sets of data")
        exit()
    f = h5py.File(directory, "w")

    for i in range(names.shape[0]):
        f.create_dataset(names[i],data=data[i])
    f.close()

def read(directory,names):
    '''
    Reading from an hdf5 file.
    :param directory: The directory of the hdf5 file
    :param names: An array of the names of the datasets in the hdf5 file. Will only return number of datasets it finds
    :return: Returns a list of the datasets
    '''

    import os
    cwd = os.getcwd()
    f = h5py.File(directory, "r")
    keys = list(f.keys())
    out_list = []
    for n in names:
        if n in keys:
            data = f[n]
            out_list.append( f[n])
        else:
            print("Error: Could not find dataset: ", n)
            exit()



    return out_list
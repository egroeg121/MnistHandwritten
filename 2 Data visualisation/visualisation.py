from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


def load_data(input_dir):
    import h5py

    f = h5py.File(input_dir, "r")
    if len(f.keys()) != 2:
        print(" ERROR: Incorrect number of datasets. There should be 2 datasets: 'images and 'labels' ")
        exit()

    images = f['images']
    labels = f['labels']

    return images, labels


def main():
    print("starting")
    dir = "0 Data\\Processed data\\NoProcessing_train.hdf5"
    data_images, data_labels = load_data(dir)

    fig = plt.figure(figsize=(2,5))
    for i in range(10):
        im = Image.fromarray(np.reshape(data_images[i], (28, 28)),'L')
        #im.show()
        fig.add_subplot(2,5,i+1)
        plt.imshow(im)

    plt.show()

if __name__ == "__main__": main()
import matplotlib
import matplotlib.pyplot as plt
from Pillow import Image


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
    data_images, data_labels = load_data("1 Data processing\P")
    im = Image.fromarray(data_images[0])
    #digit = data_images[37000]
    #digit_image = digit.reshape(28, 28)

    #plt.imshow(im, cmap=matplotlib.cm.binary, interpolation="nearest")
    im.show()
    plt.axis("off")
    plt.show()

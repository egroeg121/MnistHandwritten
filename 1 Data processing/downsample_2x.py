
def process_images(array,factor=2):
    import numpy as np
    from PIL import Image

    out_array = np.zeros( (array.shape[0],14,14),dtype= np.uint8 )
    for i in range(array.shape[0]):
        im_resized = Image.fromarray(array[0]).resize( (np.uint(array.shape[1]/factor),np.uint(array.shape[2]/factor)),Image.BILINEAR )
        out_array[i] = np.array(im_resized)

    return np.expand_dims(out_array, axis=3)

def process_labels(array):
    return array
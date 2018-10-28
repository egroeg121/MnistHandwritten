def process_images(array):
    import numpy as np

    return np.expand_dims(array, axis=3)

def process_labels(array):
    return array
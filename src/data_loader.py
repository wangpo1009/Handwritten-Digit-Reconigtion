import numpy as np
import matplotlib.pyplot as plt
import struct
import os

class MNISTDataLoader:
    def __init__(self, data_path):
        self.data_path = data_path

    # Functions to load MNIST dataset
    def load_images(self, file_name):
        full_path = os.path.join(self.data_path, file_name)
        #Open the file in binary mode
        with open(full_path, 'rb') as f:
            #Read the header information (magic number, number of images, rows, columns) in 16 first bytes
            magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
            if magic != 2051:
                raise ValueError('Invalid magic number in MNIST image file: {}'.format(magic))
            #The images are stored as unsigned bytes, which mean that there 60000 * 784 bytes after the header that represent the pixel val of each image.
            # We can read all the pixel values at once using numpy's fromfile function, and then reshape the resulting array to have the correct dimensions (num, rows * cols). 
            # Finally, we normalize the pixel values to be between 0 and 1 by dividing by 255.0. 
            images = np.fromfile(f, dtype=np.uint8).reshape(num, rows * cols)
        return images / 255.0
    
    # Functions to load MNIST labels, similar to loading images, but with a different magic number and data format. 
    # The labels are stored as unsigned bytes, and we can read them using numpy's fromfile function after verifying the magic number in the header. 
    # The resulting array will contain the label for each image in the dataset.
    def load_labels(self, file_path):
        full_path = os.path.join(self.data_path, file_path)
        with open(full_path, 'rb') as f:
            magic, num = struct.unpack(">II", f.read(8))
            if magic != 2049:
                raise ValueError('Invalid magic number in MNIST label file: {}'.format(magic))
            labels = np.fromfile(f, dtype=np.uint8)
        return labels



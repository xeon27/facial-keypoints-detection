# Import libraries
import numpy as np
import os
import pandas as pd
import cv2

from torch.utils.data import Dataset


def read_image(path):
    """ Read an image and convert to RGB"""
    image = cv2.imread(path)
    
    # Convert to RGB since cv2 reads as BGR
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    return image
    
    
def read_file(file_path, headers=False):
    """ Read csv file using python """
    data = []
    with open(file_path, "r") as f:
        for line in f.readlines():
            data.append(line.strip().split(","))
        
    # Remove line containing headers
    if headers:
        data = data[1:]
        
    return data
    
    
def read_data(path, image_dir):
    """ 
        Read the data from csv file 
        Parameters:
        path (string): path for the csv file
        image_dir (string): directory containing the image files
        Returns:
        data (list): list of dictionaries with two keys - image and key_pts
    """
    # Read into a list
    data_list = read_file(path, headers=True)
    
    # Initialise empty list for image, key points dict
    data = []
    
    # Iterate over each elm of the list
    for elm in data_list:
        # Read image name (first elm of the list)
        image = elm[0]
        # Read image using image name
        image = read_image(os.path.join(image_dir, image))
        
        # Read corresponding kep points as an array of shape (136,)
        key_pt = np.array(elm[1:])
        # Reshape to (68, 2)
        key_pt = np.reshape(key_pt, (-1, 2))

        data.append({"image": image, "key_pts": key_pt})
        
    return data
    
    
# Class for Facial Keypoints data
class FacialKeypointsDataset(Dataset):
    def __init__(self, data_path, image_dir):
        super(FacialKeypointsDataset, self).__init__()
        self.data = read_data(data_path, image_dir)
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        return self.data[idx]



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
    

def read_data(path, image_path):
    """ Read the data from csv file """
    # Read into pandas dataframe
    df = pd.read_csv(path)
    
    # Initialise empty list for image, key points dict
    data = []
    
    # Iterate over each row of the df
    for idx in range(df.shape[0]):
        # Read image name in first column
        image = df.iloc[idx, 0]
        # Read image using image name
        image = read_image(os.path.join(image_path, image))
        
        # Read corresponding kep points as an array of shape (136,)
        key_pt = np.array(df.iloc[idx, 1:])
        # Reshape to (68, 2)
        key_pt = np.reshape(key_pt, (-1, 2))

        data.append({"image": image, "key_pts": key_pt})
        
    return data
    
    
# Class for Facial Keypoints data
class FacialKeypointsDataset(Dataset):
    def __init__(self, data_path, image_path):
        super(FacialKeypointsDataset, self).__init__()
        self.data = read_data(data_path, image_path)
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        return self.data[idx]



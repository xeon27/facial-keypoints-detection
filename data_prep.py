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
        
        # Read corresponding kep points and convert from string to float
        key_pt = [float(p) for p in elm[1:]]
        # Convert to an array of shape (136,)
        key_pt = np.array(key_pt)
        # Reshape to (68, 2)
        key_pt = np.reshape(key_pt, (-1, 2))

        data.append({"image": image, "key_pts": key_pt})
        
    return data
    
    
# Class for Facial Keypoints data
class FacialKeypointsDataset(Dataset):
    def __init__(self, data_path, image_dir, transform=None):
        super(FacialKeypointsDataset, self).__init__()
        self.image_dir = image_dir
        self.data = read_data(data_path, image_dir)
        self.transform = transform
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        data = self.data[idx]
        
        # Read image using image name
        data['image'] = read_image(os.path.join(self.image_dir, data['image']))
        
        # Apply transform if exist
        if self.transform:
            data = self.transform(data)
        
        return data


# Class for rescaling the image
class Rescale():
    def __init__(self, size):
        self.size = size
        self.relative = False
        # Size as a single value
        if isinstance(size, (int, float)):
            self.relative = isinstance(size, float)
        # Size as a tuple (h, w)
        elif isinstance(size, tuple):
            self.relative = isinstance(size[0], float)
        else:
            raise TypeError("Input size can either be an int, float or a tuple")
            
            
    def __call__(self, sample_data):
        # Fetch image and key point data from dict
        image = sample_data['image']
        key_pts = sample_data['key_pts']
        
        # Convert image to gray scale
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Fetch original image shape
        (org_h, org_w) = image.shape
            
        # Obtain new height and width
        if isinstance(self.size, (int, float)):
            if not self.relative:
                # Assign value to smaller side
                if org_h <= org_w:
                    h = self.size
                    w = int(org_w * (h/org_h))
                else:
                    w = self.size
                    h = int(org_h * (w/org_w))
            else:
                h = int(org_h * self.size)
                w = int(org_w * self.size)
        elif isinstance(self.size, tuple):
            if not self.relative:
                h = self.size[0]
                w = self.size[1]
            else:
                h = int(org_h * self.size[0])
                w = int(org_w * self.size[1])
                     
        # Resize image
        image = cv2.resize(image, (w, h))
        
        # Scale key points based on the resized image
        key_pts[:, 0] = key_pts[:, 0] * (w/org_w)
        key_pts[:, 1] = key_pts[:, 1] * (h/org_h)
        
        return {"image": image, "key_pts": key_pts}
        
        
# Class for normalizing image and key points data
class Normalize():
    def __call__(self, sample_data):
        # Fetch image and key point data from dict
        image = sample_data['image']
        key_pts = sample_data['key_pts']
        
        # Normalize pixel values to (0,1)
        image = image/255.0
        
        # Normalize key point co-ordinates to (-1,1)
        size = image.shape[0]
        key_pts = (key_pts-(size/2))/(size/4)
        
        return {"image": image, "key_pts": key_pts}
        
       
# Class for cropping image using a random window of given size
class RandomCrop():
    def __init__(self, size):
        self.size = size
        
    
    def __call__(self, sample_data):
        # Fetch image and key point data from dict
        image = sample_data['image']
        key_pts = sample_data['key_pts']
        
        # Fetch original image shape
        (h, w) = image.shape
        
        if isinstance(self.size, (int, float)):
            # Square crop
            if isinstance(self.size, float):
                # Fraction of smaller side
                crop_size = int(self.size * (h if h <= w else w))
            else:
                crop_size = self.size
        
        # Generate start indices for both dims
        try:
            start_x = np.random.randint(0, int(w - crop_size))
            start_y = np.random.randint(0, int(h - crop_size))
            end_x = start_x + crop_size
            end_y = start_y + crop_size
        except ValueError as value_error:
            print("Error: Crop size cannot be greater than original image size")
            return {}
        
        # Crop the image
        image = image[start_y: end_y, start_x: end_x]    
            
        # Adjust the key points
        key_pts[:, 0] = key_pts[:, 0] - start_x
        key_pts[:, 1] = key_pts[:, 1] - start_y
        
        return {"image": image, "key_pts": key_pts}
        
        

        
        
        
        
        
        
    


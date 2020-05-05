import pandas as pd
import matplotlib.pyplot as plt

from data_prep import FacialKeypointsDataset

train_path = "./data/training_frames_keypoints.csv"
valid_path = "./data/test_frames_keypoints.csv"
train_image_path = "./data/training"

train_data = FacialKeypointsDataset(train_path, train_image_path)

print(len(train_data))
# print(train_data[36]['image'].shape)
# print(train_data[36]['key_pts'].shape)

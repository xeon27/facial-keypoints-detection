import pandas as pd
import matplotlib.pyplot as plt

from torchvision import transforms
from data_prep import FacialKeypointsDataset
from data_prep import Rescale, Normalize, RandomCrop

train_path = "./data/training_frames_keypoints.csv"
valid_path = "./data/test_frames_keypoints.csv"
train_image_path = "./data/training"

all_transforms = transforms.Compose([Rescale(250), RandomCrop(224), Normalize()])
train_data = FacialKeypointsDataset(train_path, train_image_path, transform=all_transforms)

sample_idx = 567
sample = train_data[sample_idx]

print(sample['image'].shape)

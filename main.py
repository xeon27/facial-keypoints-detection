# Import libraries
from torchvision import transforms
from torch.utils.data import DataLoader

from utils.visualize_utils import visualize_keypoints
from utils.data_utils import FacialKeypointsDataset, Rescale, Normalize, RandomCrop, Rotate, ToTensor


train_path = "./data/training_frames_keypoints.csv"
valid_path = "./data/test_frames_keypoints.csv"
train_image_path = "./data/training"

all_transforms = transforms.Compose([Rescale(250), RandomCrop(224), Normalize(), ToTensor()])
train_data = FacialKeypointsDataset(train_path, train_image_path, transform=all_transforms, augment=[Rotate(180)])

print(len(train_data))
# sample_idx = 34
# sample = train_data[sample_idx]
# print(type(sample['image']))
# print(sample['key_pts'].shape)
# print(sample['image'].shape)
# visualize_keypoints(sample['image'], sample['key_pts'])

train_loader = DataLoader(train_data, batch_size=10, shuffle=True)
print(len(train_loader))

for batch in train_loader:
    print(batch['image'].shape)
    print(batch['key_pts'].shape)
    break



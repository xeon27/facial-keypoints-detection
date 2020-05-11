# Import libraries
from torchvision import transforms
from torch.utils.data import DataLoader
from torch import nn, optim

from utils.visualize_utils import visualize_keypoints
from utils.data_utils import FacialKeypointsDataset, Rescale, Normalize, RandomCrop, Rotate, ToTensor

from model import FKNet


train_path = "./data/training_frames_keypoints.csv"
valid_path = "./data/test_frames_keypoints.csv"
train_image_path = "./data/training"

all_transforms = transforms.Compose([Rescale(250), RandomCrop(224), Normalize(), ToTensor()])
train_data = FacialKeypointsDataset(train_path, train_image_path, transform=all_transforms, augment=[])

print(len(train_data))
# sample_idx = 34
# sample = train_data[sample_idx]
# print(type(sample['image']))
# print(sample['key_pts'].shape)
# print(sample['image'].shape)
# visualize_keypoints(sample['image'], sample['key_pts'])

train_loader = DataLoader(train_data, batch_size=10, shuffle=True)
print(len(train_loader))

# for batch in train_loader:
    # print(batch['image'].shape)
    # print(batch['key_pts'].shape)
    # break
    
    
# Define the network
net = FKNet()
print(net)


criterion = nn.SmoothL1Loss()
optimizer = optim.Adam(net.parameters(), lr=0.001)

# input = next(iter(train_loader))
# optimizer.zero_grad()

# output = net(input['image'])
# print(output.shape)




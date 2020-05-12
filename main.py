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
train_data = FacialKeypointsDataset(train_path, train_image_path, truncate=500, transform=all_transforms, augment=[])

print(len(train_data))

batch_size = 10
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=False)


# Define the network
net = FKNet()
# print(net)


criterion = nn.SmoothL1Loss()
optimizer = optim.Adam(net.parameters(), lr=0.001)


num_epochs = 2

print("Training started . . .")
for epoch in range(num_epochs):
    # For loss in each epoch
    running_loss = 0.0
    
    for batch_index, batch in enumerate(train_loader):
        # Reset the gradients
        optimizer.zero_grad()
        
        # Fetch image and target keypoints
        image = batch['image']
        target = batch['key_pts']
        
        # Resize to (batch_size, 136)
        target = target.view(target.size()[0], -1)
        
        # Pass through network
        output = net(image)
        
        # Calculate loss
        loss = criterion(output, target)
        batch_loss = loss.item()
        running_loss += batch_loss
        
        # Update weights
        optimizer.step()
        
        # Print average loss
        if (batch_index + 1) % 10 == 0:
            print("Epoch {}, Batch {}, Avg. loss: {}".format(epoch+1, batch_index+1, batch_loss))
        
    print("Epoch {} complete".format(epoch+1))
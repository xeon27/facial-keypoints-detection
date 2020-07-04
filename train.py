# Import libraries
import argparse
import os
from datetime import datetime

import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from torch import nn, optim

from models.model import FKNet
from utils.visualize_utils import visualize_keypoints
from utils.data_utils import FacialKeypointsDataset, Rescale, Normalize, RandomCrop, Rotate, ToTensor


def main(args):

    # Fetch required arguments
    train_path = args.train_path
    valid_path = args.valid_path
    train_image_path = args.train_data
    valid_image_path = args.valid_data
    out_path = args.out_path
    trunc_data = args.trunc_data
    
    input_size = args.input_size
    aug_angles = args.augment
    batch_size = args.batch_size
    num_epochs = args.epochs
    lr = args.learning_rate
    print_every = args.print_every
    
    
    # Combine all image transformations
    all_transforms = transforms.Compose([Rescale(int(1.1*input_size)), RandomCrop(input_size), Normalize(), ToTensor()])
    
    # Set image augmentation if required
    aug_transforms = None
    if aug_angles:
        aug_transforms = [Rotate(angle) for angle in aug_angles]
    
    train_data = FacialKeypointsDataset(train_path, train_image_path, truncate=trunc_data, transform=all_transforms, augment=aug_transforms)
    valid_data = FacialKeypointsDataset(valid_path, valid_image_path, truncate=trunc_data, transform=all_transforms, augment=None)

    print("Train data: {}".format(len(train_data)))
    print("Valid data: {}".format(len(valid_data)))

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_data, batch_size=batch_size, shuffle=True)
    data_loader = {"train": train_loader, "valid": valid_loader}

    # debug
    sample = next(iter(train_loader))
    print(sample['image'].size())
    print(sample['key_pts'].size())
    
    # Define the network
    net = FKNet()
    print(net)

    criterion = nn.SmoothL1Loss()
    optimizer = optim.SGD(net.parameters(), lr=lr)
    
    # Create folder for log and model output
    if not os.path.isdir(out_path):
        os.mkdir(out_path)
    with open(os.path.join(out_path, './logs.txt'), 'a') as f:
        f.write("Start time: {} \n".format(datetime.now()))
        
    avg_epoch_loss = {"train": 0.0, "valid": 0.0}
    
    print("Training started . . .")
    for epoch in range(num_epochs):
    
        for mode in ['train', 'valid']:
            running_loss = 0.0
            
            if mode == 'train':
                net.train()
            else:
                net.eval()
            
            for batch_index, batch in enumerate(data_loader[mode]):
                # Reset the gradients
                if mode == 'train':
                    optimizer.zero_grad()
                
                # Fetch image and target keypoints
                image = batch['image']
                target = batch['key_pts']
                
                # Resize to (batch_size, 136)
                target = target.view(target.size(0), -1)
                
                # Calculate gradients only during training
                with torch.set_grad_enabled(mode == 'train'):
                    # Pass through network
                    output = net(image)
                                   
                    # Calculate loss
                    loss = criterion(output, target)
                    
                    # Update weights
                    if mode == 'train':
                        # print(net.conv1.weight.data[0])
                        loss.backward()
                        # print(net.conv1.weight.grad.data[0])
                        optimizer.step()
                        # print(net.conv1.weight.data[0])
                        # break
                    
                running_loss += loss.item()
                
                # Print average loss
                if (batch_index + 1) % print_every == 0:
                    print("Epoch {}, Batch {}, Avg. {} loss: {}".format(epoch+1, batch_index+1, mode, loss.item()))
            
            # break
            # Average train/valid loss for each epoch
            avg_epoch_loss[mode] = running_loss/len(data_loader[mode])
        
        ckpt = "Epoch {} complete, Avg. train loss: {}, Avg. valid loss: {} \n".format(epoch+1, avg_epoch_loss['train'], avg_epoch_loss['valid'])
        print(ckpt)
        # Write to log file
        with open(os.path.join(out_path, './logs.txt'), 'a') as f:
            f.write(ckpt)
        
    
    print("Training complete.")
    
    with open(os.path.join(out_path, './logs.txt'), 'a') as f:
        f.write("End time: {}".format(datetime.now()))
    
    # Save model
    torch.save({"epochs": (epoch+1), "model": net.state_dict()}, os.path.join(out_path, './model.pth'))
        
        
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--train_path", type=str, default="./data/training_frames_keypoints.csv", 
                        help="Path to file for train annotations")
    parser.add_argument("--valid_path", type=str, default="./data/test_frames_keypoints.csv", 
                        help="Path to file for valid annotations")
    parser.add_argument("--train_data", type=str, default="./data/training", 
                        help="Path to train set of image data")
    parser.add_argument("--valid_data", type=str, default="./data/test", 
                        help="Path to valid set of image data")
    parser.add_argument("--out_path", type=str, default="./snap", help="Path to save model snapshots")
    parser.add_argument("--trunc_data", type=int, default=None, help="sample data to use for debug") 
    
    parser.add_argument("--input_size", type=int, default=224, help="Image input size for model")
    parser.add_argument("--augment", type=int, nargs='+', default=None, 
                        help="List of angles in degrees to augment rotated image data")
    parser.add_argument("--batch_size", type=int, default=10, help="Batch size for training")
    parser.add_argument("-lr", "--learning_rate", type=float, default=0.001, help="Learning rate for train event")
    parser.add_argument("--epochs", type=int, default=1, help="No. of training epochs")
    parser.add_argument("--print_every", type=int, default=10, help="No. of batches complete before printing loss")
    
    args = parser.parse_args()
    
    main(args)
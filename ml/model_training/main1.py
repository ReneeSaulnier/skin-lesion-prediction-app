import yaml
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from torchvision.io import read_image
from sklearn.model_selection import train_test_split



# Load the config file
config_path = "config/config.yaml"
with open(config_path, "r") as file:
    config = yaml.safe_load(file)

# Image paths config
image_path_1 = config['data_processing']['path']['image_path_1']
image_path_2 = config['data_processing']['path']['image_path_2']
metadata_path = config['data_processing']['path']['metadata_path']
image_output_path = config['data_processing']['path']['image_path']
combine_image_flag = config['data_processing']['merge_folder']
# Dataset size config
validation_dataset_size = config['data_processing']['dataset_size']['validation']
test_dataset_size = config['data_processing']['dataset_size']['test']
# Model config
model_output_path = config['model_training']['model']['output_path']

# Combine the images into one folder <<< Run this only once, set the flag in config to False after running >>>
def combine_images(image_path_1, image_path_2):
    for file_name in os.listdir(image_path_1):
        if file_name.endswith(('.jpg', '.png', '.jpeg')):
            image_path = os.path.join(image_path_1, file_name)
            new_image_path = os.path.join(image_path_2, file_name)
            os.rename(image_path, new_image_path)

    # Rename the folder to images
    os.rename(image_path_2, image_output_path)
    # Delete the original folder
    os.rmdir(image_path_1)
    print('Images combined into one folder')
    return image_output_path

if combine_image_flag:
    final_image_path = combine_images(image_path_1, image_path_2)
else:
    final_image_path = image_output_path

class_to_idx = {'akiec': 0, 'bcc': 1, 'bkl': 2, 'df': 3, 'mel': 4, 'nv': 5, 'vasc': 6}

# Load the dataset
class SkinCancerDataset(Dataset):
    """
    A custom dataset class for the skin cancer dataset.
    """
    # Runs once to instantiate the object
    def __init__(self, label_dir, image_dir, transform=None, target_transform=None):
        self.image_labels = label_dir
        self.image_dir = image_dir
        self.transform = transform
        self.target_transform = target_transform

    # Returns the number of 'samples' in the dataset
    def __len__(self):
        return len(self.image_labels)
    
    # Return a single sample from the dataset at the given idx.
    # It will then locate it on disk, convert it to a tensor with 'read_image',
    # then gets the corresponding label and then returns a tensor image and label tuple.
    def __getitem__(self, idx):
        image_path = os.path.join(final_image_path, self.image_labels.iloc[idx, 0])

        # read_image is default an 8 bit int. We need to convert to 
        # float (32 bit) and normalize to [0, 1]
        image = read_image(image_path).float() / 255.0
        label = class_to_idx[self.image_labels.iloc[idx, 1]]

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label


def build_df(folder_path, metadata_path):
    # Get the metadata for the images to retrieve the labels
    metadata = pd.read_csv(metadata_path)
    image_list = []

    # Get the image path and add it to a dataframe
    for file_name in os.listdir(folder_path):
        if file_name.lower().endswith(('.jpg', '.png', '.jpeg')):
            image_list.append({"image_path": file_name})

    image_files_df = pd.DataFrame(image_list)

    # Add the extension to the image id << hardcoded for now >>
    metadata['image_id'] = metadata['image_id'] + '.jpg'

    # Merge the metadata labels with the image files
    merged_df = pd.merge(
        image_files_df,
        metadata,
        left_on='image_path',
        right_on='image_id',
        how='inner'
    )

    # Only want the image path and the label
    merged_df = merged_df[['image_id', 'dx']]
    return merged_df

image_df = build_df(final_image_path, metadata_path)

# Split the images for train/val/test
X = image_df['image_id']
y = image_df['dx']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_dataset_size, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=validation_dataset_size, random_state=42)

train_df = pd.DataFrame({'image_path': X_train, 'dx': y_train})
val_df = pd.DataFrame({'image_path': X_val, 'dx': y_val})
test_df = pd.DataFrame({'image_path': X_test, 'dx': y_test})

# Data loader for the dataset, loading in batches and avioding loops for efficiency
train_dataloader = SkinCancerDataset(train_df, final_image_path)
validation_dataloader = SkinCancerDataset(val_df, final_image_path)
test_dataloader = SkinCancerDataset(test_df, final_image_path)

# Load the data in batches
train_loader = DataLoader(train_dataloader, batch_size=32, shuffle=True)
validation_loader = DataLoader(validation_dataloader, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataloader, batch_size=32, shuffle=True)

# Create the model
class Cnn(nn.Module):
    def __init__(self):
        super(Cnn, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 109 * 147, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 7)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
model = Cnn()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
for epoch in range(5):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 1000 == 999:
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

print('Finished Training')

# Save the model
PATH = os.path.join(model_output_path, 'custom', 'custom_model.pth')
torch.save(model.state_dict(), PATH)

# Test the model
correct = 0
total = 0
with torch.no_grad():
    for data in test_loader:
        images, labels = data
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the test images: %d %%' % (
    100 * correct / total))

import yaml
import os
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torch.optim as optim
from torchvision import transforms
from torchvision.io import read_image

# Load the config file
config_path = "config/config.yaml"
with open(config_path, "r") as file:
    config = yaml.safe_load(file)

# Image paths config
image_input_path = config['model_training']['path']['image_path']
train_image_path = config['model_training']['path']['train_data']
val_image_path = config['model_training']['path']['val_data']
test_image_path = config['model_training']['path']['test_data']

# Model config
model_output_path = config['model_training']['model']['output_path']
model_output_name = config['model_training']['model']['output_name']

# Transform the images for model training
train_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(20),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load the dataset
class SkinCancerDataset(Dataset):
    """
    A custom class for the skin cancer dataset.
    """
    # Runs once to instantiate the object
    def __init__(self, label_dir, image_dir, transform=None):
        self.image_labels = label_dir
        self.image_dir = image_dir
        self.transform = transform
        self.class_to_idx = {'akiec': 0, 'bcc': 1, 'bkl': 2, 'df': 3, 'mel': 4, 'nv': 5, 'vasc': 6}

    # Returns the number of 'samples' in the dataset
    def __len__(self):
        return len(self.image_labels)
    
    # Return a single sample from the dataset at the given idx.
    # It will then locate it on disk, convert it to a tensor with 'read_image',
    # then gets the corresponding label and then returns a tensor image and label tuple.
    def __getitem__(self, idx):
        # Get the image path from the first column of the dataframe
        image_path = os.path.join(image_input_path, self.image_labels.iloc[idx, 0])

        # read_image is default an 8 bit int. We need to convert to 
        # float (32 bit) and normalize to [0, 1]
        image = read_image(image_path)
        # Get the label from the second column of the dataframe
        label_str = self.image_labels.iloc[idx, 1]
        label = self.class_to_idx[label_str]

        if self.transform:
            image = self.transform(image)
        return image, label

# Load the data
train_df = pd.read_csv(train_image_path)
val_df = pd.read_csv(val_image_path)

# Data loader for the dataset, loading in batches and avioding loops for efficiency
train_dataset = SkinCancerDataset(train_df, image_input_path, transform=train_transform)
validation_dataset = SkinCancerDataset(val_df, image_input_path, transform=val_transform)

# Load the data in batches
train_loader = DataLoader(train_dataset, batch_size=32)
validation_loader = DataLoader(validation_dataset, batch_size=32, shuffle=False)

# Create the model
class Cnn(nn.Module):
    def __init__(self):
        super(Cnn, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 53 * 53, 120)
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
        if i % 500 == 0:
            print()
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

print('Finished Training')

# Save the model
PATH = os.path.join(model_output_path, model_output_name)
torch.save(model.state_dict(), PATH)

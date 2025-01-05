import yaml
import os
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from torch import nn
import torch.nn.functional as F
from torchvision.io import read_image
from sklearn.metrics import (
    accuracy_score, precision_score,
    recall_score, f1_score
)

# Load config
config_path = "config/config.yaml"
with open(config_path, "r") as file:
    config = yaml.safe_load(file)

processed_path = config['data_processing']['path']['processed_path']
model_path = config['model_validation']['model']['path']
model_name = config['model_validation']['model']['name']

image_input_path = config['model_training']['path']['image_path']
test_image_path = config['model_training']['path']['test_data']

metrics_output_path = config['model_validation']['logs']['metrics']

# Load the dataset
class SkinCancerDataset(Dataset):
    """
    A custom class for the skin cancer dataset.
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
        class_to_idx = {'akiec': 0, 'bcc': 1, 'bkl': 2, 'df': 3, 'mel': 4, 'nv': 5, 'vasc': 6}
        # Get the image path from the first column of the dataframe
        image_path = os.path.join(image_input_path, self.image_labels.iloc[idx, 0])

        # read_image is default an 8 bit int. We need to convert to 
        # float (32 bit) and normalize to [0, 1]
        image = read_image(image_path).float() / 255.0
        # Get the label from the second column of the dataframe
        label = class_to_idx[self.image_labels.iloc[idx, 1]]

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label
    
# Load the data
test_df = pd.read_csv(test_image_path)

# Define the model architecture (same as during training)
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
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Instantiate the model
model = Cnn()

# Load the saved state_dict
model_file = os.path.join(model_path, model_name)
model.load_state_dict(torch.load(model_file))

# Set the model to evaluation mode
model.eval()

# Load the test dataset
test_dataloader = SkinCancerDataset(test_df, image_input_path)
test_loader = DataLoader(test_dataloader, batch_size=32, shuffle=False)

# Define the evaluation metrics
def evaluate_model(model, dataloader):
    y_true = []
    y_pred = []
    with torch.no_grad():  # No need for gradients during evaluation
        for images, labels in dataloader:
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            y_true.extend(labels.numpy())
            y_pred.extend(predicted.numpy())
    
    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')

    # Save the metrics
    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }
    metrics_df = pd.DataFrame(metrics, index=[0])
    metrics_df.to_csv(os.path.join(metrics_output_path, 'metrics.csv'), index=False)
    print(f"Metrics saved to {metrics_output_path}")

# Evaluate the model
evaluate_model(model, test_loader)

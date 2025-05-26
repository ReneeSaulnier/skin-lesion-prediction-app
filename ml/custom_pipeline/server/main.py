from fastapi import FastAPI, Query
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F
import torch.nn as nn
import torch
import os
import yaml

# Load the config file
config_path = "config/config.yaml"
with open(config_path, "r") as file:
    config = yaml.safe_load(file)

# Model config
model_folder = config['server']['model']['path']
model_name = config['server']['model']['custom_model']

# Load the model and processor
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load custom model and processor
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
    
model_path = os.path.join(model_folder, model_name)
model = Cnn()
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# Preprocess the image
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Initialize FastAPI
app = FastAPI()


def get_class_names(predicted_class_number):
    """
    Returns the class names for the model.

    Index	Class Name	                    Description
    0	    Melanoma	                    Dangerous form of skin cancer.
    1	    Melanocytic Nevus	            Common mole (benign).
    2	    Basal Cell Carcinoma	        Slow-growing form of skin cancer.
    3	    Actinic Keratosis	            Pre-cancerous lesion.
    4	    Benign Keratosis-like Lesion	Non-cancerous keratosis.
    5	    Dermatofibroma	                Benign fibrous skin lesion.
    6	    Vascular Lesion	                Blood vessel-related lesion.
    """
    class_names = {
        0: "Melanoma",
        1: "Melanocytic Nevus",
        2: "Basal Cell Carcinoma",
        3: "Actinic Keratosis",
        4: "Benign Keratosis-like Lesion",
        5: "Dermatofibroma",
        6: "Vascular Lesion",
    }

    return class_names[predicted_class_number]

# Endpoint to predict from an image path
@app.post("/api/predict")
async def predict(image_path: str = Query(..., description="Path to the image file on the server")):
    """
    Accepts an image file path as input and returns the predicted class name.
    """
    # Check if file exists
    if not os.path.exists(image_path):
        return {"error": "File does not exist at the specified path", "path": image_path}
    
    try:
        # Open the image
        image = Image.open(image_path).convert("RGB")

        # Preprocess the image
        img_tensor = transform(image).unsqueeze(0).to(device)

        # Run inference
        with torch.no_grad():
            outputs = model(img_tensor)
            predicted_class = torch.argmax(outputs, dim=-1).item()
            probabilities = F.softmax(outputs, dim=-1).cpu().numpy()

        # Add confidence score
        confidence = float(probabilities[0][predicted_class])
        # Get the class name
        predicted_class_name = get_class_names(predicted_class)
        probability = {get_class_names(i): float(probabilities[0][i]) for i in range(len(probabilities[0]))}

        return {"predicted_class": predicted_class_name, "confidence": confidence, "probabilities": probability}

    except Exception as e:
        return {"error": f"Failed to process the image: {str(e)}"}
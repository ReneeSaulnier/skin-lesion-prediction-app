from fastapi import FastAPI, Query
from transformers import AutoModelForImageClassification, AutoImageProcessor
from PIL import Image
import torch.nn.functional as F
import torch
import os
import yaml

# Load the config file
config_path = "config/config.yaml"
with open(config_path, "r") as file:
    config = yaml.safe_load(file)

# Model config
model_folder = config['server']['model']['path']
model_name = config['server']['model']['name']

# Load the model and processor
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model and processor
model = AutoModelForImageClassification.from_pretrained(os.path.join(model_folder, model_name))
processor = AutoImageProcessor.from_pretrained(os.path.join(model_folder, model_name))
model.to(device)
model.eval()

# Initialize FastAPI app
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
        inputs = processor(images=image, return_tensors="pt").to(device)

        # Run inference
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            predicted_class = torch.argmax(logits, dim=-1).item()
            probabilities = F.softmax(logits, dim=-1).cpu().numpy()


        # Add confidence score
        confidence = float(probabilities[0][predicted_class])
        # Get the class name
        predicted_class_name = get_class_names(predicted_class)

        return {"predicted_class": predicted_class_name, "confidence": confidence}

    except Exception as e:
        return {"error": f"Failed to process the image: {str(e)}"}
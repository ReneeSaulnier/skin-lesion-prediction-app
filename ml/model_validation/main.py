import yaml
import os
import numpy as np
import pandas as pd
import torch
from PIL import Image
from transformers import ResNetForImageClassification, AutoImageProcessor
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc

# Load the config file
config_path = "config/config.yaml"
with open(config_path, "r") as file:
    config = yaml.safe_load(file)

model_path = config['model_validation']['model']['path']
model_name = config['model_validation']['model']['name']
test_data_path = config['model_validation']['test_data']['path']
metrics_path = config['model_validation']['logs']['metrics']
roc_curve_path = config['model_validation']['logs']['roc_curve']


X_test_path = os.path.join(test_data_path, 'X_test.npy')
y_test_csv_path = os.path.join(test_data_path, 'y_test.csv')
df = pd.read_csv(y_test_csv_path)
y_test = df['dx'].values


model = ResNetForImageClassification.from_pretrained(os.path.join(model_path, model_name))
processor = AutoImageProcessor.from_pretrained(os.path.join(model_path, model_name))

model.eval()

# Load the test data
X_test = np.load(X_test_path, allow_pickle=True)

images = [Image.open(img_path).convert("RGB") for img_path in X_test]

# Preprocess all images at once (or in batches):
inputs = processor(images=images, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits
    y_pred = logits.argmax(dim=1)

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')
conf_matrix = confusion_matrix(y_test, y_pred)

# Save metrics
with open(metrics_path, 'w') as f:
    f.write(f"Accuracy: {accuracy}\n")
    f.write(f"Precision: {precision}\n")
    f.write(f"Recall: {recall}\n")
    f.write(f"F1 Score: {f1}\n")
    f.write(f"Confusion Matrix:\n{conf_matrix}\n")

# Save ROC curve
fpr, tpr, _ = roc_curve(y_test, y_pred)
roc_auc = auc(fpr, tpr)
np.savez(roc_curve_path, fpr=fpr, tpr=tpr, roc_auc=roc_auc)

print("Validation complete.")


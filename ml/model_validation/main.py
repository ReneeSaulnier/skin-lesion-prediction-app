import yaml
import os
import numpy as np
import pandas as pd
import torch
from PIL import Image
from transformers import ResNetForImageClassification, AutoImageProcessor
from torch.utils.data import DataLoader
from sklearn.preprocessing import label_binarize
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score,
    recall_score, f1_score, roc_curve, auc
)

# 1. Load config
config_path = "config/config.yaml"
with open(config_path, "r") as file:
    config = yaml.safe_load(file)

processed_path = config['data_processing']['path']['processed_path']
model_path = config['model_validation']['model']['path']
model_name = config['model_validation']['model']['name']

X_test_path = os.path.join(processed_path, 'X_test.npy')
y_test_path = os.path.join(processed_path, 'y_test.csv')
metrics_path = config['model_validation']['logs']['metrics']
roc_curve_path = config['model_validation']['logs']['roc_curve']

# 2. Load test data
X_test = np.load(X_test_path, allow_pickle=True)
y_test = pd.read_csv(y_test_path)['dx'].values

# 3. Convert class labels to numeric
class_names = np.unique(y_test)            # e.g., ['akiec', 'bcc', 'bkl', ...]
class_to_idx = {cls: i for i, cls in enumerate(class_names)}
idx_to_class = {v: k for k, v in class_to_idx.items()}  # for inverse mapping
y_test_numeric = np.array([class_to_idx[label] for label in y_test], dtype=np.int64)

# 4. Load model and processor
model = ResNetForImageClassification.from_pretrained(os.path.join(model_path, model_name))
processor = AutoImageProcessor.from_pretrained(os.path.join(model_path, model_name))

# 5. Create Dataset & DataLoader
class SkinCancerDataset(torch.utils.data.Dataset):
    def __init__(self, image_paths, labels, processor):
        self.image_paths = image_paths
        self.labels = labels
        self.processor = processor

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # Load and preprocess the image dynamically
        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt")
        
        return {
            'pixel_values': inputs['pixel_values'].squeeze(0),  # Remove batch dimension
            'labels': torch.tensor(label, dtype=torch.long)
        }

test_dataset = SkinCancerDataset(X_test, y_test_numeric, processor)

test_dataloader = DataLoader(
    test_dataset,
    batch_size=16,  # Adjust as per your GPU/CPU memory
    shuffle=False
)

# 6. Run model inference on test set
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

all_preds = []
all_labels = []
all_logits = []

with torch.no_grad():
    for batch in test_dataloader:
        pixel_values = batch["pixel_values"].to(device)  # shape: (B, 3, H, W)
        labels = batch["labels"].to(device)              # shape: (B,)
        
        outputs = model(pixel_values)      # forward pass
        logits = outputs.logits            # shape: (B, num_classes)
        
        # Get class predictions
        preds = torch.argmax(logits, dim=-1)  # shape: (B,)
        
        # Save results
        all_preds.append(preds.cpu().numpy())
        all_labels.append(labels.cpu().numpy())
        all_logits.append(logits.cpu().numpy())

# 7. Flatten the lists
all_preds = np.concatenate(all_preds)
all_labels = np.concatenate(all_labels)
all_logits = np.concatenate(all_logits)  # you might need this for ROC

# 8. Compute metrics
acc = accuracy_score(all_labels, all_preds)
precision = precision_score(all_labels, all_preds, average='weighted')
recall = recall_score(all_labels, all_preds, average='weighted')
f1 = f1_score(all_labels, all_preds, average='weighted')

print("Test Accuracy:   ", acc)
print("Test Precision:  ", precision)
print("Test Recall:     ", recall)
print("Test F1:         ", f1)

# 9. Confusion Matrix
cm = confusion_matrix(all_labels, all_preds)
print("Confusion Matrix:\n", cm)

num_classes = len(class_names)
# Binarize the labels for each class
y_test_binarized = label_binarize(all_labels, classes=list(range(num_classes)))

# We also need the predicted probabilities (softmax) for each class
softmaxes = np.exp(all_logits) / np.exp(all_logits).sum(axis=1, keepdims=True)

fpr_dict, tpr_dict, roc_auc_dict = {}, {}, {}
for i in range(num_classes):
    fpr_dict[i], tpr_dict[i], _ = roc_curve(y_test_binarized[:, i], softmaxes[:, i])
    roc_auc_dict[i] = auc(fpr_dict[i], tpr_dict[i])

# For example, print AUC for each class
for i in range(num_classes):
    print(f"Class {idx_to_class[i]} AUC: {roc_auc_dict[i]:.4f}")

# 11. Save results
# You can store numeric metrics in a CSV or JSON. For example:
if not os.path.exists(metrics_path):
    os.makedirs(metrics_path)

metrics_df = pd.DataFrame({
    "accuracy": [acc],
    "precision": [precision],
    "recall": [recall],
    "f1": [f1]
})
metrics_df.to_csv(os.path.join(metrics_path, "test_metrics.csv"), index=False)

# Confusion matrix
cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
cm_df.to_csv(os.path.join(metrics_path, "confusion_matrix.csv"))

print("Evaluation complete. Metrics saved!")
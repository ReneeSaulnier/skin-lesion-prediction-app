import yaml
import torch
import os
import pandas as pd
import numpy as np
from PIL import Image
from transformers import ResNetForImageClassification, AutoImageProcessor, Trainer, TrainingArguments, DefaultDataCollator
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


# Load the config file
config_path = "config/config.yaml"
with open(config_path, "r") as file:
    config = yaml.safe_load(file)

# Model config
model_name = config['model_training']['model']['name']
num_of_classes = config['model_training']['model']['num_of_classes']
output_path = config['model_training']['model']['output_path']
model_result_log = config['model_training']['logs']['path']

# Path config
processed_path = config['data_processing']['path']['processed_path']
X_train_path = os.path.join(processed_path, 'X_train.npy')
X_test_path = os.path.join(processed_path, 'X_test.npy')
y_train_path = os.path.join(processed_path, 'y_train.csv')
y_test_path = os.path.join(processed_path, 'y_test.csv')

# Load data
X_train = np.load(X_train_path, allow_pickle=True)
X_test = np.load(X_test_path, allow_pickle=True)
y_train = pd.read_csv(y_train_path)['dx'].values
y_test = pd.read_csv(y_test_path)['dx'].values
merged_df = pd.read_csv(os.path.join(processed_path, "merged_data.csv"))

# Get the unique labels (predictions)
class_names = np.unique(y_train)
class_to_idx = {cls: i for i, cls in enumerate(class_names)}
y_train_numeric = np.array([class_to_idx[label] for label in y_train], dtype=np.int64)
y_test_numeric = np.array([class_to_idx[label] for label in y_test], dtype=np.int64)

processor = AutoImageProcessor.from_pretrained(model_name)
model = ResNetForImageClassification.from_pretrained(model_name)

class SkinCancerDataset(Dataset):
    def __init__(self, image_paths, labels, processor):
        """
        A custom PyTorch dataset for dynamically loading and preprocessing images.

        :param image_paths: List of image file paths
        :param labels: List of numeric labels corresponding to the images
        :param processor: Hugging Face AutoImageProcessor for preprocessing
        """
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

        # Return preprocessed image and label
        return {
            'pixel_values': inputs['pixel_values'].squeeze(0),  # Remove batch dimension
            'labels': torch.tensor(label, dtype=torch.long)
        }
    
train_dataset = SkinCancerDataset(X_train, y_train_numeric, processor)
test_dataset = SkinCancerDataset(X_test, y_test_numeric, processor)

train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

training_args = TrainingArguments(
    output_dir=output_path,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="eval_accuracy",
    greater_is_better=True,
    learning_rate=5e-5,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    num_train_epochs=10,
    weight_decay=0.01,
    logging_dir=model_result_log,
    logging_steps=10,
    save_total_limit=2,
    push_to_hub=False
)

def compute_metrics(eval_pred):
    """
    eval_pred is a tuple: (logits, labels)
    where logits are the raw, unnormalized model outputs.
    """
    logits, labels = eval_pred
    # Get the predicted class by taking the argmax over logits
    predictions = np.argmax(logits, axis=-1)
    
    # Calculate metrics
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average='weighted'
    )
    acc = accuracy_score(labels, predictions)
    
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

data_collator = DefaultDataCollator()

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=processor,  # The processor works as a tokenizer here
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

if not os.path.exists(os.path.join(output_path, model_name)):
    os.makedirs(os.path.join(output_path, model_name))

trainer.train()

trainer.save_model(os.path.join(output_path, model_name))
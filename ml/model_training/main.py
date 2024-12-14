import yaml
import torch
import os
import pandas as pd
import numpy as np
from transformers import ResNetForImageClassification, AutoImageProcessor, Trainer, TrainingArguments, DefaultDataCollator

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
X_train = np.load(X_train_path)
X_test = np.load(X_test_path)
y_train = pd.read_csv(y_train_path)['dx'].values
y_test = pd.read_csv(y_test_path)['dx'].values

# Get the unique labels (predictions)
class_names = np.unique(y_train)
class_to_idx = {cls: i for i, cls in enumerate(class_names)}
y_train_numeric = np.array([class_to_idx[label] for label in y_train], dtype=np.int64)
y_test_numeric = np.array([class_to_idx[label] for label in y_test], dtype=np.int64)

processor = AutoImageProcessor.from_pretrained(model_name)
processor.do_rescale = False
# Preprocess NumPy arrays into tensors
def preprocess(X, y):
    pixel_values = processor(images=X, return_tensors="pt")['pixel_values']  # Normalize and convert to tensors
    labels = torch.tensor(y, dtype=torch.long)
    return pixel_values, labels

X_train, y_train = preprocess(X_train, y_train)
X_test, y_test = preprocess(X_test, y_test)

# Create datasets directly from preprocessed tensors
train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
test_dataset = torch.utils.data.TensorDataset(X_test, y_test)

# Model
model = ResNetForImageClassification.from_pretrained(model_name, num_labels=num_of_classes)

# Define training arguments
training_args = TrainingArguments(
    output_dir=model_result_log,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    num_train_epochs=5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    learning_rate=5e-5,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=50,
)

# Initialize trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=processor,  # Not strictly necessary for images, but some pipelines expect this
    data_collator=DefaultDataCollator(return_tensors="pt"),  # Handles batching
    do_rescale=False,
)

# Train
trainer.train()

# Evaluate
metrics = trainer.evaluate()
print(metrics)

# Save the model
trainer.save_model(output_path)
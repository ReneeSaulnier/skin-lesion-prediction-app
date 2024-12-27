import kagglehub
import yaml
import os
import shutil

# Load the config file
config_path = "config/config.yaml"
with open(config_path, "r") as file:
    config = yaml.safe_load(file)

dataset = config["data_collection"]["dataset"]
download_path = config["data_collection"]["download_path"]

# Download the dataset (to kagglehub's default cache path)
cached_path = kagglehub.dataset_download(dataset)

# Create a download path
if not os.path.exists(download_path):
    os.makedirs(download_path)

# Move the files from the cache path to the desired local path
for file_name in os.listdir(cached_path):
    shutil.move(os.path.join(cached_path, file_name), download_path)

# Print the path to the dataset in the desired folder
print(f"Dataset files moved to: {download_path}")
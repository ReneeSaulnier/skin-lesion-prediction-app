import kagglehub
import yaml

# Load the config file
config_path = "config/config.yaml"
with open(config_path, "r") as file:
    config = yaml.safe_load(file)

dataset = config["data_collection"]["dataset"]
download_path = config["data_collection"]["download_path"]

# Download the dataset and save to raw folder
kagglehub.dataset_download(dataset)
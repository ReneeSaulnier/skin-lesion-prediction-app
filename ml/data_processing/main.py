import pandas as pd
import yaml
import os
from PIL import Image

# Load the config file
config_path = "config/config.yaml"
# Load the config file
with open(config_path, "r") as file:
    config = yaml.safe_load(file)

# initialize the paths
metadata_path = config["data_processing"]["metadata_path"]
image_path_1 = config["data_processing"]["image_path_1"]
image_path_2 = config["data_processing"]["image_path_2"]
processed_path = config["data_processing"]["processed_path"]


def load_images(path):
    """
    Load the images from the path and return a dataframe
    :param path: str: path to the images
    :return: pd.DataFrame: dataframe of the images
    """
    images = []
    for file_name in os.listdir(path):
        image_path = os.path.join(path, file_name)
        with Image.open(image_path) as img:
            images.append({"filename": file_name, "image": img})

    return pd.DataFrame(images)

# Combine the images into one df
image_df_1 = load_images(image_path_1)
image_df_2 = load_images(image_path_2)
# Combine the image dfs
image_df = pd.concat([image_df_1, image_df_2], axis=0)

# Load the metadata file
metadata_df = pd.read_csv(metadata_path)

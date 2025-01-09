import yaml
import os
import pandas as pd
from sklearn.model_selection import train_test_split

# Load the config file
config_path = "config/config.yaml"
with open(config_path, "r") as file:
    config = yaml.safe_load(file)

# Image paths config
image_path_1 = config['data_processing']['path']['image_path_1']
image_path_2 = config['data_processing']['path']['image_path_2']
metadata_path = config['data_processing']['path']['metadata_path']
image_output_path = config['data_processing']['path']['image_path']
processed_data_path = config['data_processing']['path']['processed_path']
combine_image_flag = config['data_processing']['merge_folder']
# Dataset size config
validation_dataset_size = config['data_processing']['dataset_size']['validation']
test_dataset_size = config['data_processing']['dataset_size']['test']

# Combine the images into one folder <<< Run this only once, set the flag in config to False after running >>>
def combine_images(image_path_1, image_path_2):
    for file_name in os.listdir(image_path_1):
        if file_name.endswith(('.jpg', '.png', '.jpeg')):
            image_path = os.path.join(image_path_1, file_name)
            new_image_path = os.path.join(image_path_2, file_name)
            os.rename(image_path, new_image_path)

    # Rename the folder to images
    os.rename(image_path_2, image_output_path)
    # Delete the original folder
    os.rmdir(image_path_1)
    print('Images combined into one folder')
    return image_output_path

if combine_image_flag:
    final_image_path = combine_images(image_path_1, image_path_2)
else:
    final_image_path = image_output_path


def build_df(folder_path, metadata_path):
    # Get the metadata for the images to retrieve the labels
    metadata = pd.read_csv(metadata_path)
    image_list = []

    # Get the image path and add it to a dataframe
    for file_name in os.listdir(folder_path):
        if file_name.lower().endswith(('.jpg', '.png', '.jpeg')):
            image_list.append({"image_path": file_name})

    image_files_df = pd.DataFrame(image_list)

    # Add the extension to the image id << hardcoded for now >>
    metadata['image_id'] = metadata['image_id'] + '.jpg'

    # Merge the metadata labels with the image files
    merged_df = pd.merge(
        image_files_df,
        metadata,
        left_on='image_path',
        right_on='image_id',
        how='inner'
    )

    # Only want the image path and the label
    merged_df = merged_df[['image_id', 'dx']]
    return merged_df

image_df = build_df(final_image_path, metadata_path)

# Split the images for train/val/test
X = image_df['image_id']
y = image_df['dx']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_dataset_size, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=validation_dataset_size, random_state=42)

# Save as csv
train_df = pd.DataFrame({'image_path': X_train, 'dx': y_train})
val_df = pd.DataFrame({'image_path': X_val, 'dx': y_val})
test_df = pd.DataFrame({'image_path': X_test, 'dx': y_test})

train_df.to_csv(os.path.join(processed_data_path, 'train.csv'), index=False)
val_df.to_csv(os.path.join(processed_data_path, 'val.csv'), index=False)
test_df.to_csv(os.path.join(processed_data_path, 'test.csv'), index=False)


# Augment the images!
# Blurring, flipping, rotating, zooming, etc





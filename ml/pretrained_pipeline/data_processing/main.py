import pandas as pd
import numpy as np
import yaml
import os
from PIL import Image
from sklearn.model_selection import train_test_split

# Load the config file
config_path = 'config/config.yaml'
# Load the config file
with open(config_path, 'r') as file:
    config = yaml.safe_load(file)

# Paths from config
metadata_path = config['data_processing']['path']['metadata_path']
image_path_1 = config['data_processing']['path']['image_path_1']
image_path_2 = config['data_processing']['path']['image_path_2']
processed_path = config['data_processing']['path']['processed_path']

# Merging flags from config
combine_image_df = config['data_processing']['combine_image_df']
combing_image_and_metadata = config['data_processing']['combing_image_and_metadata']

# Dataset size config
validation_dataset_size = config['data_processing']['dataset_size']['validation']
test_dataset_size = config['data_processing']['dataset_size']['test']


def load_image_paths_into_single_df(path):
    """
    Load the image paths into a dataframe without loading the actual images.
    :param path: str: path to the image directory
    :return: pd.DataFrame: dataframe of image paths
    """

    if not os.path.exists(path):
        print(f"{path} does not exists, its likely the folders have already been combined.. Switch both falgs to False in config")
        return
    
    image_paths = []
    for file_name in os.listdir(path):
        if file_name.endswith(('.jpg', '.png', '.jpeg')):
            image_path = os.path.join(path, file_name)
            image_paths.append({'filename': file_name, 'image_path': image_path})

    return pd.DataFrame(image_paths)

if combine_image_df:
    image_path_1_df = load_image_paths_into_single_df(image_path_1)
    image_path_2_df = load_image_paths_into_single_df(image_path_2)
    image_df = pd.concat([image_path_1_df, image_path_2_df], axis=0)
    print('Combined image paths into a single dataframe')


# Load the metadata file
metadata_df = pd.read_csv(metadata_path)
metadata_df['image_id'] = metadata_df['image_id'] + '.jpg'


def merge_image_and_metadata_df(image_df, metadata_df):
    """
    Load the image_id and corresponding dx rows from the 
    metadata df into the image df. This is due to the large
    computation involved with combining.
    :param image_df: pd.Dataframe: dataframe of the images
    :param metadata_df: pd.Dataframe: dataframe of the metadata
    """
    merged_df = pd.merge(
        image_df,
        metadata_df[['image_id', 'dx']],
        how='left',
        left_on='filename',
        right_on='image_id'
    )

    return merged_df

if combing_image_and_metadata:
    merged_df = merge_image_and_metadata_df(image_df, metadata_df)

    # Make a processed data path
    if not os.path.exists(processed_path):
        os.makedirs(processed_path)

    # Export the final df
    merged_df.to_csv(os.path.join(processed_path, 'merged_data.csv'), index=False)
    print('Combining image and metadata df')
else:
    merged_df = pd.read_csv(os.path.join(processed_path, 'merged_data.csv'))
    print('Loaded pre-merged data')

# Split the data for train and test
X = merged_df['image_path']  # X = Features (images)
y = merged_df['dx']     # y = Target (diagnosis) 7 classes total



# => X_val = 10% of total dataset
# => X_train = 70% of total dataset
# => X_test  = 20% of total dataset
X_temp, X_test, y_temp, y_test = train_test_split(
    X,
    y,
    test_size=test_dataset_size,
    random_state=42
)

X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, 
    test_size=validation_dataset_size,
    random_state=42
)

# Save arrays as .npy for lower computation time
np.save(os.path.join(processed_path, 'X_train.npy'), X_train)
np.save(os.path.join(processed_path, 'X_val.npy'), X_val)
np.save(os.path.join(processed_path, 'X_test.npy'), X_test)

# Save labels as CSV
y_train.to_csv(os.path.join(processed_path, 'y_train.csv'), index=False)
y_val.to_csv(os.path.join(processed_path, 'y_val.csv'), index=False)
y_test.to_csv(os.path.join(processed_path, 'y_test.csv'), index=False)

print(f"Train and test sets saved in {processed_path}")
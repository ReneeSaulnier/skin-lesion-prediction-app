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

# Process image flags from config
image_resize = config['data_processing']['preprocess_images']['resize']
image_resize_dimensions = config['data_processing']['preprocess_images']['dimensions']
image_normalize = config['data_processing']['preprocess_images']['normalize']

def load_images_into_single_df(path, image_resize):
    """
    Load the images from the path and return a dataframe
    :param path: str: path to the images
    :return: pd.DataFrame: dataframe of the images
    """
    images = []
    for file_name in os.listdir(path):
        image_path = os.path.join(path, file_name)
        with Image.open(image_path) as img:
            if image_resize:
                img = img.resize(image_resize_dimensions)
            # Convert the image into a numpy array after resizing
            img_array = np.array(img, dtype=np.float32)
            images.append({'filename': file_name, 'image': img_array})

    return pd.DataFrame(images)

if combine_image_df:
    image_df_1 = load_images_into_single_df(image_path_1, image_resize)
    image_df_2 = load_images_into_single_df(image_path_2, image_resize)
    image_df = pd.concat([image_df_1, image_df_2], axis=0)
    print('Combinging image folders into one df')


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
    # Export the final df
    merged_df.to_csv(os.path.join(processed_path, 'merged_data.csv'), index=False)
    print('Combining image and metadata df')
else:
    merged_df = pd.read_csv(os.path.join(processed_path, 'merged_data.csv'))
    print('Loaded pre-merged data')

# Split the data for train and test
X = merged_df['image']  # X = Features (images)
y = merged_df['dx']     # y = Target (diagnosis)

X_train_df, X_test_df, y_train_df, y_test_df = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Normalize
def normalize_image(image):
    """
    Normalize the image to the range [0, 1] using min-max scaling.
    """
    train_min = np.min(image)
    train_max = np.max(image)
    return (image - train_min) / (train_max - train_min + 1e-7)

if image_normalize:
    X_train_df = X_train_df.apply(lambda img: np.array(img, dtype=np.float32))
    X_train_df = X_train_df.apply(lambda img: normalize_image(img))
    X_test_df = X_test_df.apply(lambda img: normalize_image(img))    
import numpy as np
import pandas as pd
import json
import os
import sys
import imageio
from multiprocessing import Pool
import re
import pickle
from PIL import Image


def get_json_files(path):
    """Gets the JSON metadata files

    Args:
        path (str): Path to the JSON files

    Returns:
        list: List of JSON metadata files
    """

    json_files = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if 'msrgb' not in file and '.json' in file:
                json_files.append(os.path.join(root, file))
    return json_files


def get_context(obj_width, obj_height, img_width, img_height):
    """Computes the size of the context around an image -- we got this code
    from the fMoW baseline code

    Args:
        obj_width (int): Width of the object of interest
        obj_height (int): Height of the object of interest
        img_width (int): Desired image width
        img_height (int): Desired image height

    Returns:
        int: width_buffer
        int: height_buffer
    """

    # Define our initial context width and height multiplier
    context_mult_width = 0.15
    context_mult_height = 0.15

    # Define the width and height ratio
    width_ratio = float(obj_width / img_width)
    height_ratio = float(obj_height / img_height)

    # Update the context multi width/height appropriately
    if width_ratio < 0.5 and width_ratio >= 0.4:
        context_mult_width = 0.2
    if width_ratio < 0.4 and width_ratio >= 0.3:
        context_mult_width = 0.3
    if width_ratio < 0.3 and width_ratio >= 0.2:
        context_mult_width = 0.5
    if width_ratio < 0.2 and width_ratio >= 0.1:
        context_mult_width = 1
    if width_ratio < 0.1:
        context_mult_width = 2

    if height_ratio < 0.5 and height_ratio >= 0.4:
        context_mult_height = 0.2
    if height_ratio < 0.4 and height_ratio >= 0.3:
        context_mult_height = 0.3
    if height_ratio < 0.3 and height_ratio >= 0.2:
        context_mult_height = 0.5
    if height_ratio < 0.2 and height_ratio >= 0.1:
        context_mult_height = 1
    if height_ratio < 0.1:
        context_mult_height = 2

    # Using the above values we can now compute our height and width
    # buffers
    width_buffer = int((obj_width * context_mult_width) / 2.0)
    height_buffer = int((obj_height * context_mult_height) / 2.0)
    return width_buffer, height_buffer


def read_crop_img(file, min_img_size, is_test):
    """Reads and crops an image

    Args:
        file (str): Path of image file
        min_img_size (int): Min image size we're willing to work with
        is_test (bool): Whether we're working with the test set

    Returns:
        str: image file
        img_file: full image file
        int: image height
        int: image width
    """

    # First we need to change the file ending from .json to .jpg so we
    # can actually read in the image
    img_file = re.sub('.json', '.jpg', file)

    # Now we can read in the image
    try:
        img = imageio.imread(img_file)
    except OSError:
        print('Could not read in: {}'.format(file))
        return

    # Now we need to get the metadata info so we can see if want to even
    # bother to crop the image
    try:
        with open(file) as json_file:
            metadata = json.load(json_file)
    except FileNotFoundError:
        print('Could not read in: {}'.format(file))

    # Let's get the image size and see if we even want to crop the image
    box = metadata['bounding_boxes'][0]['box']
    width = box[2]
    height = box[3]

    if width <= min_img_size or height <= min_img_size:
        return

    # Get our height and width buffer
    width_buffer, height_buffer = get_context(
        obj_width=width, obj_height=height, img_width=img.shape[0],
        img_height=img.shape[1])

    # Update the values where we're going to crop our image
    r1 = box[1] - height_buffer
    r2 = box[1] + box[3] + height_buffer
    c1 = box[0] - width_buffer
    c2 = box[0] + box[2] + width_buffer

    # Make possible corrections to our crop locations
    if r1 < 0:
        r1 = 0
    if r2 > img.shape[0]:
        r2 = img.shape[0]
    if c1 < 0:
        c1 = 0
    if c2 > img.shape[1]:
        c2 = img.shape[1]

    # If we cannot properly crop the image we will just exit
    if r1 >= r2 or c1 >= c2:
        return

    # Crop the image
    img_crop = img[r1:r2, c1:c2, :]

    # Update our image file
    img_paths = os.path.split(img_file)
    loc = np.core.defchararray.rfind(img_paths[0], '/')
    base_path = img_paths[0][:loc]

    if is_test:
        base_path = re.sub('val', 'val_crop', base_path)
    else:
        base_path = re.sub('train', 'train_crop', base_path)

    img_file = os.path.join(base_path, img_paths[1])

    # Finally save our image to disk
    if not os.path.isfile(img_file):
        imageio.imwrite(img_file, img_crop)

    # So that we can keep track of which images we're working with we're
    # going to return the JSON file and the our final image height and width
    return [file, img_file, (r2 - r1), (c2 - c1)]


def get_metadata(file_list, y_dict):
    """Gets the metadata from the JSON files

    Args:
        file_list (list): List of data we got from reading in the image
        y_dict (dict): Dictionary mapping class names to numbers

    Returns:
        object: Pandas DataFrame of metadata
    """

    # First we need to separate the components of our list
    json_file = file_list[0]
    img_file = file_list[1]
    height = file_list[2]
    width = file_list[3]

    # Read in the JSON file; if it fails, print the error and return
    # nothing
    try:
        with open(json_file) as file:
            metadata = json.load(file)
    except:
        return

    # Extract the features of interest
    gsd = metadata['gsd']
    target_azimuth = metadata['target_azimuth_dbl']
    sun_azimuth = metadata['sun_azimuth_dbl']
    sun_elevation = metadata['sun_elevation_dbl']
    off_nadir_angle = metadata['off_nadir_angle_dbl']
    country = metadata['country_code']

    # Handle special country codes (KO-, CA-):
    if country == 'CA-':
        country = 'RUS'

    if country == 'KO-':
        country = 'KOS'

    # Get the target for the train/val sets and img id
    img_id = metadata['bounding_boxes'][0]['ID']
    target = metadata['bounding_boxes'][0]['category']
    target = y_dict[target]
    return pd.DataFrame({'width': width, 'height': height, 'gsd': gsd,
                         'target_azimuth': target_azimuth,
                         'sun_azimuth': sun_azimuth,
                         'sun_elevation': sun_elevation,
                         'off_nadir_angle': off_nadir_angle,
                         'country': country, 'target': target,
                         'img_id': img_id, 'file': img_file},
                        index=[0])


def add_country_features(df, country_df):
    """Adds country features to our metadata

    Args:
        df (object): Pandas DataFrame we're working with
        country_df (object): Pandas DataFrame containing country info

    Returns:
        object: Merged pandas DataFrame
    """

    # Merge the GDP data separately because it's different from
    # the other data sets
    gdp = pd.read_csv('/pool001/zblanks/side_info/gdp.csv')
    df = df.merge(right=gdp, on='country', how='left')

    # Get our side info data sets
    loc = '/pool001/zblanks/side_info/'
    side_files = os.listdir(loc)

    # Remove the strings that we don't know and will confuse the functions
    side_files.remove('other_space_files')
    side_files.remove('gdp.csv')
    side_files.remove('base_country_data.csv')

    try:
        side_files.remove('metadata_train.csv')
        side_files.remove('metadata_val.csv')
    except FileNotFoundError:
        pass

    # Merge our data to our overall data
    for file in side_files:
        try:
            df = read_merge_data(merge_df=file, country_df=country_df,
                                 overall_df=df)
        except KeyError:
            print(file)
            sys.exit(1)

    # Fill the NaNs for our DataFrame
    country = df.country
    file = df.file
    df.drop(['country', 'file'], axis=1, inplace=True)
    df = df.apply(lambda x: x.fillna(x.mean()), axis=0)
    df['country'] = country
    df['file'] = file
    return df


def read_merge_data(merge_df, country_df, overall_df):
    """Helper function to read and merge side data

    Args:
        merge_df (object): Pandas DataFrame to merge with
        country_df (object): Country DataFrame
        overall_df (object): Overall metadata DataFrame we're building

    Returns:
        object: Overall metadata DataFrame
    """

    # Read in the merge df
    try:
        merge_df = pd.read_csv(os.path.join('/pool001/zblanks/side_info',
                                            merge_df))
    except pd.io.common.EmptyDataError:
        print(merge_df)
        sys.exit(1)

    # First merge on country_name to get our country codes (ex: USA)
    merge_df = merge_df.merge(right=country_df[['country_name',
                                                'country']],
                              on='country_name', how='left')

    # Remove the country name feature since we don't need it anymore
    merge_df.drop('country_name', axis=1, inplace=True)

    # Merge to our overall data frame
    overall_df = overall_df.merge(right=merge_df, on='country', how='left')
    return overall_df


if __name__ == '__main__':
    wd = sys.argv[1]
    min_img_size = int(sys.argv[2])
    Image.MAX_IMAGE_PIXELS = None

    # We will need our y dictionary for various tasks in this script
    with open(os.path.join(wd, 'prelim_files/y_dict.pickle'), 'rb') as pkl:
        y_dict = pickle.load(pkl)

    # We will also need our country DataFrame to help us build our country
    # features
    country = pd.read_csv(os.path.join(wd, 'side_info/base_country_data.csv'))

    # First we need to get the files which correspond to our training and
    # validation metadata files
    train_json = get_json_files(os.path.join(wd, 'train'))
    val_json = get_json_files(os.path.join(wd, 'val'))

    # Now we need to crop our images and get the files and other data
    # we need to build the metadata
    with Pool() as p:
        train_file_list = p.starmap(
            read_crop_img, zip(train_json, [min_img_size] * len(train_json),
                               [False] * len(train_json)))

        val_file_list = p.starmap(
            read_crop_img, zip(val_json, [min_img_size] * len(val_json),
                               [True] * len(train_json)))

    # Since it's possible we weren't able to process all of the metadat
    # we need to remove any None from the lists
    train_file_list = list(filter(None.__ne__, train_file_list))
    val_file_list = list(filter(None.__ne__, val_file_list))

    # Using the list of files and other data we can know get our metadata
    # vectors
    with Pool() as p:
        meta_train = p.starmap(
            get_metadata, zip(train_file_list,
                              [y_dict] * len(train_file_list)))

        meta_val = p.starmap(
            get_metadata, zip(val_file_list, [y_dict] * len(val_file_list)))

    # We need to get our training and validation metadata into one DataFrame
    meta_train = pd.concat(meta_train, ignore_index=True)
    meta_val = pd.concat(meta_val, ignore_index=True)

    # Now let's add the country based features to our data
    meta_train = add_country_features(df=meta_train, country_df=country)
    meta_val = add_country_features(df=meta_val, country_df=country)

    # As a double check we're going to make the image ID and target integers
    # if they are already not integers
    meta_train[['img_id', 'target']] = meta_train[['img_id',
                                                   'target']].astype(int)
    meta_val[['img_id', 'target']] = meta_val[['img_id', 'target']].astype(int)

    # Finally let's save our metadata to disk
    meta_train.to_csv(os.path.join(wd, 'side_info/metadata_train.csv'),
                      index=False)

    meta_val.to_csv(os.path.join(wd, 'side_info/metadata_val.csv'),
                    index=False)

import os
import sys
import re
import shutil


def fix_directory_name(folder):
    """Makes the folder name lowercase and removes spaces

    Args:
        folder (str): Folder name

    Returns:
        str: Fixed folder name
    """
    folder = folder.lower()
    folder = folder.replace(' ', '_')
    return folder


def change_directory_name(wd, is_train):
    '''Changes the directory name to be lowercase and remove spaces

    Args:
        wd (str): Working directory
        is_train (bool): Whether this is the training or testing set

    Returns:
        Nothing -- just changes the directory names
    '''

    if is_train:
        for folder in os.listdir(os.path.join(wd, 'train')):
            new_folder = fix_directory_name(folder=folder)

            # Change the name
            os.rename(os.path.join(wd, 'train', folder),
                      os.path.join(wd, 'train', new_folder))
    else:
        for folder in os.listdir(os.path.join(wd, 'test')):
            new_folder = fix_directory_name(folder=folder)
            os.rename(os.path.join(wd, 'test', folder),
                      os.path.join(wd, 'test', new_folder))


def fix_file_name(wd, directory, is_train):
    """Changes the file names in a particular directory

    Args:
        wd (str): Working directory
        directory (str): File directory
        is_train (bool): Whether we're working with the training or test set

    Returns:
        list: List of corrected file names
    """

    # Get the original file names
    if is_train:
        files = os.listdir(os.path.join(wd, 'train', directory))
    else:
        files = os.listdir(os.path.join(wd, 'test', directory))

    # Define a list to hold our new files
    new_files = [None] * len(files)

    # Change the file names to be "folder_name_img_#.jpg" and return the list
    for (i, file) in enumerate(files):
        new_files[i] = directory + '_' + str(i) + '.jpg'
    return new_files


def change_file_names(wd, is_train):
    """Changes the image file naming scheme to be more consistent and
    understandable

    Args:
        wd (str): Working directory
        is_train (bool): Whether we're working with the train or test set

    Returns:
        Nothing -- just changes the file names
    """

    # Loop through all of the directories and fix the file names
    if is_train:
        for directory in os.listdir(os.path.join(wd, 'train')):
            old_files = os.listdir(os.path.join(wd, 'train', directory))
            new_files = fix_file_name(wd=wd, directory=directory,
                                      is_train=True)
            # Change the file names
            for (old_file, new_file) in zip(old_files, new_files):
                os.rename(os.path.join(wd, 'train', directory, old_file),
                          os.path.join(wd, 'train', directory, new_file))
    else:
        for directory in os.listdir(os.path.join(wd, 'test')):
            old_files = os.listdir(os.path.join(wd, 'test', directory))
            new_files = fix_file_name(wd=wd, directory=directory,
                                      is_train=False)
            # Change the file names
            for (old_file, new_file) in zip(old_files, new_files):
                os.rename(os.path.join(wd, 'test', directory, old_file),
                          os.path.join(wd, 'test', directory, new_file))


def combine_redundant_labels(wd, is_train):
    """Combines redundant labels creating a new directory, moving the files
    in the old directories, and then deleting them

    Args:
        wd (str): Working directory
        is_train (bool): Whether we're working with the train or test set

    Returns:
        Nothing -- just does everything in the function description
    """

    # Get a list of all of the numbered directories
    r = re.compile('_[0-9]$')
    if is_train:
        folders = os.listdir(os.path.join(wd, 'train'))
    else:
        folders = os.listdir(os.path.join(wd, 'test'))
    redundant_labels = list(filter(r.search, folders))

    # Get our compiled directory names
    new_folders = [re.sub('_[0-9]$', '', label) for label in redundant_labels]
    new_folders = list(set(new_folders))

    # Create new directories with all of the new folders we found
    for new_folder in new_folders:
        if is_train:
            if not os.path.exists(os.path.join(wd, 'train', new_folder)):
                os.mkdir(os.path.join(wd, 'train', new_folder))
        else:
            if not os.path.exists(os.path.join(wd, 'test', new_folder)):
                os.mkdir(os.path.join(wd, 'test', new_folder))

    # Move the files from the redundant labels to their new folders
    for label in redundant_labels:
        # Get the files that correspond to this label
        if is_train:
            files = os.listdir(os.path.join(wd, 'train', label))
        else:
            files = os.listdir(os.path.join(wd, 'test', label))

        # Determine the destination folder for our files
        for new_folder in new_folders:
            if re.sub('_[0-9]$', '', label) == new_folder:
                dest_folder = new_folder
                break

        # Move the files to the dest folder
        if is_train:
            for file in files:
                shutil.move(os.path.join(wd, 'train', label, file),
                            os.path.join(wd, 'train', dest_folder, file))
        else:
            for file in files:
                shutil.move(os.path.join(wd, 'test', label, file),
                            os.path.join(wd, 'test', dest_folder, file))

        # Delete the redundant directories
        if is_train:
            shutil.rmtree(os.path.join(wd, 'train', label))
        else:
            shutil.rmtree(os.path.join(wd, 'test', label))


if __name__ == '__main__':
    # Get our working directory for the script
    wd = sys.argv[1]

    # Fix our directory names for the train and test set
    change_directory_name(wd=wd, is_train=True)
    change_directory_name(wd=wd, is_train=False)

    # Change the file names for the train and test set
    change_file_names(wd=wd, is_train=True)
    change_file_names(wd=wd, is_train=False)

    # Combine the redundant labels for the train and test set
    combine_redundant_labels(wd=wd, is_train=True)
    combine_redundant_labels(wd=wd, is_train=False)

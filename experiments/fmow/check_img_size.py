from glob import glob
from os import path
from joblib import Parallel, delayed
import numpy as np
from PIL import Image, ImageFile
import re
from skimage import io


def main(wd="/pool001/zblanks/"):
    """
    Reads in the images and determines the distribution
    """

    # Get all of the training and testing files
    train_path = path.join(wd, "train_crop")
    train_files = np.array(glob(train_path + "/**/*.jpg", recursive=True))

    # Remove the files for airports, shipyards, and ports beacuse they
    # are humongous and we already know we are going to have to shrink them
    regex = "airport_[0-9]{1,4}|shipyard_[0-9]{1,4}|port_[0-9]{1,4}"
    file_matches = [re.search(regex, file) for file in train_files]
    bad_files = [isinstance(match, type(None)) for match in file_matches]
    train_files = train_files[bad_files]

    with Parallel(n_jobs=-1, verbose=10) as p:
        # imgs = p(
        #     delayed(Image.open)(file) for file in train_files
        # )

        imgs = p(
            delayed(io.imread)(file) for file in train_files
        )

    # Get a list containing each of the sizes of the images
    img_sizes = [img.shape[0:2] for img in imgs]
    size_arr = np.array(img_sizes)

    # Compute size distribution values
    mean_size = size_arr.mean(axis=0)
    min_size = size_arr.min(axis=0)
    median_size = np.median(size_arr, axis=0)
    sd_size = np.std(size_arr, axis=0)
    return mean_size, min_size, median_size, sd_size


if __name__ == "__main__":
    Image.MAX_IMAGE_PIXELS = 10000000000000
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    sizes = main()
    print(sizes)

import os
from core.transform_data import TransformData


def main(height=224, width=224, nchannels=3, wd="/pool001/zblanks"):
    """
    Transforms the fMoW images into a X in R^{n x p} space to be used
    for our methods
    """

    # Define the path to the image data
    train_path = os.path.join(wd, "train_crop")

    # Define a TransformData object so that we can transform our data
    train_savepath = os.path.join(wd, "label_reduction_data/fmow", "train.h5")
    train_transform = TransformData(train_path, train_savepath, "densenet",
                                    img_shape=(height, width, nchannels))

    # Transform the training data and save it to disk
    train_transform.transform()


if __name__ == "__main__":
    main()

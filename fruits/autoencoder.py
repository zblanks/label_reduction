# from keras.layers import Conv2D, MaxPooling2D, UpSampling2D
# from keras.models import Model
# from keras.callbacks import CSVLogger, ModelCheckpoint
# import os
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('wd', help='Working directory for script',
                        type=str)
    args = parser.parse_args()

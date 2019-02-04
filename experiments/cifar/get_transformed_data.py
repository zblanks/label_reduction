from core.transform_data import TransformData
import argparse
from keras.datasets import cifar100
import numpy as np
from os import path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("wd", type=str, nargs="?",
                        default="/pool001/zblanks/label_reduction_data/cifar")
    parser.add_argument('--model', type=str, nargs='?', default='densenet')
    parser.add_argument("--ngpus", type=int, nargs="?", default=2)
    args = vars(parser.parse_args())

    # Get the CIFAR data with fine labels from Keras
    (X_train, y_train), (X_test, y_test) = cifar100.load_data(label_mode="fine")

    # Combine the training and test samples to have a consistent format
    # for running our experiments
    X = np.concatenate([X_train, X_test], axis=0)
    y = np.concatenate([y_train, y_test])

    # We'll keep the current image size because they're all the same size
    img_size = X.shape[1:]

    # Finally we will generate the transformed data
    savepath = path.join(args['wd'], 'data.h5')
    transformer = TransformData(data_path="", save_path=savepath,
                                model_name=args['model'], img_shape=img_size,
                                ngpu=args['ngpus'])

    transformer.transform(X=X, y=y)


if __name__ == '__main__':
    main()

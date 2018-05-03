import numpy as np
import sys
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import os
import glob
from multiprocessing import Pool
import pandas as pd


def map_labels(label_map, orig_labels):
    """Re-maps our labels according to the provided map

    Args:
        label_map (ndarray shape=[n_class, k]): Label map found through IP
        orig_labels (ndarray, shape=[n_sample, ]): Original labels

    Returns:
        (ndarray, shape=[n_sample, ]): Re-mapped labels
    """

    # Compute the dictionary to map our labels
    n_class = len(np.unique(orig_labels))
    label_dict = dict(zip(range(n_class), label_map.argmax(axis=1)))

    # Use the dictionary to re-map our labels
    return np.vectorize(label_dict.get)(orig_labels)


def train_eval_model(train_data, label_map, is_val_search):
    """Trains and evaluates the model to determine the best value of k

    Args:
        train_data (DataFrame): Training data
        label_map (ndarray shape=[n_class, k]): Label map found through IP
        is_val_search (bool): Whether we're doing our validation seach
                              or training our final model

    Returns:
        float: Validation accuracy
    """

    # Get our data and labels
    y = train_data.target.as_matrix()
    X = train_data.drop(['target'], axis=1).as_matrix()

    # Re-map our labels
    y = map_labels(label_map=label_map, orig_labels=y)

    # Define our SVM object
    svm = LinearSVC(class_weight='balanced', random_state=17)

    # Split into training and validation
    if is_val_search:
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.25,
                                                          random_state=17)
        # Train our model
        svm.fit(X_train, y_train)

        # Asses the model's validation performance
        return svm.score(X_val, y_val)
    else:
        svm.fit(X, y)
        return svm


def classification_report_csv(report, path):
    report_data = []
    lines = report.split('\n')
    for line in lines[2:-3]:
        row = {}
        row_data = line.split('      ')
        row['class'] = row_data[0]
        row['precision'] = float(row_data[1])
        row['recall'] = float(row_data[2])
        row['f1_score'] = float(row_data[3])
        row['support'] = float(row_data[4])
        report_data.append(row)
    df = pd.DataFrame.from_dict(report_data)
    df.to_csv(path, index = False)


if __name__ == '__main__':
    # Get the script arguments
    wd = sys.argv[1]

    # Get a list of all of the maps
    label_map_paths = glob.glob(os.path.join(wd, 'fruit_maps/final_map*'),
                                recursive=True)

    # Read in the maps
    label_maps = [None] * len(label_map_paths)
    for (i, path) in enumerate(label_map_paths):
        label_maps[i] = np.loadtxt(path, delimiter=',')

    # Get the training and testing data
    train = pd.read_csv(os.path.join(wd, 'fruits_matrix/train.csv'))
    test = pd.read_csv(os.path.join(wd, 'fruits_matrix/test.csv'))

    # Train all of our models and get the validation score for each
    with Pool() as p:
        val_scores = p.starmap(train_eval_model,
                               zip([train] * len(label_maps), label_maps,
                                   [True] * len(label_maps)))

    # Determine the best value for k
    best_k = np.argmax(val_scores)

    # Train the model with the best value for k and then evaluate it's
    # performance out of sample
    clf = train_eval_model(train_data=train, label_map=label_maps[best_k],
                           is_val_search=False)
    y_test = test.target.as_matrix()
    y_test = map_labels(label_map=label_maps[best_k], orig_labels=y_test)
    X_test = test.drop(['target'], axis=1)

    # Get our test set predictions
    y_pred = clf.predict(X_test)
    class_report = classification_report(y_true=y_test, y_pred=y_pred)
    classification_report_csv(class_report,
                              path=os.path.join(wd, 'fruits_res/remap_model',
                                                'class_report.csv'))

    # Get our confusion matrix
    c_mat = confusion_matrix(y_true=y_test, y_pred=y_pred)
    pd.DataFrame(c_mat).to_csv(os.path.join(wd, 'fruits_res/remap_model',
                                            'c_mat.csv'), index=False)

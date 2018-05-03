import pandas as pd
from sklearn.decomposition import PCA
from sklearn.svm import LinearSVC
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import sys
import os
import numpy as np


def classification_report_csv(report, wd):
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
    df.to_csv(os.path.join(wd, 'fruits_res/initial_model',
                           'classification_report.csv'),
              index = False)


if __name__ == '__main__':
    # Script arguments
    wd = sys.argv[1]

    # Get the data
    train = pd.read_csv(os.path.join(wd, 'fruits_matrix', 'train.csv'))
    test = pd.read_csv(os.path.join(wd, 'fruits_matrix', 'test.csv'))
    y_train = train.label.as_matrix()
    y_test = test.label.as_matrix()
    X_train = train.drop(['label'], axis=1).as_matrix()
    X_test = test.drop(['label'], axis=1).as_matrix()

    # Use PCA to reduce the dimensionality
    pca = PCA(n_components=1000, random_state=17)
    X_train = pca.fit_transform(X_train)
    X_test = pca.transform(X_test)

    # Save the data to disk
    X_train_pca = pd.DataFrame(X_train)
    X_train_pca.loc[:, 'target'] = y_train
    X_train_pca.to_csv(os.path.join(wd, 'fruits_matrix', 'train_pca.csv'),
                       index=False)

    X_test_pca = pd.DataFrame(X_test)
    X_test_pca.loc[:, 'target'] = y_test
    X_test_pca.to_csv(os.path.join(wd, 'fruits_matrix', 'test_pca.csv'),
                      index=False)

    # Define the grid of parameters to search over
    np.random.seed(17)
    c_vals = np.power(np.repeat(10, repeats=10000),
                      np.random.uniform(-3, 3, 10000))
    param_distn = {'C': c_vals}
    svm = LinearSVC(class_weight='balanced', random_state=17)
    clf = RandomizedSearchCV(svm, param_distributions=param_distn, n_iter=10,
                             n_jobs=-1, cv=3, verbose=1, random_state=17,
                             return_train_score=False)

    # Train our model
    clf.fit(X_train, y_train)
    cv_res = pd.DataFrame(clf.cv_results_)
    cv_res.to_csv(os.path.join(wd, 'fruits_res/initial_model', 'cv_res.csv'),
                  index=False)

    # Save the classification report to disk
    y_pred = clf.predict(X_test)
    class_report = classification_report(y_true=y_test, y_pred=y_pred)

    # Save the confusion matrix to disk
    c_mat = confusion_matrix(y_true=y_test, y_pred=y_pred)
    pd.DataFrame(c_mat).to_csv(os.path.join(wd, 'fruits_res/initial_model',
                                            'c_mat.csv'), index=False)

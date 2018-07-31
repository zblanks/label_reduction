from sklearn.cluster import SpectralClustering
from sklearn.svm import LinearSVC
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import numpy as np


def build_tree(X: np.ndarray, y: np.ndarray, nnodes: int, ninit: int):
    # Get the train-test split
    rng = np.random.RandomState(17)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=rng
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.20, random_state=rng
    )

    # Train the OVR classifier for step one of the tree inference
    svm = LinearSVC(loss="hinge", class_weight="balanced", random_state=rng)
    svm.fit(X_train, y_train)

    # Compute the confusion matrix for the validation set
    y_pred = svm.predict(X_val)
    cmat = confusion_matrix(y_val, y_pred)

    # Compute the affinity matrix
    A = 1/2 * (cmat + cmat.T)

    # Perform spectral clustering on the affinity matrix
    spectral = SpectralClustering(n_clusters=nnodes, random_state=rng,
                                  affinity="precomputed", n_init=ninit,
                                  n_jobs=-1)
    assignments = spectral.fit_predict(A)

    # Generate the label map from the cluster assignments
    nclass = len(np.unique(y))
    z = np.zeros(shape=(nclass, nnodes), dtype=int)
    for (label, assign) in zip(range(nclass), assignments):
        z[label, assign] = 1
    return z

from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.cluster import SpectralClustering
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import Pipeline
import numpy as np
from time import time
from joblib import Parallel, delayed


def train_node(X: np.ndarray, y: np.ndarray, rng: np.random.RandomState,
               estimator: str):
    """
    Trains a given node for any type of classifier
    """

    # Get the number of unique labels in the target vector to do
    # supervised dimensionality reduction
    nlabels = len(np.unique(y))

    # Define a scaler object to ensure the data has zero mean and a variance
    # of one
    scaler = StandardScaler()

    # Define the estimator provided to the function
    if estimator == "log":
        model = SGDClassifier(loss="log", random_state=rng,
                              class_weight="balanced", warm_start=True,
                              max_iter=1000, tol=1e-3)

    elif estimator == 'knn':
        model = KNeighborsClassifier(n_jobs=-1)

    else:
        model = RandomForestClassifier(random_state=rng, n_estimators=100,
                                       class_weight="balanced", n_jobs=-1)

    # Combine all of the elements to create a sklean pipeline to simplify
    # the process of training a particular model
    if nlabels < X.shape[1]:
        lda = LinearDiscriminantAnalysis(n_components=(nlabels - 1))
        steps = [('scaler', scaler), ('lda', lda), ('model', model)]
    else:
        steps = [('scaler', scaler), ('model', model)]

    # Fit the model
    pipe = Pipeline(steps)
    pipe.fit(X, y)
    return pipe


def remap_labels(y: np.ndarray, label_groups: np.ndarray) -> np.ndarray:
    """
    Maps a label vector according to the proscribed partition
    """
    y_new = np.empty_like(y)
    ngroup = len(np.unique(label_groups))
    for i in range(ngroup):
        idx = np.isin(y, np.where(label_groups == i)[0])
        y_new[idx] = i
    return y_new


def flat_model(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray,
               rng: np.random.RandomState, estimator: str) -> dict:
    """
    Trains the FC and gets the out of sample predictions
    """

    # Train the model
    start_time = time()
    clf = train_node(X_train, y_train, rng, estimator)
    train_time = time() - start_time

    # Get the test set predictions
    proba_pred = clf.predict_proba(X_test)

    return {"train_time": train_time, "proba_pred": proba_pred}


def clean_proba_pred(proba_pred: np.ndarray) -> tuple:
    """
    Cleans a probability prediction matrix for issues with Inf or NaN so that
    we can compute evaluation metrics
    """

    # First identify the location of bad indices in the Y_hat matrix
    bad_idx = np.unique(np.argwhere(~np.isfinite(proba_pred))[:, 0])

    # Next using the bad indices we just inferred we need to get the
    # remaining workable indices
    good_idx = np.setdiff1d(np.arange(proba_pred.shape[0]), bad_idx)
    return proba_pred[good_idx, :], good_idx


def hc_pred(models: list, X_test: np.ndarray, label_groups: np.ndarray) -> dict:
    """
    Get the out of sample HC predictions
    """

    # Infer the constants for the function
    nclasses = len(label_groups)
    ngroups = len(np.unique(label_groups))
    leaf_idx = np.where(np.bincount(label_groups) > 1)[0]
    nmodels = len(models)

    # Grab the sklearn pipelines
    clfs = [models[i] for i in range(nmodels)]

    # Get the probability predictions from the HC root
    root_proba_pred = clfs[-1].predict_proba(X_test)
    lone_leaves = np.setdiff1d(np.arange(ngroups), leaf_idx)
    label_groups = [np.where(label_groups == i)[0] for i in range(ngroups)]
    root_map = dict(zip(range(ngroups), label_groups))

    # Define a place holder for each of the probability predictions from
    # the HC nodes
    n = X_test.shape[0]
    node_proba_preds = [np.zeros(shape=(n, len(root_map[idx])))
                        for idx in leaf_idx]

    # For all of the non-lone leaves, we will compute their posterior and
    # update the probabilities in the node matrices
    for (i, val) in enumerate(leaf_idx):
        tmp_pred = clfs[i].predict_proba(X_test)

        # Adjust the probability values for the appropriate labels
        # in the correct node probability matrix
        node_proba_preds[i] = tmp_pred

    # Combining the root probability prediction and the node predictions
    # we can generate the final posterior distribution for all samples
    proba_pred = np.zeros(shape=(n, nclasses))

    # First we will go through the degenerate nodes and get the final
    # probability predictions
    for label in lone_leaves:
        proba_pred[:, root_map[label]] = root_proba_pred[:, [label]]

    # Now we will go through each of the non-degenerate nodes and get their
    # final posterior probabilities
    for (i, idx) in enumerate(leaf_idx):
        root_pred = root_proba_pred[:, [idx]]
        node_pred = node_proba_preds[i]
        proba_pred[:, root_map[idx]] = root_pred * node_pred

    # Clean up the final prediction to remove any possible Inf or NaN
    proba_pred, good_idx = clean_proba_pred(proba_pred)

    return {"proba_pred": proba_pred, "root_proba_pred": root_proba_pred,
            "node_proba_preds": node_proba_preds, "good_idx": good_idx}


def hierarchical_model(X_train: np.ndarray, y_train: np.ndarray,
                       X_val: np.ndarray, y_val: np.ndarray,
                       label_groups: np.ndarray,
                       rng: np.random.RandomState,
                       estimator: str) -> dict:
    """
    Trains the HC and gets the out of sample predictions
    """

    # Re-map the initial set of labels
    y_root = remap_labels(y_train, label_groups)

    # Determine which (if any) of the leaves are by themselves so that
    # we don't unnecessarily train models
    leaf_idx = np.where(np.bincount(label_groups) > 1)[0]

    # Compute the indices for each of the label groups
    idx_list = [np.isin(y_train, np.where(label_groups == i)[0])
                for i in leaf_idx]

    X_list = [X_train[idx] for idx in idx_list]
    y_list = [y_train[idx] for idx in idx_list]
    X_list.append(X_train)
    y_list.append(y_root)

    # Train each of the nodes
    start_time = time()
    # with Parallel(n_jobs=-1) as p:
    #     models = p(delayed(train_node)(X, y, rng, estimator)
    #                for (X, y) in zip(X_list, y_list))
    nmodels = len(idx_list) + 1
    models = []
    for i in range(nmodels):
        models.append(train_node(X_list[i], y_list[i], rng, estimator))

    train_time = time() - start_time

    # Get the test set predictions
    val_res = hc_pred(models, X_val, label_groups)

    # Now we can make predictions leaf-level predictions from the good indices
    y_pred = val_res["proba_pred"].argmax(axis=1)
    y_val = y_val[val_res["good_idx"]]
    acc = accuracy_score(y_val, y_pred)

    # Compute the AUC
    y_mat = OneHotEncoder(sparse=False).fit_transform(y_val.reshape(-1, 1))
    auc = roc_auc_score(y_mat, val_res["proba_pred"])

    return {"train_time": train_time, "models": models, "acc": acc, "auc": auc}


def spectral_model(X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray,
                   y_val: np.ndarray, k: int, rng: np.random.RandomState,
                   estimator: str, **kwargs):
    """
    Trains the hierarchical classifier using the spectral clustering
    approach to finding Z
    """
    # Train the initial model (if we not already called the spectral function)
    if 'affinity_mat' not in kwargs.keys():
        start_time = time()
        init_clf = train_node(X_train, y_train, rng, estimator)
        init_train_time = time() - start_time

        # Get the confusion matrix to use for spectral clustering
        cmat = confusion_matrix(y_val, init_clf.predict(X_val))
        A = .5 * (cmat * cmat.T)
        affinity_mat = np.copy(A)
    else:
        # Use the existing confusion matrix to perform spectral clustering
        init_train_time = 0.
        A = kwargs['affinity_mat']
        affinity_mat = np.copy(A)

    # Perform spectral clustering on the affinity matrix
    start_time = time()
    sc = SpectralClustering(n_clusters=k, random_state=rng,
                            affinity="precomputed")
    label_groups = sc.fit_predict(affinity_mat)
    cluster_time = time() - start_time

    # Feed the label groups into the standard way of training the HC
    res = hierarchical_model(X_train, y_train, X_val, y_val, label_groups,
                             rng, estimator)

    # Update the training time of the HC to include the time we spent training
    # the initial classifier
    res['train_time'] += init_train_time
    res['cluster_time'] = cluster_time

    if 'affinity_mat' not in kwargs.keys():
        return res, A, label_groups
    else:
        return res, label_groups

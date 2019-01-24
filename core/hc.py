from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.cluster import SpectralClustering
import numpy as np
from time import time
from multiprocessing import Pool
from itertools import repeat
from copy import copy


def train_node(X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray,
               y_val: np.ndarray, niter: int, rng: np.random.RandomState,
               estimator: str, method: str) -> dict:
    """
    Trains a given node for any type of classifier
    """

    # Extract features for the model using PCA
    if method == "f" or method == "hci":
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        pca = PCA(n_components=50, random_state=rng)
        X_train = pca.fit_transform(X_train)
        X_val = pca.transform(X_val)
    else: # the paper did not say to take these actions therefore we will not
        pca = ""
        scaler = ""

    if estimator == "log":
        clf = SGDClassifier(loss="log", random_state=rng,
                            class_weight="balanced", warm_start=True,
                            max_iter=1000, tol=1e-3, n_jobs=-1)

        # Generate regularization values
        alpha_vals = 10 ** rng.uniform(-5, 5, niter)

        clfs = [copy(clf) for _ in range(niter)]
        for i in range(niter):
            clfs[i] = clfs[i].set_params(**{"alpha": alpha_vals[i]})
            clfs[i].fit(X_train, y_train)

    else:
        clf = RandomForestClassifier(n_jobs=-1, random_state=rng,
                                     class_weight="balanced")

        # Generate forest sizes
        ntrees = np.ceil(10 ** rng.uniform(2, 3, niter)).astype(int)

        clfs = [copy(clf) for _ in range(niter)]
        for i in range(niter):
            clfs[i] = clfs[i].set_params(**{"n_estimators": ntrees[i]})
            clfs[i].fit(X_train, y_train)

    val_losses = np.array([clfs[i].score(X_val, y_val) for i in range(niter)])
    best_model = val_losses.argmax()
    return {"clf": clfs[best_model], "pca": pca, "scaler": scaler}


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


def flat_model(X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray,
               y_val: np.ndarray, X_test: np.ndarray,
               rng: np.random.RandomState, estimator: str, niter=10) -> dict:
    """
    Trains the FC and gets the out of sample predictions
    """

    # Train the model
    start_time = time()
    models = train_node(X_train, y_train, X_val, y_val, niter, rng,
                        estimator=estimator, method="f")
    train_time = time() - start_time

    # Test the model
    pca = models["pca"]
    scaler = models["scaler"]
    clf = models["clf"]
    test_X = pca.transform(scaler.transform(X_test))
    proba_pred = clf.predict_proba(test_X)

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

    # Grab the object of interest
    clfs = [models[i]["clf"] for i in range(nmodels)]
    pcas = [models[i]["pca"] for i in range(nmodels)]
    scalers = [models[i]["scaler"] for i in range(nmodels)]

    # Put a placeholder for the predictions
    n = X_test.shape[0]

    # Get the probability predictions from the HC root
    test_X = pcas[-1].transform(scalers[-1].transform(X_test))
    root_proba_pred = clfs[-1].predict_proba(test_X)
    lone_leaves = np.setdiff1d(np.arange(ngroups), leaf_idx)
    label_groups = [np.where(label_groups == i)[0] for i in range(ngroups)]
    root_map = dict(zip(range(ngroups), label_groups))

    # Define a place holder for each of the probability predictions from
    # the HC nodes
    node_proba_preds = [np.zeros(shape=(n, len(root_map[idx])))
                        for idx in leaf_idx]

    # For all of the non-lone leaves, we will compute their posterior and
    # update the probabilities in the node matrices
    for (i, val) in enumerate(leaf_idx):
        test_X = pcas[i].transform(scalers[i].transform(X_test))
        tmp_pred = clfs[i].predict_proba(test_X)

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
                       estimator: str, niter=10) -> dict:
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

    X_val_rep = repeat(X_val)
    y_val_rep = repeat(y_val)
    niter_rep = repeat(niter)
    rng_rep = repeat(rng)
    estimator_rep = repeat(estimator)
    method_rep = repeat("hci")
    with Pool() as p:
        models = p.starmap(train_node, zip(X_list, y_list, X_val_rep, y_val_rep,
                                           niter_rep, rng_rep, estimator_rep,
                                           method_rep))
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


def spectral_model(X_train: np.ndarray, y_train: np.ndarray,
                   X_test: np.ndarray, L: int, rng: np.random.RandomState,
                   estimator: str, niter=10) -> dict:
    """
    Trains the hierarchical classifier using the spectral clustering
    approach to finding Z
    """
    # Train the initial model
    start_time = time()
    init_clf = train_node(X_train, y_train, niter, rng, estimator=estimator,
                          method="f")
    init_train_time = time() - start_time

    # Compute the validation confusion matrix
    train_X, val_X, train_y, val_y = train_test_split(
        X_train, y_train, test_size=0.25, random_state=rng
    )
    val_X = init_clf["pca"].transform(val_X)
    cmat = confusion_matrix(val_y, init_clf["clf"].predict(val_X))

    # Prepare the confusion matrix to be used in spectral clustering
    a = 1/2 * (cmat + cmat.T)
    sc = SpectralClustering(n_clusters=L, random_state=rng,
                            affinity="precomputed")
    label_pred = sc.fit_predict(a)

    # Gather the label groups
    label_groups = [np.where(label_pred == i)[0] for i in range(L)]

    # Feed the label groups into the standard way of training the HC
    res = hierarchical_model(X_train, y_train, X_test, label_groups, rng,
                             estimator=estimator, niter=niter)

    # Update the training time of the HC to include the time we spent training
    # the initial classifier
    res["train_time"] += init_train_time
    return res


def db2_model(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray,
              tree_dict: dict, rng: np.random.RandomState, estimator: str,
              niter=10) -> dict:
    """
    Implements the DB2 benchmark
    """

    # Scale and reduce the dimensionality of the data only for the starting
    # data since they did not feed different features to each of the parent
    # nodes in the DB2 tree
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    pca = PCA(n_components=50, random_state=rng)
    X_train = pca.fit_transform(X_train)

    # Get the indices for each of the parents in the DB2 tree
    nkeys = len(tree_dict.keys())
    idx = [np.empty(shape=(y_train.shape[0],), dtype=bool)] * nkeys
    for (i, key) in enumerate(tree_dict.keys()):
        labels = tree_dict[key][0] + tree_dict[key][1]
        idx[i] = np.isin(y_train, labels)

    # Re-map the labels and X matrices for each of the parents in the tree
    X_list = [np.empty_like(X_train)] * nkeys
    y_list = [np.empty_like(y_train)] * nkeys
    for (i, key) in enumerate(tree_dict.keys()):
        # Get the new X matrix
        X_new = X_train[idx[i], :]
        X_list[i] = X_new

        # Re-map the labels
        y_new = y_train[idx[i]]
        y_new = remap_labels(y_new, tree_dict[key])
        y_list[i] = y_new

    # Train each of the parent nodes in parallel
    start_time = time()
    niter_rep = repeat(niter)
    rng_rep = repeat(rng)
    estimator_rep = repeat(estimator)
    method_rep = repeat("db2")
    with Pool() as p:
        models = p.starmap(train_node, zip(X_list, y_list, niter_rep, rng_rep,
                                           estimator_rep, method_rep))
    train_time = time() - start_time

    # Update the models object to have the key --> clf
    clfs = [models[i]["clf"] for i in range(len(models))]
    model_dict = dict(zip(tree_dict.keys(), clfs))

    # Get the out-of-sample predictions
    y_pred = np.empty(shape=(X_test.shape[0]), dtype=int)
    idx = np.arange(X_test.shape[0])
    label_group = str(np.unique(y_train).tolist())
    y_pred = db2_pred(X_test, label_group, y_pred, tree_dict, model_dict, idx)

    return {"train_time": train_time, "y_pred": y_pred}


def db2_pred(X: np.ndarray, label_group: str, y_pred: np.ndarray,
             tree_dict: dict, model_dict: dict, idx: np.ndarray) -> np.ndarray:
    """
    Gets the predictions for the DB2 benchmark
    """

    # Get the predictions for the given label group and data
    pred = model_dict[str(label_group)].predict(X[idx, :])

    # Since we are restricting ourselves to binary the resulting output
    # will be {0, 1}
    unique_pred = np.unique(pred)
    for val in unique_pred:
        pred_idx = idx[np.where(pred == val)]

        # If we are at the case where the child of the given label_group
        # parent is a singleton, we know we can make the final prediction
        if len(tree_dict[label_group][val]) == 1:
            # Update the final prediction values
            y_pred[pred_idx] = tree_dict[label_group][val][0]
        else:
            # We are in a case where we are not dealing with a singleton
            # and thus need to recurse down the DB2 tree
            new_label_group = str(tree_dict[label_group][val])
            y_pred = db2_pred(X[idx, :], new_label_group, y_pred,
                              tree_dict, model_dict)

    return y_pred

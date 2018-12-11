from core.hc import flat_model, hierarchical_model, hc_pred
from core.kmeans import kmeans_mean
from os import path
import h5py
import argparse
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils import resample
import pandas as pd
import hashlib
import pickle
from time import time


def gen_id(args: dict):
    """
    Generates the id(s) for our data using a SHA algorithm
    """

    # Generate the string(s) needed to create the IDs
    if args["method"] == "f":
        k_vals = ["62"]
    else:
        k_vals = list(map(str, args["k_vals"]))

    # Convert the numeric arguments into strings
    n = len(k_vals)
    e = [args["estimator"]] * n
    m = [args["method"]] * n
    ni = list(map(str, [args["niter"]] * n))
    um = list(map(str, [args["use_meta"]] * n))
    rn = list(map(str, [args["run_num"]] * n))

    # Generate the strings used for the SHA algorithm
    hashes = [""] * n
    for i in range(n):
        tmp_str = "_".join((rn[i], m[i], e[i], ni[i], um[i], k_vals[i]))
        tmp_str = tmp_str.encode("UTF-8")
        hashes[i] = hashlib.sha1(tmp_str).hexdigest()
    return hashes


def create_experiment_df(args: dict, ids: list):
    """
    Creates a DataFrame detailing the settings for a given experiment
    """

    n = len(ids)
    if args["method"] == "f":
        k_vals = [62]
    else:
        k_vals = args["k_vals"]

    df = pd.DataFrame(
        data={"id": ids, "run_num": np.repeat([args["run_num"]], n),
              "method": np.repeat([args["method"]], n),
              "estimator": np.repeat([args["estimator"]], n),
              "use_meta": np.repeat([args["use_meta"]], n),
              "niter": np.repeat([args["niter"]], n),
              "k": k_vals}
    )
    return df


def get_best_k(X_train: np.ndarray, y_train: np.ndarray, args: dict):
    """
    Down-samples the data so that we can determine the best value for k
    using the validation loss
    """

    # Split the data into training and validation so we can determine the
    # best k
    splitter = StratifiedShuffleSplit(
        n_splits=1, test_size=0.20, random_state=17
    )

    idx = splitter.split(X_train, y_train)
    idx = [val for val in idx]
    train_idx = idx[0][0]
    val_idx = idx[0][1]
    train_X = X_train[train_idx]
    train_y = y_train[train_idx]
    val_X = X_train[val_idx]
    val_y = y_train[val_idx]

    # Determine the best k
    n_vals = len(args["k_vals"])
    k_res = [dict()] * n_vals
    label_groups = [[]] * n_vals
    for i in range(n_vals):
        print("Searching over k = {}".format(args["k_vals"][i]))
        # Infer the groups for the given value of k and train the HC
        start_time = time()
        label_groups[i] = kmeans_mean(train_X, train_y, args["k_vals"][i])
        cluster_time = time() - start_time

        k_res[i] = hierarchical_model(
            train_X, train_y, val_X, label_groups[i],
            np.random.RandomState(17), args["estimator"]
        )

        # Add the time it took to find the label groups
        k_res[i]["train_time"] += cluster_time

    # We need to check for NaN or Inf samples in the probability predictions
    # and then remove them so that we don't error out
    bad_idx = [
        np.unique(np.argwhere(~np.isfinite(k_res[i]["proba_pred"]))[:, 0])
        for i in range(n_vals)
    ]
    print(bad_idx)

    good_idx = [
        np.setdiff1d(np.arange(k_res[i]["proba_pred"].shape[0]), bad_idx[i])
        for i in range(n_vals)
    ]

    # Get the arg-max predictions for each of the values of k
    y_preds = [k_res[i]["proba_pred"][good_idx[i], :].argmax(axis=1)
               for i in range(n_vals)]

    # Compute the validation accuracy values
    acc_vals = np.array([accuracy_score(val_y[good_idx[i]], y_preds[i])
                         for i in range(n_vals)])

    # Compute the validation AUC values
    enc = OneHotEncoder(sparse=False)
    true_y = enc.fit_transform(val_y.reshape(-1, 1))

    auc_vals = np.array([roc_auc_score(true_y[good_idx[i], :],
                                       k_res[i]["proba_pred"][good_idx[i], :])
                         for i in range(n_vals)])

    # Get the total training time
    train_times = np.array([k_res[i]["train_time"] for i in range(n_vals)])
    total_train_time = train_times.sum()

    # Determine the best model from the accuracy value
    best_model = acc_vals.argmax()

    # Get the hashes to act as the PKs for the results data
    ids = gen_id(args)

    # Build a matrix which stores the results from the validation search
    # so that we can evaluate them later
    metrics = ["top1", "auc", "train_time"]
    nmetrics = len(metrics)
    search_df = pd.DataFrame(
        data={"id": np.tile(ids, nmetrics),
              "metric": np.repeat(metrics, len(ids)),
              "value": np.concatenate([acc_vals, auc_vals, train_times])}
    )

    # Build a data frame containing the label group information for
    # future evaluation
    classes = np.unique(y_train)
    nclasses = len(classes)
    group_vect = np.empty(shape=(n_vals * nclasses))
    count = 0
    for i in range(n_vals):
        for j in range(nclasses):
            for k in range(len(label_groups[i])):
                if j in label_groups[i][k]:
                    group_vect[count] = k
                    count += 1
                    break

    group_df = pd.DataFrame(
        data={"id": np.repeat(ids, nclasses),
              "label": np.tile(classes, n_vals),
              "group": group_vect}
    )

    # Return the best model model and the DataFrames that we will evaluate
    # later
    return {"models": k_res[best_model]["models"],
            "train_time": total_train_time,
            "search_df": search_df, "group_df": group_df,
            "label_groups": label_groups[best_model],
            "ids": ids, "best_model": best_model}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_num", type=int)
    parser.add_argument("method", type=str)
    parser.add_argument("estimator", type=str)
    parser.add_argument("use_meta", type=int)
    parser.add_argument("--niter", type=int, nargs="?", default=10)
    parser.add_argument("--k_vals", type=int, nargs="*",
                        default=list(range(2, 62)))
    parser.add_argument("--wd", type=str, nargs="?",
                        default="/pool001/zblanks/label_reduction_data/fmow")
    args = vars(parser.parse_args())

    # Get the data
    if args["use_meta"] == 1:
        f_train = h5py.File(path.join(args["wd"], "train_meta.h5"), "r")
    else:
        f_train = h5py.File(path.join(args["wd"], "train.h5"), "r")

    X_train = np.array(f_train["X_train"])
    y_train = np.array(f_train["y_train"]).flatten()

    # Down-sample the data to make sure everything works
    splitter = StratifiedShuffleSplit(
        n_splits=1, test_size=0.75, random_state=17
    )
    idx = splitter.split(X_train, y_train)
    idx = [val for val in idx]
    train_idx = idx[0][0]
    X_train = X_train[train_idx]
    y_train = y_train[train_idx]

    # Split the data into training and validation so we can determine the
    # best k
    splitter = StratifiedShuffleSplit(
        n_splits=1, test_size=0.20, random_state=17
    )

    idx = splitter.split(X_train, y_train)
    idx = [val for val in idx]
    train_idx = idx[0][0]
    val_idx = idx[0][1]
    train_X = X_train[train_idx]
    train_y = y_train[train_idx]
    val_X = X_train[val_idx]
    val_y = y_train[val_idx]

    # Save the val_y to disk so that we can evaluate the prediction
    # results later
    label_path = path.join(args["wd"], "test_labels.csv")
    n = len(val_y)
    label_data = np.zeros((n,), dtype=[("sample_id", int), ("label", int)])
    label_data["sample_id"] = np.arange(n)
    label_data["label"] = val_y

    if not path.exists(label_path):
        np.savetxt(label_path, X=label_data, fmt="%i,%i",
                   header="sample_id,label")

    # To allow us to have an understanding of the distribution of our
    # estimator's performance we are going bootstrap the training data
    rng = np.random.RandomState(args["run_num"])
    train_X, train_y = resample(train_X, train_y, random_state=rng)

    # Determine if we need to just use a flat classifier or the HC
    if args["method"] == "f":
        res = flat_model(train_X, train_y, val_X, rng,
                         args["estimator"], args["niter"])

        # Get the hash ID for the flat classifier
        ids = gen_id(args)

        # Save the probability predictions to disk
        proba_pred = res["proba_pred"]
        file = "f_" + ids[0] + ".csv"
        savepath = path.join(args["wd"], "proba_pred", file)
        np.savetxt(fname=savepath, X=proba_pred)

        # Save the FC results to disk
        fc_df = pd.DataFrame(
            data={"id": ids, "metric": ["train_time"],
                  "value": [res["train_time"]]}
        )

        file = path.join(args["wd"], "fc_prelim_res.csv")
        if path.exists(file):
            fc_df.to_csv(file, mode="a", header=False, index=False)
        else:
            fc_df.to_csv(file, index=False)

    else:
        # Determine the best value for k in validation
        k_search_res = get_best_k(train_X, train_y, args)

        # Test the best model out of sample
        res = hc_pred(k_search_res["models"], val_X,
                      k_search_res["label_groups"])

        # Grab the ids
        ids = k_search_res["ids"]
        best_id = ids[k_search_res["best_model"]]

        # Grab the DataFrames detailing the group information and the
        # validation search results
        search_df = k_search_res["search_df"]
        group_df = k_search_res["group_df"]

        # Save the DataFrames to disk
        search_path = path.join(args["wd"], "search_res.csv")
        group_path = path.join(args["wd"], "group_res.csv")

        if path.exists(search_path):
            search_df.to_csv(search_path, mode="a", header=False, index=False)
        else:
            search_df.to_csv(search_path, index=False)

        if path.exists(group_path):
            group_df.to_csv(group_path, mode="a", header=False, index=False)
        else:
            group_df.to_csv(group_path, index=False)

        # Save the probability predictions to disk
        root_path = "root_" + best_id + ".csv"
        root_path = path.join(args["wd"], "proba_pred", root_path)
        root_proba_pred = res["root_proba_pred"]
        np.savetxt(root_path, X=root_proba_pred)

        node_path = "node_" + best_id + ".pickle"
        node_path = path.join(args["wd"], "proba_pred", node_path)
        node_proba_preds = res["node_proba_preds"]
        with open(node_path, "wb") as p:
            pickle.dump(node_proba_preds, p)

        full_path = "hc_" + best_id + ".csv"
        full_path = path.join(args["wd"], "proba_pred", full_path)
        proba_pred = res["proba_pred"]
        np.savetxt(full_path, X=proba_pred)

    # Record the experimental settings with the ID hash and the various
    # settings
    exp_df = create_experiment_df(args, ids)
    exp_path = path.join(args["wd"], "experiment_settings.csv")
    if path.exists(exp_path):
        exp_df.to_csv(exp_path, mode="a", header=False, index=False)
    else:
        exp_df.to_csv(exp_path, index=False)

    return None


if __name__ == "__main__":
    main()

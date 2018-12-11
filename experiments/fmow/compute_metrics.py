import numpy as np
from sklearn.metrics import roc_auc_score
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import re
import os


def compute_auc(y_true: np.ndarray, y_proba_pred: np.ndarray) -> float:
    """
    Computes the AUC of the predictions
    """

    # One-hot encode the y_true vector
    enc = OneHotEncoder(sparse=False)
    true_y = enc.fit_transform(y_true.reshape(-1, 1))

    # Compute the auc
    return roc_auc_score(true_y, y_proba_pred)


def top_k_accuracy(y_true: np.ndarray, y_proba_pred: np.ndarray,
                   k: int) -> float:
    """
    Computes the top K accuracy for a given set of predictions
    """

    # Account for the case when k = 1
    n = y_true.shape[0]
    if k == 1:
        # The top 1 prediction is simply the argmax of the probabilities
        y_pred = y_proba_pred.argmax(axis=1)

        # Compute the accuracy
        return np.sum(y_true == y_pred) / n
    else:
        p = y_proba_pred.shape[1]

        # Get the top k predictions
        top_k_pred = np.argsort(y_proba_pred, axis=1)[:, (p - k):]

        # Go through each sample and see if the given true sample
        # belongs in the top k
        in_top_k = [y_true[i] in top_k_pred[i, :] for i in range(n)]
        return np.sum(in_top_k) / n


def node_top1(y_true: np.ndarray, y_proba_pred: np.ndarray,
              label_map: dict) -> float:
    """
    Computes the node_top1 metric for the flat classifier (which has to be
    done slightly differently versus the HC
    """
    # A label is counted as correct if (for example S = {{1, 2}, {3, 4}, {5}}
    # and the arg-max prediction for the FC was either 1 or 2 and the node
    # label is 1

    # Get the arg-max predictions for each sample
    y_pred = y_proba_pred.argmax(axis=1)

    # Re-map the predictions according to the label map
    y_pred = remap_target(y_pred, label_map)

    # Count the number of correct predictions
    return (y_pred == y_true).sum() / y_true.shape[0]


def remap_root_probs(proba_pred: np.ndarray,
                     label_map: dict) -> np.ndarray:
    """
    Re-maps the provided probabilities for a FC so that we can compute the root
    metrics in addition to the leaf metrics
    """

    # Define the new probability prediction matrix
    n = proba_pred.shape[0]
    ngroups = len(label_map)
    new_y_proba_pred = np.empty(shape=(n, ngroups))

    # For each of the label groups we are going to the sum the
    # probabilities that correspond to the given meta-class
    for i in range(ngroups):
        new_y_proba_pred[:, i] = proba_pred[:, label_map[i]].sum(axis=1)
    return new_y_proba_pred


def remap_target(y_true: np.ndarray, label_map: dict) -> np.ndarray:
    """
    Re-maps the target vector to match the values in the meta-classes
    """

    # Define the new target vector and update the values according to the
    # dictionary
    new_y_true = np.empty_like(y_true)
    ngroups = len(label_map)
    for i in range(ngroups):
        idx = np.isin(y_true, label_map[i])
        new_y_true[idx] = i
    return new_y_true


def define_res_df(ids: np.ndarray, file_type: str) -> pd.DataFrame:
    """
    Defines an empty results DataFrame for the metrics we are computing
    """
    if file_type == "f":
        metrics = ["leaf_auc", "leaf_top1", "leaf_top3", "node_auc",
                   "node_top1"]
    elif file_type == "hc":
        metrics = ["leaf_auc", "leaf_top1", "leaf_top3"]
    else:
        metrics = ["node_auc", "node_top1"]

    # Build the DataFrame
    n = len(ids)
    nmetrics = len(metrics)
    metrics = np.tile(metrics, n)
    ids = np.repeat(ids, nmetrics)
    values = np.empty(shape=(n * nmetrics))
    df = pd.DataFrame(data={"id": ids, "metric": metrics, "value": values})
    return df


def compute_metrics(files: list, file_type: str, y_true: np.ndarray, wd: str,
                    label_maps: dict) -> pd.DataFrame:
    """
    Computes the metrics for the given file type
    """

    # Read in the probability files
    file_ids = np.array([extract_id(file) for file in files])

    # If the file type is "f" we have account for bad IDs
    if file_type == "f":
        all_ids = list(label_maps.keys())
        bad_ids = np.setdiff1d(file_ids, all_ids)
        good_ids = ~np.isin(file_ids, bad_ids)
        file_ids = file_ids[good_ids]

    # Define a placeholder DataFrame to store our results
    df = define_res_df(file_ids, file_type)

    # Iterate through every file ID and compute the relevant metric
    for id_val in file_ids:
        # Get the appropriate probability prediction given the ID
        path = os.path.join(wd, file_type + "_" + id_val + ".csv")
        proba_pred = np.loadtxt(path)

        # Account for the possibility that there might be a NaN and we have
        # to remove these indices from the predictions and the true vector
        bad_idx = np.unique(np.argwhere(~np.isfinite(proba_pred))[:, 0])

        if len(bad_idx) >= 1:
            print("File type: {}; Bad idx: {}".format(file_type, bad_idx))

        good_idx = np.setdiff1d(np.arange(proba_pred.shape[0]), bad_idx)
        true_y = y_true[good_idx]
        proba_pred = proba_pred[good_idx, :]

        # We will compute different metrics depending on the file type we
        # are working with
        if file_type == "f":
            # Compute the leaf metrics
            leaf_auc = compute_auc(true_y, proba_pred)
            leaf_top1 = top_k_accuracy(true_y, proba_pred, k=1)
            leaf_top3 = top_k_accuracy(true_y, proba_pred, k=3)

            # Compute the root metrics
            y_root = remap_target(true_y, label_maps[id_val])
            root_proba_pred = remap_root_probs(proba_pred, label_maps[id_val])
            node_auc = compute_auc(y_root, root_proba_pred)
            n_top1 = node_top1(y_root, proba_pred, label_maps[id_val])

            # Update the values in the results DataFrame
            values = [leaf_auc, leaf_top1, leaf_top3, node_auc, n_top1]
            df.loc[df["id"] == id_val, "value"] = values

        elif file_type == "hc":
            # Compute the leaf metrics
            leaf_auc = compute_auc(true_y, proba_pred)
            leaf_top1 = top_k_accuracy(true_y, proba_pred, k=1)
            leaf_top3 = top_k_accuracy(true_y, proba_pred, k=3)

            values = [leaf_auc, leaf_top1, leaf_top3]
            df.loc[df["id"] == id_val, "value"] = values

        else:
            # Re-map the target vector to work with the root predictions
            y_root = remap_target(true_y, label_maps[id_val])
            node_auc = compute_auc(y_root, proba_pred)
            n_top1 = top_k_accuracy(y_root, proba_pred, k=1)

            values = [node_auc, n_top1]
            df.loc[df["id"] == id_val, "value"] = values

    return df


def extract_id(file: str) -> str:
    """
    Extracts the id from the file name
    """

    # Remove the model identifier
    path = re.search(r"_.*", file).group()

    # Remove the _ and .csv
    return re.sub(r"(.csv)|(_)", "", path)


def get_files(path: str) -> dict:
    """
    Get the files that correspond to the probability predictions
    """
    files = os.listdir(path)

    # Remove the files that are not hc, f, or root
    new_files = []
    for file in files:
        if re.match(r"(root_|hc_|f_)", file):
            new_files.append(file)

    # Separate the files by file type
    root_files = []
    hc_files = []
    fc_files = []
    for file in new_files:
        if re.match("root_", file):
            root_files.append(file)
        elif re.match("hc_", file):
            hc_files.append(file)
        else:
            fc_files.append(file)

    return {"root": root_files, "hc": hc_files, "f": fc_files}


def infer_bad_ids(fc_ids: list, hc_ids: list, settings_df: pd.DataFrame):
    """
    Infers the the IDs that correspond to a difference in the run number
    because of potential issues with the function and thus we need
    to remove those from the FC list so that we don't have errors
    downstream
    """

    # Get the run numbers that correspond to the given IDs for both the
    # FC and HC
    use_meta_vals = [0, 0, 1, 1]
    method_ids = [fc_ids, hc_ids, fc_ids, hc_ids]
    n = len(use_meta_vals)
    run_nums = [
        settings_df.loc[(settings_df["id"].isin(method_ids[i])) &
                        (settings_df["use_meta"] == use_meta_vals[i]),
                        "run_num"]
        for i in range(n)
    ]

    # Check what the difference is between these values -- i.e. if
    # fc = {0, 1, 2, 3} and hc = {0, 1, 2} we know that 3 did not run
    # and thus need to remove it so that we don't have downstream issues
    set_diff_no_meta = np.setdiff1d(run_nums[0], run_nums[1])
    set_diff_use_meta = np.setdiff1d(run_nums[2], run_nums[3])

    # Grab the bad IDs for both cases
    bad_no_meta_ids = settings_df.loc[
        (settings_df["run_num"].isin(set_diff_no_meta)) &
        (settings_df["method"] == "f") &
        (settings_df["use_meta"] == 0),
        "id"
    ]

    bad_use_meta_ids = settings_df.loc[
        (settings_df["run_num"].isin(set_diff_use_meta)) &
        (settings_df["method"] == "f") &
        (settings_df["use_meta"] == 1),
        "id"
    ]

    # Combine the bad IDs
    return pd.concat([bad_no_meta_ids, bad_use_meta_ids])


def get_label_map(run_id: str, df: pd.DataFrame) -> dict:
    """
    Gets the label map given the run ID
    """

    # Subset the DataFrame for the given ID
    df = df.loc[df["id"] == run_id, ["label", "group"]]

    # Grab the unique group values for this given ID
    group_vals = np.unique(df["group"])
    label_map = {}
    for val in group_vals:
        label_map[val] = df.loc[df["group"] == val, "label"].values
    return label_map


def infer_label_maps(fc_files: list, hc_files: list,
                     settings_df: pd.DataFrame,
                     group_df: pd.DataFrame) -> dict:
    """
    Infers the label maps to for computing the root metrics for the
    FC from the best model for a given run number
    """

    # Grab the list of file IDs from the FC files and HCI files
    fc_ids = [extract_id(file) for file in fc_files]
    hc_ids = [extract_id(file) for file in hc_files]

    # Get the IDs that will cause problems in our analysis
    bad_ids = infer_bad_ids(fc_ids, hc_ids, settings_df)

    # Remove the FC ids that belong to the bad set
    fc_ids = np.setdiff1d(fc_ids, bad_ids)

    # Check that the FC IDs and the HC IDs have the same length to ensure
    # we don't have errors downstream
    assert len(fc_ids) == len(hc_ids)

    # To speed up the search we are going to have the IDs ordered by
    # run number and use_meta so that we can just zip along them
    fc_ids = settings_df.loc[settings_df["id"].isin(fc_ids)].sort_values(
        by=["use_meta", "run_num"]).id

    hc_ids = settings_df.loc[settings_df["id"].isin(hc_ids)].sort_values(
        by=["use_meta", "run_num"]).id

    # Build the label maps for each of the FC and HC runs
    label_maps = {}
    for (fc_id, hc_id) in zip(fc_ids, hc_ids):
        tmp_label_map = get_label_map(hc_id, group_df)
        label_maps[fc_id] = tmp_label_map
        label_maps[hc_id] = tmp_label_map

    return label_maps


def main():
    # Get all of the probability files
    wd = "/pool001/zblanks/label_reduction_data/fmow/proba_pred"
    prob_files = get_files(wd)

    # Get the target vector
    path = "/pool001/zblanks/label_reduction_data/fmow/test_labels.csv"
    y_true = pd.read_csv(path)
    y_true = y_true.label.values

    # Get the DataFrame containing all of the group information given
    # the ID and the experiment settings so that we can get the inferred
    # label map from the data
    path = "/pool001/zblanks/label_reduction_data/fmow/group_res.csv"
    group_df = pd.read_csv(path)
    group_df["group"] = group_df["group"].astype(int)

    path = "/pool001/zblanks/label_reduction_data/fmow/experiment_settings.csv"
    settings_df = pd.read_csv(path)

    # Get the label_maps for the FC
    label_maps = infer_label_maps(prob_files["f"], prob_files["hc"],
                                  settings_df, group_df)

    # Compute the relevant metrics for each of the file types
    metrics_res = [pd.DataFrame()] * len(prob_files.keys())
    for (i, key) in enumerate(prob_files.keys()):
        print("Evaluating: " + key)
        metrics_res[i] = compute_metrics(prob_files[key], key, y_true,
                                         wd, label_maps)

    # Concatenate the results
    df = pd.concat(metrics_res, ignore_index=True)

    # Save the results to disk
    path = "/pool001/zblanks/label_reduction_data/fmow/leaf_root_res.csv"
    df.to_csv(path, index=False)


if __name__ == "__main__":
    main()

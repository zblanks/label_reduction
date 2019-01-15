import pandas as pd
import numpy as np
from itertools import combinations
import re
import os
import hashlib
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils import resample
from joblib import Parallel, delayed


def find_exp_pairs(df: pd.DataFrame, exp_vars: list,
                   consider_cols: np.ndarray):
    """
    Finds all experiment pairs so we can find which ones go together and
    correct any potential run mis-matches
    """

    # First generate all choose(n, 2) row combinations to compare
    row_combos = combinations(df.index, 2)

    # Go through every row combination and check if it is match with one
    # another; we can tell it is match if the following conditions hold
    # 1. There is only one difference between the rows
    # 2. The method cannot be the same EXCEPT if it's with the group_algo AND
    # method == "hci"
    exp_pairs = []
    for combo in row_combos:
        sub_df = df.iloc[list(combo), :]
        sub_df = sub_df.loc[:, consider_cols].reset_index(drop=True)

        # First calculate the number of differences between the rows
        row_diffs = sub_df.nunique()
        only_one_diff = len(np.where(row_diffs.values > 1)[0]) == 1

        # Check if methods are different
        diff_methods = sub_df.loc[0, "method"] != sub_df.loc[1, "method"]

        # UNLESS it's HCI and group_algo are different
        both_hci = np.all(sub_df["method"] == "hci")
        diff_group_algo = sub_df.loc[0, "group_algo"] != sub_df.loc[1, "group_algo"]

        # Combine the second condition
        second_cond = diff_methods or (both_hci and diff_group_algo)

        # Check if both the first and second condition are true; if so then
        # we know they are experiment pairs
        if only_one_diff and second_cond:
            # Generate two string statements that define the filtering
            # conditions for the first and second experiments that go
            # with one another (ex: "(use_meta == 0) & (method == 'f')")
            # This will be used as a "query" for a DataFrame

            # Go through both of the experiment row pairs
            filter_conds = {"query": [""] * 2, "method0": "", "method1": ""}
            for i in range(2):
                n = len(exp_vars)
                query_conds = [""] * n

                # Generate the query string
                for j in range(n):
                    var = exp_vars[j]
                    var_val = sub_df.loc[i, var]
                    if isinstance(var_val, str):
                        var_val = '"{}"'.format(var_val)
                    else:
                        var_val = str(var_val)

                    query_conds[j] = '(' + var + ' == ' + var_val + ')'

                # Join the query string
                query_str = " & ".join(query_conds)
                filter_conds["query"][i] = query_str

                # Add the i-th method
                dict_key = "method" + str(i)
                filter_conds[dict_key] = sub_df.loc[i, "method"]

            # Add both of the filtering conditions to the overall experimental
            # list to be used for later
            exp_pairs.append(filter_conds)

    return exp_pairs


def infer_uniq_experiments(exp_df: pd.DataFrame, exp_vars: list):
    """
    Infers the unique experiment pairs and returns them as a list of query
    strings
    """

    # First we need to get the unique rows in the DataFrame excluding the
    # ID and run number
    all_cols = exp_df.columns.values
    consider_cols = np.setdiff1d(all_cols, ['id', 'run_num', 'k'])
    uniq_df = exp_df.drop_duplicates(subset=consider_cols)
    uniq_df.reset_index(drop=True, inplace=True)

    # Quick look at how the experiment function is working
    return find_exp_pairs(uniq_df, exp_vars, consider_cols)


def extract_id(file: str) -> str:
    """
    Extracts the id from the file name
    """

    # Remove the model identifier
    path = re.search(r"_.*", file).group()

    # Remove the _ and .csv
    return re.sub(r"(.npy)|(_)", "", path)


def get_files(path: str) -> dict:
    """
    Get the files that correspond to the probability predictions
    """
    files = os.listdir(path)

    # Remove the files that are not hc, f, or root
    new_files = []
    for file in files:
        if re.match(r"root_|hc_|f_|idx_", file):
            new_files.append(file)

    # Separate the files by file type
    node_files = []
    hc_files = []
    fc_files = []
    idx_files = []
    for file in new_files:
        if re.match("root_", file):
            node_files.append(file)
        elif re.match("hc_", file):
            hc_files.append(file)
        elif re.match("idx_", file):
            idx_files.append(file)
        else:
            fc_files.append(file)

    return {"node": node_files, "hci": hc_files, "f": fc_files,
            "idx": idx_files}


def get_good_ids(exp_df: pd.DataFrame, method0_ids: list, method1_ids: list,
                 query: list):
    """
    Infers the run numbers that were mis-matched during experimentation
    """

    # Get the run numbers for method 0 and method 1 using their respective
    # query string
    method0_df = exp_df.query(query[0])
    method0_runs = method0_df.loc[method0_df["id"].isin(method0_ids), "run_num"]

    method1_df = exp_df.query(query[1])
    method1_runs = method1_df.loc[method1_df["id"].isin(method1_ids), "run_num"]

    # We need the intersection of the run numbers between the methods so we can
    # infer which IDs need to be removed; we will add this to the query string
    # and then extract the appropriate run IDs
    good_runs = np.intersect1d(method0_runs, method1_runs)

    # Now we need to use the set of good runs, add this to each of the query
    # strings so we can extract the final IDs for each of the methods
    run_str = '[' + ", ".join(str(val) for val in good_runs) + ']'
    run_query = ' & (run_num in ' + run_str + ')'

    # Get the new method0 and method1 IDs
    query0 = query[0] + run_query
    query1 = query[1] + run_query

    # Get the updated IDs for each method
    method0_df = exp_df.query(query0)
    method0_df = method0_df[method0_df['id'].isin(method0_ids)]
    method0_ids = method0_df.sort_values(by='run_num').id.tolist()

    method1_df = exp_df.query(query1)
    method1_df = method1_df[method1_df['id'].isin(method1_ids)]
    method1_ids = method1_df.sort_values(by='run_num').id.tolist()

    return method0_ids, method1_ids


def get_label_map(run_id: str, df: pd.DataFrame) -> np.ndarray:
    """
    Gets the label map given the run ID
    """

    # Subset the DataFrame for the given ID
    return df[df['id'] == run_id].sort_values(by="label").group.values.astype(int)


def infer_label_map(group_df: pd.DataFrame, method0: str, method1: str,
                    method0_ids: list, method1_ids: list):
    """
    Gets the final label map so that we can compute node-based metrics
    """

    # Define a placeholder to store all of the label maps for the
    n = len(method0_ids) + len(method1_ids)

    if method0 == 'f':
        nlabels = group_df.loc[group_df['id'] == method1_ids[0], 'group'].shape[0]
    else:
        nlabels = group_df.loc[group_df['id'] == method0_ids[0], 'group'].shape[0]

    all_ids = method0_ids + method1_ids
    empty_maps = np.empty(shape=(n, nlabels))
    label_maps = dict(zip(all_ids, empty_maps))

    # If either method0 or method1 is "f", then we need to simply get the label
    # map for the other method and use that to compute node metrics
    if method0 == "f":
        # Iterate through each of the IDs, get the map for method1 and add
        # the label maps for each
        for (method0_id, method1_id) in zip(method0_ids, method1_ids):
            # Get the label map
            label_map = get_label_map(method1_id, group_df)

            # Add the same label map to each
            label_maps[method0_id] = label_map
            label_maps[method1_id] = label_map
    elif method1 == "f":
        # Same thing as above just reversed
        for (method0_id, method1_id) in zip(method0_ids, method1_ids):
            label_map = get_label_map(method0_id, group_df)
            label_maps[method0_id] = label_map
            label_maps[method1_id] = label_map
    else:
        # If neither are f then we need to get the label map for each for
        # each of them
        for (method0_id, method1_id) in zip(method0_ids, method1_ids):
            label_map0 = get_label_map(method0_id, group_df)
            label_map1 = get_label_map(method1_id, group_df)

            label_maps[method0_id] = label_map0
            label_maps[method1_id] = label_map1

    return label_maps


def get_final_files(files: list, ids: list, proba_path: str):
    """
    Gets the final set of files needed to compute the metrics given a set of
    experiment IDs
    """

    # Go through each of the IDs and get its corresponding file
    n = len(ids)
    final_files = [""] * n

    for i in range(n):
        # Get the index where the file is located with the given ID
        idx = np.flatnonzero(np.core.defchararray.find(files, ids[i]) != -1)[0]
        final_files[i] = os.path.join(proba_path, files[idx])

    return final_files


def compute_auc(y_true: np.ndarray, proba_mat: np.ndarray) -> float:
    """
    Computes the AUC of the predictions
    """

    # One-hot encode the y_true vector
    enc = OneHotEncoder(sparse=False, categories='auto')
    true_y = enc.fit_transform(y_true.reshape(-1, 1))

    # Compute the auc
    return roc_auc_score(true_y, proba_mat)


def top_k_accuracy(y_true: np.ndarray, proba_mat: np.ndarray,
                   k: int) -> float:
    """
    Computes the top K accuracy for a given set of predictions
    """

    # Account for the case when k = 1
    n = y_true.shape[0]
    if k == 1:
        # The top 1 prediction is simply the argmax of the probabilities
        y_pred = proba_mat.argmax(axis=1)

        # Compute the accuracy
        return np.sum(y_true == y_pred) / n
    else:
        p = proba_mat.shape[1]

        # Get the top k predictions
        top_k_pred = np.argsort(proba_mat, axis=1)[:, (p - k):]

        # Go through each sample and see if the given true sample
        # belongs in the top k
        in_top_k = [y_true[i] in top_k_pred[i, :] for i in range(n)]
        return np.sum(in_top_k) / n


def node_top1(y_true: np.ndarray, proba_mat: np.ndarray,
              label_map: np.ndarray) -> float:
    """
    Computes the node_top1 metric for the flat classifier (which has to be
    done slightly differently versus the HC
    """
    # A label is counted as correct if (for example S = {{1, 2}, {3, 4}, {5}}
    # and the arg-max prediction for the FC was either 1 or 2 and the node
    # label is 1

    # Get the arg-max predictions for each sample
    y_pred = proba_mat.argmax(axis=1)

    # Re-map the predictions according to the label map
    y_pred = remap_target(y_pred, label_map)

    # Count the number of correct predictions
    return (y_pred == y_true).sum() / y_true.shape[0]


def remap_node_probs(proba_pred: np.ndarray,
                     label_map: np.ndarray) -> np.ndarray:
    """
    Re-maps the provided probabilities for a FC so that we can compute the node
    metrics in addition to the leaf metrics
    """

    # Define the new probability prediction matrix
    n = proba_pred.shape[0]
    ngroups = len(np.unique(label_map))
    new_proba_mat = np.empty(shape=(n, ngroups))

    # For each of the label groups we are going to the sum the
    # probabilities that correspond to the given meta-class
    for i in range(ngroups):
        labels = label_map[np.where(label_map == i)]
        new_proba_mat[:, i] = proba_pred[:, labels].sum(axis=1)
    return new_proba_mat


def remap_target(y_true: np.ndarray, label_map: np.ndarray) -> np.ndarray:
    """
    Re-maps the target vector to match the values in the meta-classes
    """

    # Define the new target vector and update the values according to the
    # dictionary
    new_y_true = np.empty_like(y_true)
    nlabels = len(label_map)
    for i in range(nlabels):
        new_y_true[np.where(y_true == i)] = label_map[i]

    return new_y_true


def format_res_df(nfiles: int, ids: list):
    """
    Formats the results DataFrame in the expected format for later analysis
    """
    # metrics = ['leaf_auc', 'leaf_top1', 'leaf_top3', 'node_auc', 'node_top1']
    metrics = ['leaf_top1', 'leaf_top3', 'node_top1']
    nmetrics = len(metrics)
    metrics_vect = np.tile(metrics, nfiles)
    ids_vect = np.repeat(ids, nmetrics)
    df = pd.DataFrame({"id": ids_vect, "metric": metrics_vect,
                       "value": np.empty(shape=(nfiles * nmetrics))})

    return df


def flat_parallel(file: str, id_val: str, y_true: np.ndarray,
                  label_map: np.ndarray):
    """
    Helper function to run the metric computations in parallel
    """

    # Read in the file
    proba_pred = np.load(file)

    # Compute the leaf metrics
    leaf_top1 = top_k_accuracy(y_true, proba_pred, k=1)
    leaf_top3 = top_k_accuracy(y_true, proba_pred, k=3)

    # Compute the node metrics
    y_node = remap_target(y_true, label_map)
    n_top1 = node_top1(y_node, proba_pred, label_map)

    # Return the DataFrame result
    res = [leaf_top1, leaf_top3, n_top1]
    metrics = ['leaf_top1', 'leaf_top3', 'node_top1']

    return pd.DataFrame({"id": id_val, "metric": metrics, "value": res})


def compute_flat_metrics(files: list, ids: list, y_true: np.ndarray,
                         label_maps: dict):
    """
    Computes the relevant metrics for an FC
    """

    # Get the formated DataFrame
    # n = len(files)
    # df = format_res_df(n, ids)
    # nmetrics = len(df['metric'].unique())

    # Try in parallel
    with Parallel(n_jobs=-1, verbose=5) as p:
        dfs = p(delayed(flat_parallel)(file, id_val, y_true, label_maps[id_val])
                for (file, id_val) in zip(files, ids))

    df = pd.concat(dfs, ignore_index=True)

    # # Go through each file, compute the relevant metric, and then add it
    # # to the DataFrame
    # count = 0
    # for (file, id_val) in zip(files, ids):
    #     # Read in the probability file
    #     proba_pred = np.load(file)
    #
    #     # Compute each of the metrics
    #     # leaf_auc = compute_auc(y_true, proba_pred)
    #     leaf_top1 = top_k_accuracy(y_true, proba_pred, k=1)
    #     leaf_top3 = top_k_accuracy(y_true, proba_pred, k=3)
    #
    #     # Compute the node metrics
    #     y_node = remap_target(y_true, label_maps[id_val])
    #     # node_proba_pred = remap_node_probs(proba_pred, label_maps[id_val])
    #     # node_auc = compute_auc(y_node, node_proba_pred)
    #     n_top1 = node_top1(y_node, proba_pred, label_maps[id_val])
    #
    #     # Combine all of the results and add them to the DataFrame
    #     # res = [leaf_auc, leaf_top1, leaf_top3, node_auc, n_top1]
    #     res = [leaf_top1, leaf_top3, n_top1]
    #     df.loc[count:(count + nmetrics - 1), 'value'] = res
    #     count += nmetrics

    return df


def hc_parallel(leaf_file: str, node_file: str, id_val: str, y_true: np.ndarray,
                label_map: np.ndarray, idx_file: str):
    """
    Parallel helper function for computing HC metrics
    """

    # Read in the leaf, node, and idx files
    leaf_proba_pred = np.load(leaf_file)
    node_proba_pred = np.load(node_file)
    idx = np.load(idx_file)

    # Update the y_true vector
    true_y = y_true[idx]

    # Compute the leaf metrics
    leaf_top1 = top_k_accuracy(true_y, leaf_proba_pred, k=1)
    leaf_top3 = top_k_accuracy(true_y, leaf_proba_pred, k=3)

    # Compute the node metrics
    y_node = remap_target(y_true, label_map)
    n_top1 = top_k_accuracy(y_node, node_proba_pred, k=1)

    # Format the results
    res = [leaf_top1, leaf_top3, n_top1]
    metrics = ["leaf_top1", "leaf_top3", "node_top1"]
    return pd.DataFrame({"id": id_val, "metric": metrics, "value": res})


def compute_hc_metrics(leaf_files: list, node_files: list, ids: list,
                       y_true: np.ndarray, label_maps: dict,
                       idx_files: list):
    """
    Computes the metrics for a HC
    """

    # Get the formatted DataFrame
    n = len(leaf_files)
    # df = format_res_df(n, ids)
    # nmetrics = len(df.metric.unique())

    # Compute the metrics in parallel to decrease time
    with Parallel(n_jobs=-1, verbose=5) as p:
        dfs = p(delayed(hc_parallel)(
            leaf_file, node_file, id_val, y_true, label_maps[id_val], idx_file)
        for (leaf_file, node_file, id_val, idx_file) in zip(leaf_files,
                                                            node_files,
                                                            ids, idx_files)
        )

    df = pd.concat(dfs, ignore_index=True)

    # # Go through each file and compute the relevant metrics
    # count = 0
    # for (leaf_file, node_file, id_val, idx_file) in zip(leaf_files, node_files,
    #                                                     ids, idx_files):
    #     # Get the leaf and node files
    #     leaf_proba_pred = np.load(leaf_file)
    #     node_proba_pred = np.load(node_file)
    #
    #     # Get the list of good indices to ensure that we don't have an
    #     # index error
    #     # TODO: remove when done
    #     try:
    #         idx = np.load(idx_file)
    #     except OSError:
    #         idx = np.arange(leaf_proba_pred.shape[0])
    #
    #     # Update the y_true vector with the correct indices
    #     true_y = y_true[idx]
    #
    #     # Check if the leaf_proba_pred has the same number of samples
    #     # the true_y vector
    #     assert true_y.shape[0] == leaf_proba_pred.shape[0]
    #
    #     # Compute the leaf metrics
    #     leaf_auc = compute_auc(true_y, leaf_proba_pred)
    #     leaf_top1 = top_k_accuracy(true_y, leaf_proba_pred, k=1)
    #     leaf_top3 = top_k_accuracy(true_y, leaf_proba_pred, k=3)
    #
    #     # Compute the node level metrics
    #     y_node = remap_target(y_true, label_maps[id_val])
    #     node_auc = compute_auc(y_node, node_proba_pred)
    #     n_top1 = top_k_accuracy(y_node, node_proba_pred, k=1)
    #
    #     # Combine all of the results and add them to the DataFrame
    #     res = [leaf_auc, leaf_top1, leaf_top3, node_auc, n_top1]
    #     df.loc[count:(count + nmetrics), 'value'] = res

    return df


def gen_id(id0: str, id1: str) -> str:
    """
    Generates a hash ID to identify an experiment pair
    """
    hash_str = id0 + id1
    hash_str = hash_str.encode("UTF-8")
    return hashlib.sha1(hash_str).hexdigest()


def format_boot_res(bootstrap_samples: int, pair_id: str, metric: str):
    """
    Generates the expected DataFrame output for the bootstrap distribution
    """

    # Define an empty DataFrame
    metric_vect = np.repeat(metric, bootstrap_samples)
    exp_id_vect = np.repeat(pair_id, bootstrap_samples)
    df = pd.DataFrame({"pair_id": exp_id_vect,
                       "niter": np.arange(bootstrap_samples),
                       "metric": metric_vect,
                       "value": np.empty(shape=(bootstrap_samples,))})

    return df


def format_pair_map_df(pair_id: str, id0: str, id1: str):
    """
    Formats the DataFrame which maps the experiment pair back to their
    original settings
    """
    return pd.DataFrame({"pair_id": pair_id, "id0": id0, "id1": id1})


def gen_boot_mean(metric_vals: np.ndarray, bootstrap_samples: int):
    """
    Generates the bootstrapped mean vector
    """

    n = len(metric_vals)
    boot_vals = resample(metric_vals, n_samples=(n * bootstrap_samples),
                         random_state=17).reshape(bootstrap_samples, n)
    return boot_vals.mean(axis=1)


def mean_diff(mean0: np.ndarray, mean1: np.ndarray):
    """
    Computes the mean difference between the first and second bootstrapped
    mean vectors
    """
    perc_diff = (mean1 - mean0) / mean0
    return perc_diff * 100


def get_boot_distn(first_df: pd.DataFrame, second_df: pd.DataFrame,
                   metric: str, bootstrap_samples: int, id0: str,
                   id1: str):
    """
    Computes the bootstrap distribution comparing the first and second
    experiment to one another for a given metric
    """

    # First subset the first and second DataFrame on the given metric
    metric0 = first_df.loc[first_df['metric'] == metric, 'value'].values
    metric1 = second_df.loc[second_df['metric'] == metric, 'value'].values

    # Generate the bootstrap distribution for the given metric for each
    # DataFrame
    mean0 = gen_boot_mean(metric0, bootstrap_samples)
    mean1 = gen_boot_mean(metric1, bootstrap_samples)

    # Compute the mean difference between the two vectors
    diff_vect = mean_diff(mean0, mean1)

    # Store the results in a DataFrame
    pair_id = gen_id(id0, id1)
    df = format_boot_res(bootstrap_samples, pair_id, metric)
    df['metric'] = diff_vect

    # Get the experiment pair map
    pair_df = format_pair_map_df(pair_id, id0, id1)
    return df, pair_df


def get_all_boot_distns(first_df: pd.DataFrame, second_df: pd.DataFrame,
                        bootstrap_samples: int, id0: str,
                        id1: str):
    """
    Gets the bootstrap distributions for all metrics
    """

    # Compute the distribution for each metric
    metrics = first_df.metric.unique.tolist()
    # nmetrics = len(metrics)
    # pair_dfs = [pd.DataFrame()] * nmetrics
    # boot_dfs = [pd.DataFrame()] * nmetrics

    with Parallel(n_jobs=-1, verbose=5) as p:
        res = p(delayed(get_boot_distn)(first_df, second_df, metric,
                                        bootstrap_samples, id0, id1)
                for metric in metrics)

    # Get the final DataFrames
    n = len(res)
    boot_df = pd.concat([res[i][0] for i in range(n)], ignore_index=True)
    pair_df = pd.concat([res[i][1] for i in range(n)], ignore_index=True)

    # for i in range(nmetrics):
    #     boot_dfs[i], pair_dfs[i] = get_boot_distn(
    #         first_df, second_df, metrics[i], bootstrap_samples, id0, id1
    #     )
    #
    # # Combine all of the DataFrames
    # boot_df = pd.concat(boot_dfs, ignore_index=True)
    # pair_df = pd.concat(pair_dfs, ignore_index=True)
    return boot_df, pair_df


def compute_metrics(exp_pair: dict, prob_files: dict, group_df: pd.DataFrame,
                    exp_df: pd.DataFrame, y_true: np.ndarray,
                    bootstrap_samples: int, proba_path: str):
    """
    Computes the relevant metrics for a given experiment pair
    """

    # Get the relevant files for the first and second method of the
    # experiment
    method0_files = prob_files[exp_pair['method0']]
    method1_files = prob_files[exp_pair['method1']]

    # From the files, extract the IDs
    method0_ids = list(map(extract_id, method0_files))
    method1_ids = list(map(extract_id, method1_files))

    # Get the updated list of IDs that we will use for the experiment
    # correcting for mis-aligned runs
    method0_ids, method1_ids = get_good_ids(exp_df, method0_ids, method1_ids,
                                            exp_pair['query'])

    # Using the new IDs, get the label maps for each the methods
    label_maps = infer_label_map(group_df, exp_pair['method0'],
                                 exp_pair['method1'], method0_ids, method1_ids)

    # Using the final IDs for the first and second methods, we need to get
    # the final set of files for both methods and the node-level files
    if exp_pair['method0'] == 'f':
        # Get the leaf and node files for method1 and the total file for
        # method0
        hc_leaf = get_final_files(method1_files, method1_ids, proba_path)
        hc_node = get_final_files(prob_files['node'], method1_ids, proba_path)
        idx_files = get_final_files(prob_files['idx'], method1_ids, proba_path)
        f_all = get_final_files(method0_files, method0_ids, proba_path)

        # Compute the metrics for each of the file groups
        f_res = compute_flat_metrics(f_all, method0_ids, y_true, label_maps)
        hc_res = compute_hc_metrics(hc_leaf, hc_node, method1_ids, y_true,
                                    label_maps, idx_files)

        # Get the final bootstrap results for each metric and the DataFrame
        # mapping the experiment pair
        boot_df, pair_df = get_all_boot_distns(
            f_res, hc_res, bootstrap_samples, method0_ids[0], method1_ids[0]
        )

    elif exp_pair['method1'] == 'f':
        # Same thing as above except reversed; however, I always want the
        # comparison to be percent difference of HCI - FC therefore we will
        # pretend that it goes 0, 1 with FC and HC
        hc_leaf = get_final_files(method0_files, method0_ids, proba_path)
        hc_node = get_final_files(prob_files['node'], method0_ids, proba_path)
        idx_files = get_final_files(prob_files['idx'], method0_ids, proba_path)
        f_all = get_final_files(method1_files, method1_ids, proba_path)

        f_res = compute_flat_metrics(f_all, method1_files, y_true, label_maps)
        hc_res = compute_hc_metrics(hc_leaf, hc_node, method0_ids, y_true,
                                    label_maps, idx_files)

        # Get the final bootstrap results
        boot_df, pair_df = get_all_boot_distns(
            f_res, hc_res, bootstrap_samples, method0_ids[0], method1_ids[0]
        )

    else:
        # We're in the case of comparing grouping algorithms or alternative
        # methods so the order is irrelevant to me
        method0_leaf = get_final_files(method0_files, method0_ids, proba_path)
        method0_node = get_final_files(prob_files['node'], method0_ids,
                                       proba_path)
        idx0_files = get_final_files(prob_files['idx'], method0_ids, proba_path)

        method1_leaf = get_final_files(method1_files, method1_ids, proba_path)
        method1_node = get_final_files(prob_files['node'], method1_ids,
                                       proba_path)
        idx1_files = get_final_files(prob_files['idx'], method1_ids, proba_path)

        # Compute their experimental results
        method0_res = compute_hc_metrics(method0_leaf, method0_node,
                                         method0_ids, y_true, label_maps,
                                         idx0_files)

        method1_res = compute_hc_metrics(method1_leaf, method1_node,
                                         method1_ids, y_true, label_maps,
                                         idx1_files)

        # Get the final bootstrap results
        boot_df, pair_df = get_all_boot_distns(
            method0_res, method1_res, bootstrap_samples, method0_ids[0],
            method1_ids[0]
        )

    return boot_df, pair_df


def compare_experiments(exp_path: str, group_path: str, proba_path: str,
                        label_path: str, exp_vars: list,
                        bootstrap_samples: int):
    """
    Computes the metrics for all of the relevant experiments
    """

    # First we need to get all of the probability prediction files
    prob_files = get_files(proba_path)

    # Second we need the target vector
    y_true = pd.read_csv(label_path).values

    # Third we need the experiment settings and grouping results
    exp_df = pd.read_csv(exp_path)
    group_df = pd.read_csv(group_path)

    # Using the experiment settings DataFrame, we need to infer all of the
    # unique experiments that occurred
    exp_pairs = infer_uniq_experiments(exp_df, exp_vars)

    # For each of the experiment pairs we need to compute the relevant
    # metrics, bootstrap the results, and then return the results in
    # the expected format
    n = len(exp_pairs)
    boot_dfs = [pd.DataFrame()] * n
    pair_dfs = [pd.DataFrame()] * n
    for (i, exp_pair) in enumerate(exp_pairs):
        boot_dfs[i], pair_dfs[i] = compute_metrics(
            exp_pair, prob_files, group_df, exp_df, y_true, bootstrap_samples,
            proba_path
        )

    # Combine all of the DataFrames and return it to the user for further
    # analysis
    boot_df = pd.concat(boot_dfs, ignore_index=True)
    pair_df = pd.concat(pair_dfs, ignore_index=True)
    return boot_df, pair_df

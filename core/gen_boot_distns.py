import pandas as pd
import numpy as np
from itertools import combinations
import re
import os
import hashlib
from sklearn.utils import resample
from joblib import Parallel, delayed


def combine_dfs(df_list: list) -> pd.DataFrame:
    """
    Combines a list of DataFrames
    """
    df = pd.concat(df_list, ignore_index=True)
    df = df.reset_index(drop=True)
    return df


def get_proba_files(proba_path: str, experiment_id: str):
    """
    Gets the file(s) that correspond to a given experiment ID
    """

    # Get all of the base probability files in the proba_path
    basefiles = os.listdir(proba_path)

    # Go through each of the basefiles and identify all of the files that
    # contain the experiment ID
    good_files = [file for file in basefiles if experiment_id in file]

    # Go through each of the good files and add them a dictionary which
    # corresponds to their file type (ex: idx, hci, etc.)
    files = {}
    for file in good_files:
        final_file = os.path.join(proba_path, file)
        if re.match('root_', file):
            files['node'] = final_file
        elif re.match('hc_', file):
            files['hci'] = final_file
        elif re.match('idx_', file):
            files['idx'] = final_file
        else:
            files['f'] = final_file

    return files


def infer_uniq_experiments(exp_df: pd.DataFrame, exp_vars: list):
    """
    Infers the unique experiments and generates a query string so we can
    compute values later
    """

    # Only grab the unique rows in the DataFrame excluding the ID, run number,
    # and number of meta-classes
    uniq_df = exp_df.drop_duplicates(subset=exp_vars)
    uniq_df.reset_index(drop=True, inplace=True)

    # Generate all of the query strings
    n_experiments = len(uniq_df)
    n_vars = len(exp_vars)
    queries = [''] * n_experiments
    for i in range(n_experiments):
        # Define a placeholder for all of the query conditions
        query_conditions = [''] * n_vars
        for j in range(n_vars):
            var = uniq_df.loc[i, exp_vars[j]]
            if isinstance(var, str):
                var = '"{}"'.format(var)
            else:
                var = str(var)

            query_conditions[j] = '(' + exp_vars[j] + ' == ' + var + ')'

        # Join the query string
        query_str = ' & '.join(query_conditions)
        queries[i] = query_str

    return queries


def extract_id(file: str) -> str:
    """
    Extracts the ID from the file name
    """

    # Only work with the base file name
    basefile = os.path.basename(file)

    # Remove the model identifier
    path = re.search(r"_.*", basefile).group()

    # Remove the _ and .csv
    return re.sub(r".npy|_", "", path)


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


def compute_leaf_vals(files: list, experiment_ids: list, y_true: np.ndarray,
                      metric: str):
    """
    Computes the leaf top 1 for all of the experiments
    """

    # Get the probability matrices
    proba_mats = [np.load(file) for file in files]

    # Compute the leaf top 1 accuracy for all matrices
    if metric == 'leaf_top1':
        vals = [top_k_accuracy(y_true, proba_mat, k=1)
                for proba_mat in proba_mats]
    else:
        vals = [top_k_accuracy(y_true, proba_mat, k=3)
                for proba_mat in proba_mats]

    # Create a DataFrame to store the results
    return pd.DataFrame({'id': experiment_ids[0], 'metric': metric,
                         'value': vals})


def compute_leaf_value(prob_file: str, y_true: np.ndarray, metric: str):
    """
    Computes a leaf-level metric for a given experiment
    """

    # Load the probability matrix
    proba_mat = np.load(prob_file)

    # Compute the appropriate metric
    if metric == 'leaf_top1':
        val = top_k_accuracy(y_true, proba_mat, k=1)
    else:
        val = top_k_accuracy(y_true, proba_mat, k=3)

    return pd.DataFrame({'metric': [metric], 'value': [val]})


def compute_metrics(exp_df: pd.DataFrame, final_ids: np.ndarray,
                    query_str: str, metric: str, y_true: np.ndarray,
                    proba_path: str):
    """
    Computes the relevant metrics for a given experiment
    """

    # Get a DataFrame that only contains the relevant values from the query
    sub_df = exp_df.query(query_str)

    # Get the experiment IDs that correspond to the particular query
    experiment_ids = sub_df['id'].values

    # Get the intersection of our query IDs with the final IDs
    experiment_ids = np.intersect1d(experiment_ids, final_ids)

    # Go through each of the experiment IDs and get its files
    files = [get_proba_files(proba_path, exp_id) for exp_id in experiment_ids]

    # If the method is HCI then we need to account for the possibility
    # of some bad indices and thus need to adjust the target vectors
    n = len(files)
    exp_method = sub_df['method'].values[0]
    if exp_method == 'hci':
        idx_vecs = [np.load(files[i]['idx']) for i in range(n)]
        target_vecs = [y_true[idx] for idx in idx_vecs]
    else:
        target_vecs = [np.copy(y_true) for _ in range(n)]

    # Compute the relevant metric for all of the provided experiments
    if exp_method == 'hci':
        res_dfs = [compute_leaf_value(files[i]['hci'], target_vecs[i], metric)
                   for i in range(n)]
    else:
        res_dfs = [compute_leaf_value(files[i]['f'], target_vecs[i], metric)
                   for i in range(n)]

    res_df = combine_dfs(res_dfs)
    res_df['id'] = experiment_ids[0]
    return res_df


def get_boot_distn(df: pd.DataFrame, nsamples: int):
    """
    Generate the bootstrap distribution after computing the metric values
    for a given experiment
    """

    # Get the vector computed metric values
    vals = df['value'].values
    n = len(vals)
    boot_vals = resample(vals, n_samples=(n * nsamples), random_state=17)
    boot_vals = boot_vals.reshape(nsamples, n)
    boot_distn = boot_vals.mean(axis=1)

    # Convert the bootstrap distribution into a DataFrame
    exp_id = df['id'].values[0]
    metric = df['metric'].values[0]
    return pd.DataFrame({'exp_id': exp_id, 'metric': metric,
                         'value': boot_distn})


def get_boot_distns(exp_df: pd.DataFrame, final_ids: np.ndarray,
                    query_str: str, metrics: list, y_true: np.ndarray,
                    proba_path: str, nsamples: int):
    """
    Gets all the bootstrap distributions for a given query string and
    set of metrics
    """

    # Go through each of the metrics and compute the relevant metric values
    # and generate the bootstrap distribution
    n = len(metrics)
    boot_dfs = [pd.DataFrame()] * n
    for i in range(n):
        tmp_res = compute_metrics(exp_df, final_ids, query_str, metrics[i],
                                  y_true, proba_path)
        boot_dfs[i] = get_boot_distn(tmp_res, nsamples)

    # Combine the DataFrames
    return combine_dfs(boot_dfs)


def check_one_difference(df: pd.DataFrame):
    """
    Checks if a DataFrame has only one difference between the experiment
    settings with some exceptions
    """

    # If one of the methods is "f" in which case we need to ignore the
    # the group_algo because we gave an arbitrary value
    if 'f' in df.method.values:
        # Drop the group_algo column because it's not relevant
        new_df = df.drop(labels=['group_algo'], axis=1)
    else:
        new_df = df.copy()

    # Check if there is only one difference between the experiment settings
    new_df = new_df.drop(labels=['id'], axis=1)
    row_diffs = new_df.nunique()
    return len(np.where(row_diffs.values > 1)[0]) == 1


def gen_id(id0: str, id1: str) -> str:
    """
    Generates a hash ID to identify an experiment pair
    """
    hash_str = id0 + id1
    hash_str = hash_str.encode("UTF-8")
    return hashlib.sha1(hash_str).hexdigest()


def find_experiment_pairs(exp_df: pd.DataFrame, uniq_ids: np.ndarray,
                          exp_vars: list):
    """
    Finds all unique experiment pairs and generates a DataFrame mapping
    the ID to a pair ID
    """

    # First subset the experiment DataFrame with only the unique IDs and
    # the relevant variables
    df = exp_df.loc[exp_df['id'].isin(uniq_ids), exp_vars + ['id']]
    df.reset_index(drop=True, inplace=True)

    # Generate all choose(n, 2) experiment pairs
    exp_combos = combinations(df.index, 2)

    # Go through every row combination and check if it is match with one
    # another; we can tell it is match if the following conditions hold
    # 1. There is only one difference between the rows
    # 2. The method cannot be the same EXCEPT if it's with the group_algo AND
    # method == "hci"
    exp_pairs = []
    for (i, combo) in enumerate(exp_combos):
        sub_df = df.iloc[list(combo), :].reset_index(drop=True)

        # Check if there is only one difference with the exception for
        # method == 'f'
        only_one_diff = check_one_difference(sub_df)

        # Check if methods are different
        diff_methods = sub_df.loc[0, "method"] != sub_df.loc[1, "method"]

        # UNLESS it's HCI and group_algo are different
        both_hci = np.all(sub_df["method"] == "hci")
        diff_group_algo = sub_df.loc[0, "group_algo"] != sub_df.loc[1, "group_algo"]

        # Combine the second condition
        second_cond = diff_methods or (both_hci and diff_group_algo)

        # If both the first and second condition are true then we know they
        # are experiment pairs
        if only_one_diff and second_cond:
            # Generate a uniq ID for the pair
            exp_ids = sub_df['id'].values
            pair_id = gen_id(exp_ids[0], exp_ids[1])

            # Add the pair ID along with the experiment IDs
            exp_pairs.append((pair_id, exp_ids[0], exp_ids[1]))

    # Build the DataFrames mapping the combinations for the boot and pair
    # DataFrame
    pair_df = pd.DataFrame(exp_pairs, columns=['pair_id', 'id0', 'id1'])
    return pair_df


def get_final_ids(proba_path: str):
    """
    Gets the list of final experiment IDs that made it to the probability
    prediction phase
    """

    # Get the base file names
    basefiles = os.listdir(proba_path)

    # For each file we need to remove the method identifier (e.g., f_)
    # and we need to remove the file extension
    n = len(basefiles)
    final_ids = [''] * n
    for i in range(n):
        # Extract the ID from the file
        final_ids[i] = re.sub(r'.npy|_', '',
                              re.search(r'_.*', basefiles[i]).group())

    # Grab only the unique IDs
    return np.unique(final_ids)


def gen_boot_df(exp_path: str, proba_path: str, label_path: str,
                exp_vars: list, metrics: list, nsamples=1000):
    """
    Generates the bootstrap DataFrame so we can visualize the results
    downstream
    """

    # Get the target vector to compute the metrics
    y_true = pd.read_csv(label_path, header=None).values.flatten()

    # We need the experiment settings to infer the unique experiments
    exp_df = pd.read_csv(exp_path)

    # In the event that we have run the same experiment (i.e. maybe we updated
    # the algorithm) we need to check for duplicates with respect to the ID
    # and then remove the duplicates
    exp_df = exp_df[~exp_df.duplicated(subset=['id'], keep='last')]
    exp_df.to_csv(exp_path, index=False)

    # Determine the unique experiments in the data and generate query
    # strings to subset the experiments DataFrame
    exp_queries = infer_uniq_experiments(exp_df, exp_vars)

    # Grab the unique IDs from the final prediction files
    final_ids = get_final_ids(proba_path)

    # Using each of the query strings, get the corresponding bootstrap
    # DataFrames
    with Parallel(n_jobs=-1, verbose=5) as p:
        boot_dfs = p(delayed(get_boot_distns)(exp_df, final_ids, query_str,
                                              metrics, y_true, proba_path,
                                              nsamples)
                     for query_str in exp_queries)

    boot_df = combine_dfs(boot_dfs)

    # # Get all the unique experiment pairs to update boot_df
    # uniq_ids = boot_df['exp_id'].unique()
    # pair_df = find_experiment_pairs(exp_df, uniq_ids, exp_vars)
    # return boot_df, pair_df
    return boot_df

import pandas as pd
import numpy as np
import re
import os
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
    queries = [''] * n_experiments
    for i in range(n_experiments):
        # If we're working with a flat classifier then we need to drop the
        # group_algo part of the argument
        if uniq_df.loc[i, 'method'] == 'f':
            new_exp_vars = exp_vars[:]
            new_exp_vars.remove('group_algo')
        else:
            new_exp_vars = exp_vars[:]

        # Define a placeholder for all of the query conditions
        n_vars = len(new_exp_vars)
        query_conditions = [''] * n_vars
        for j in range(n_vars):
            var = uniq_df.loc[i, new_exp_vars[j]]

            if isinstance(var, str):
                var = '"{}"'.format(var)
            else:
                var = str(var)

            query_conditions[j] = '(' + new_exp_vars[j] + ' == ' + var + ')'

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


def permute_node_target(y_node: np.ndarray, nsamples=1000):
    """
    Permutes the node target vector so that we can compute the distribution
    of node-level performance
    """
    rng = np.random.RandomState(17)
    y_random = np.array([rng.permutation(y_node) for _ in range(nsamples)])
    return y_random.reshape(nsamples, len(y_node))


def get_label_map(exp_id: str, df: pd.DataFrame) -> np.ndarray:
    """
    Gets the label map given the run ID
    """

    # Subset the DataFrame for the given ID
    return df[df['id'] == exp_id].sort_values(by="label").group.values.astype(int)


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


def compute_metric_value(prob_file: str, y_true: np.ndarray, metric: str,
                         **kwargs):
    """
    Computes a leaf-level metric for a given experiment
    """

    # Load the probability matrix
    proba_mat = np.load(prob_file)

    # Compute the appropriate metric
    try:
        if metric == 'leaf_top1':
            val = top_k_accuracy(y_true, proba_mat, k=1)
        elif metric == 'leaf_top3':
            val = top_k_accuracy(y_true, proba_mat, k=3)
        else:
            # For the node_top1, we have to provide the random target vectors
            # so that we can compare our values relative to a random classifier
            n = len(y_true)
            y_pred = proba_mat.argmax(axis=1)
            true_val = np.sum(y_true == y_pred) / n

            nsamples = kwargs['rand_targets'].shape[0]
            rand_distn = np.array(
                [np.sum(kwargs['rand_targets'][i, :].flatten() == y_pred) / n
                 for i in range(nsamples)]
            )

            # Compute where in the distribution the true values lies in the
            # random distribution
            val = true_val / rand_distn.mean()
    except IndexError:
        print(prob_file)
        val = -1.

    return pd.DataFrame({'metric': [metric], 'value': [val]})


def compute_metrics(exp_df: pd.DataFrame, final_ids: np.ndarray,
                    query_str: str, metric: str, y_true: np.ndarray,
                    proba_path: str, **kwargs):
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
    if (exp_method == 'hci') and (metric != 'node_top1'):
        idx_vecs = [np.load(files[i]['idx']) for i in range(n)]
        target_vecs = [y_true[idx] for idx in idx_vecs]
    else:
        target_vecs = [np.copy(y_true) for _ in range(n)]

    # If the group DataFrame has been passed (i.e. we want to compute the
    # node level metrics) we need to get the corresponding label maps for
    # each of the experiment IDs and then we need to re-map the target vectors
    if ('group_df' in kwargs.keys()) and (metric == 'node_top1'):
        # Get all the label maps
        label_maps = [get_label_map(exp_id, kwargs['group_df'])
                      for exp_id in experiment_ids]

        # Update the values for the target vectors
        target_vecs = [remap_target(target_vecs[i], label_maps[i])
                       for i in range(n)]

    # Compute the relevant metric for all of the provided experiments
    if exp_method == 'hci':
        if 'leaf' in metric:
            res_dfs = [compute_metric_value(files[i]['hci'], target_vecs[i],
                                            metric) for i in range(n)]
        else:
            # To compute the node_top1 we need the true and random target
            # vectors
            rand_targets = [permute_node_target(target_vecs[i])
                            for i in range(n)]
            print(query_str)
            res_dfs = [
                compute_metric_value(files[i]['node'], target_vecs[i],
                                     metric, rand_targets=rand_targets[i])
                for i in range(n)
            ]
    else:
        res_dfs = [compute_metric_value(files[i]['f'], target_vecs[i], metric)
                   for i in range(n)]

    res_df = combine_dfs(res_dfs)
    res_df['id'] = experiment_ids
    return res_df, experiment_ids[0]


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


def get_boot_distns(exp_df: pd.DataFrame, group_df: pd.DataFrame,
                    final_ids: np.ndarray, query_str: str,
                    metrics: list, y_true: np.ndarray,
                    proba_path: str, nsamples: int):
    """
    Gets all the bootstrap distributions for a given query string and
    set of metrics
    """

    # If we are working with a flat classifier then we cannot compute
    # the node_top1 metric and thus need to remove it from the metrics
    # list
    if '(method == "f")' in query_str:
        new_metrics = metrics[:]
        new_metrics.remove('node_top1')
    else:
        new_metrics = metrics[:]

    # Go through each of the metrics and compute the relevant metric values
    # and generate the bootstrap distribution
    n = len(new_metrics)
    boot_dfs = [pd.DataFrame()] * n
    raw_dfs = [pd.DataFrame()] * n
    for i in range(n):
        if '(method == "f")' in query_str:
            raw_dfs[i], exp_id = compute_metrics(
                exp_df, final_ids, query_str, new_metrics[i], y_true, proba_path
            )
        else:
            # If HCI then we have to pass the group_df
            raw_dfs[i], exp_id = compute_metrics(
                exp_df, final_ids, query_str, new_metrics[i], y_true,
                proba_path, group_df=group_df
            )

        tmp_df = raw_dfs[i].copy()

        # Only use the first experiment ID for the bootstrap results to
        # simplify the experiment interface downstream
        tmp_df['id'] = exp_id
        boot_dfs[i] = get_boot_distn(tmp_df, nsamples)

    # Combine the DataFrames
    return {'boot_dfs': combine_dfs(boot_dfs), 'raw_dfs': combine_dfs(raw_dfs)}


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


def fix_entries(group_df: pd.DataFrame, nlabels: int, group_path: str):
    """
    Fixes the entries in the group DataFrame which may have been combined
    due to writing issues
    """

    sub_df = group_df.groupby('id', as_index=False)['label'].count()
    bad_ids = sub_df.loc[sub_df['label'] < nlabels, 'id']
    bad_idx = group_df[group_df['id'].isin(bad_ids)].index
    group_df.drop(index=bad_idx, inplace=True)

    # Save the result to disk so that this is no longer an issue
    group_df.to_csv(group_path, index=False)
    return group_df


def gen_boot_df(wd: str, exp_vars: list, metrics: list, nsamples=1000):
    """
    Generates the bootstrap DataFrame so we can visualize the results
    downstream
    """

    # Define the file paths for all the items we need
    label_path = os.path.join(wd, 'test_labels.csv')
    exp_path = os.path.join(wd, 'experiment_settings.csv')
    proba_path = os.path.join(wd, 'proba_pred')
    group_path = os.path.join(wd, 'group_res.csv')

    # Get the target vector to compute the metrics
    y_true = pd.read_csv(label_path, header=None).values.flatten()

    # We need the experiment settings to infer the unique experiments
    exp_df = pd.read_csv(exp_path)
    group_df = pd.read_csv(group_path, error_bad_lines=False)

    # Fix potential issues with the group DataFrame
    nlabels = len(np.unique(y_true))
    group_df = fix_entries(group_df, nlabels, group_path)

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
        res = p(delayed(get_boot_distns)(exp_df, group_df, final_ids,
                                         query_str, metrics, y_true,
                                         proba_path, nsamples)
                for query_str in exp_queries)

    n = len(res)
    boot_df = combine_dfs([res[i]['boot_dfs'] for i in range(n)])
    raw_df = combine_dfs([res[i]['raw_dfs'] for i in range(n)])
    return boot_df, raw_df

from core.hc import flat_model, hierarchical_model, hc_pred, spectral_model
from core.group_labels import group_labels
import hashlib
import pandas as pd
import numpy as np
from time import time
import h5py
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.utils import resample
import os


def gen_id(args: dict):
    """
    Generates the id(s) for our data using a SHA algorithm
    """

    # If we're working with a flat model then the k_vals = -1 since it does
    # not matter; otherwise, we need to map the values to strings
    if args["method"] == "f" or \
            (args['method'] == 'hci' and args['group_algo'] == 'comm'):
        k_vals = ["-1"]
    else:
        k_vals = list(map(str, args["k_vals"]))

    # Identify the non-string elements and map their values to strings
    # so we can more easily hash them
    non_str_keys = []
    for key in args.keys():
        # Check for the k_vals which we will ignore
        if key == "k_vals":
            continue

        if not isinstance(args[key], str):
            non_str_keys.append(key)

    # Convert the non-string keys to strings so that we can work with the
    # hashing algorithm
    n = len(k_vals)
    hashes = [""] * n
    str_keys = np.setdiff1d(list(args.keys()), non_str_keys + ["k_vals"])
    for i in range(n):
        # Define a temporary list to hold the values before we combine them
        # into a single string
        tmp_list = []

        # Convert and add the non-string values to our temp list
        for key in non_str_keys:
            tmp_list.append(str(args[key]))

        # Add the remaining keys
        for key in str_keys:
            if key == "wd":
                continue

            tmp_list.append(args[key])

        # Add the i-th k values
        tmp_list.append(k_vals[i])

        # Using the values from the temporary list we're going to combine
        # them into a single string and then generate the hash
        tmp_str = "_".join(tmp_list).encode("UTF-8")
        hashes[i] = hashlib.sha1(tmp_str).hexdigest()

    return hashes


def create_experiment_df(args: dict, ids: list):
    """
    Creates a DataFrame detailing the settings for a given experiment
    """

    # Get the number of meta-classes for a given run
    n = len(ids)
    if args["method"] == "f":
        k_vals = [-1]
    else:
        k_vals = args["k_vals"]

    # We need to iterate through the keys of the args dictionary and generate
    # a dictionary which will hold our experiment settings data
    data = {}
    for key in args.keys():
        # Exclude certain keys {k_vals, wd}
        if key in ["k_vals", "wd"]:
            continue

        # Otherwise add it to the data dict
        if key != "method":
            data[key] = np.repeat([args[key]], n)
        else:
            data["k"] = k_vals
            data["method"] = args['method']

    # Add the hash IDs
    data["id"] = ids

    # Build the final DataFrame
    return pd.DataFrame(data)


def build_group_df(ids: list, label_groups: np.ndarray) -> pd.DataFrame:
    """
    Builds the label grouping DataFrame so we can perform further analysis
    on the grouping at a later time
    """

    # Get the grouping vector for each label
    group_vect = label_groups.flatten()

    # Repeat the ID and label columns
    n_vals, nclasses = label_groups.shape
    id_vect = np.repeat(ids, nclasses)
    label_vect = np.tile(np.arange(nclasses), n_vals)

    return pd.DataFrame({"id": id_vect, "label": label_vect,
                         "group": group_vect})


def build_search_df(ids: list, k_res: list) -> pd.DataFrame:
    """
    Builds a DataFrame that stores the validation results
    """

    # Get the values from the various stored metrics
    n = len(k_res)
    acc_vals = [k_res[i]["acc"] for i in range(n)]
    auc_vals = [k_res[i]["auc"] for i in range(n)]
    train_times = [k_res[i]["train_time"] for i in range(n)]
    cluster_times = [k_res[i]["cluster_time"] for i in range(n)]

    # Build the DataFrame
    metrics = ["top1", "auc", "train_time", "cluster_time"]
    nmetrics = len(metrics)

    search_df = pd.DataFrame(
        data={"id": np.tile(ids, nmetrics),
              "metric": np.repeat(metrics, len(ids)),
              "value": np.concatenate([acc_vals, auc_vals, train_times,
                                       cluster_times])}
    )

    return search_df


def train_hc(X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray,
             y_val: np.ndarray, args: dict, rng: np.random.RandomState):
    """
    Searches over every label grouping and gets the best HC
    """

    # If we're doing community detection then we only have one inferred
    # community; otherwise we have to use the clustering-based approach
    if args['group_algo'] == 'comm':
        n = 1
    else:
        n = len(args['k_vals'])

    # First get the label groupings for each of the provided k_vals for
    k_res = [dict()] * n
    nlabels = len(np.unique(y_train))
    label_groups = np.empty(shape=(n, nlabels), dtype=np.int32)
    for i in range(n):
        if args['group_algo'] == 'comm':
            print('Inferring communities and fitting HC')
        else:
            print("Searching over k = {}".format(args["k_vals"][i]))

        # Infer the label groups
        start_time = time()
        label_groups[i, :] = group_labels(X_train, y_train, args["k_vals"][i],
                                          args["group_algo"], rng,
                                          args['niter'])
        cluster_time = time() - start_time

        # Fixing the label groups, get the best HC over the hyper-parameter
        # space
        k_res[i] = hierarchical_model(
            X_train, y_train, X_val, y_val, label_groups[i, :], rng,
            args["estimator"]
        )

        # Add the cluster time
        k_res[i]["cluster_time"] = cluster_time

    # Get the best model
    best_model = np.array([k_res[i]["acc"] for i in range(n)]).argmax()

    # Return all of the models so we can get validation results and the
    # best model so we can evaluate it out-of-sample
    return {"all_models": k_res, "final_model": k_res[best_model]["models"],
            "label_groups": label_groups, "best_model": best_model}


def train_spectral(X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray,
                   y_val: np.ndarray, args: dict, rng: np.random.RandomState):
    """
    Trains the spectral model across all values of k
    """

    # First get the label groupings for each of the provided k_vals for
    n = len(args['k_vals'])
    k_res = [dict()] * n
    nlabels = len(np.unique(y_train))
    label_groups = np.empty(shape=(n, nlabels), dtype=np.int32)
    A = np.empty(shape=(nlabels, nlabels))
    for i in range(n):
        print('Searching over k = {}'.format(args['k_vals'][i]))

        # For the first run we need to compute the flat model and generate
        # the affinity matrix for spectral clustering; for the remaining
        # runs we'll just use the affinity matrix we computed
        if i == 0:
            k_res[i], A, tmp_groups = spectral_model(
                X_train, y_train, X_val, y_val, args['k_vals'][i], rng,
                args['estimator'],
            )
        else:
            k_res[i], tmp_groups = spectral_model(
                X_train, y_train, X_val, y_val, args['k_vals'][i], rng,
                args['estimator'], affinity_mat=A
            )

        # Add the label groups to the matrix
        label_groups[i, :] = tmp_groups

    # Get the best model
    best_model = np.array([k_res[i]["acc"] for i in range(n)]).argmax()

    # Return all of the models so we can get validation results and the
    # best model so we can evaluate it out-of-sample
    return {"all_models": k_res, "final_model": k_res[best_model]["models"],
            "label_groups": label_groups, "best_model": best_model}


def prep_data(datapath: str, savepath: str, rng: np.random.RandomState,
              downsample_prop=0.50) -> dict:
    """
    Prepares the data by forming the train-validation-test split,
    saves y_test to disk, and bootstraps the training data
    """

    # Load the data from disk
    f = h5py.File(datapath, "r")
    X = np.array(f["X"])
    y = np.array(f["y"]).flatten()
    f.close()

    # If we need to, we will down-sample the data proportional to y so
    # that we can decrease computation time
    if downsample_prop > 0.:
        # If the proportion is more than 0, then we will downsample the data
        # accordingly
        splitter = StratifiedShuffleSplit(
            n_splits=1, test_size=(1 - downsample_prop), random_state=17
        )

        idx = splitter.split(X, y)
        idx = [val for val in idx]
        new_idx = idx[0][0]
        X = X[new_idx]
        y = y[new_idx]

    # First generate the train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=17
    )

    # Now generate the train-validation split
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=17
    )

    # Save y_test to disk so that we can work with it later
    if not os.path.exists(savepath):
        pd.Series(y_test).to_csv(savepath, index=False, header=False)

    # Finally we will bootstrap the training data so that we can have a
    # distribution of estimator performance
    X_train, y_train = resample(X_train,  y_train, random_state=rng)

    return {"X_train": X_train, "y_train": y_train, "X_val": X_val,
            "y_val": y_val, "X_test": X_test}


def fit_fc(data_dict: dict, rng: np.random.RandomState, args: dict):
    """
    Fits and gets the out-of-sample predictions for a FC
    """

    # Train/predict the FC
    X_train = data_dict["X_train"]
    y_train = data_dict["y_train"]
    X_test = data_dict["X_test"]
    res = flat_model(X_train, y_train, X_test, rng, args['estimator'])
    # res = flat_model(X_train, y_train, X_test, rng, args["estimator"],
    #                  args["niter"])

    # Generate the run ID
    run_id = gen_id(args)

    # Prepare the results DataFrame
    fc_df = pd.DataFrame({"id": run_id, "metric": ["train_time"],
                          "value": [res["train_time"]]})

    # Return the preliminary results DataFrame and the out-of-sample pred
    return {"proba_pred": res["proba_pred"], "fc_df": fc_df, 'ids': run_id}


def get_hc_res(k_res: dict, X_test: np.ndarray, args: dict):
    """
    Generates the test results and DataFrames for a hierarchical classifier
    """

    # Get the best model
    best_model = k_res['best_model']
    res = hc_pred(k_res['final_model'], X_test,
                  k_res['label_groups'][best_model, :])

    # # If we're working with the community detection algorithm, we need
    # # to add information about how many groups were inferred (the dict will
    # # be updated so no need to pass it as a result)
    # if args['group_algo'] == 'comm':
    #     args['k_vals'] = [len(np.unique(k_res['label_groups'][0, :]))]

    # Generate the run ID(s)
    ids = gen_id(args)

    # Build the group and validation search DataFrames
    group_df = build_group_df(ids, k_res['label_groups'])
    search_df = build_search_df(ids, k_res['all_models'])

    return {"res": res, "group_df": group_df, "search_df": search_df,
            "best_model": best_model, 'ids': ids}


def fit_hc(data_dict: dict, rng: np.random.RandomState, args: dict):
    """
    Fits an HC, gets the test predictions, and generates results DataFrames
    """

    # First determine the best model across all meta-class groups
    X_train = data_dict["X_train"]
    y_train = data_dict["y_train"]
    X_val = data_dict['X_val']
    y_val = data_dict['y_val']
    X_test = data_dict['X_test']
    k_res = train_hc(X_train, y_train, X_val, y_val, args, rng)

    # Get the results from the experiments
    return get_hc_res(k_res, X_test, args)


def fit_spectral(data_dict: dict, rng: np.random.RandomState, args: dict):
    """
    Fits the spectral clustering based benchmark and generates the DataFrames
    """

    # Determine the best model across all meta-class groups
    X_train = data_dict["X_train"]
    y_train = data_dict["y_train"]
    X_val = data_dict['X_val']
    y_val = data_dict['y_val']
    X_test = data_dict['X_test']
    k_res = train_spectral(X_train, y_train, X_val, y_val, args, rng)

    # Get the test results from the spectral HC
    return get_hc_res(k_res, X_test, args)


def save_fc_res(fc_res: dict, wd: str):
    """
    Saves the results from the FC to disk
    """

    # Get the final elements from the FC model
    proba_pred = fc_res["proba_pred"]
    fc_df = fc_res["fc_df"]
    run_id = fc_res['ids']

    # Save the probability prediction to disk
    file = "f_" + run_id[0] + ".npy"
    savepath = os.path.join(wd, "proba_pred", file)
    np.save(savepath, proba_pred)

    # Save the FC preliminary results to disk
    file = os.path.join(wd, "fc_prelim_res.csv")
    if os.path.exists(file):
        fc_df.to_csv(file, mode="a", header=False, index=False)
    else:
        fc_df.to_csv(file, index=False)

    return None


def save_hc_res(hci_res: dict, wd: str):
    """
    Saves the results from the HC to disk
    """

    # Save the validation and grouping results to disk
    search_df = hci_res["search_df"]
    group_df = hci_res["group_df"]
    ids = hci_res['ids']

    search_path = os.path.join(wd, "search_res.csv")
    group_path = os.path.join(wd, "group_res.csv")

    if os.path.exists(search_path):
        search_df.to_csv(search_path, mode="a", header=False, index=False)
    else:
        search_df.to_csv(search_path, index=False)

    if os.path.exists(group_path):
        group_df.to_csv(group_path, mode="a", header=False, index=False)
    else:
        group_df.to_csv(group_path, index=False)

    # Save the probability predictions to disk
    best_id = ids[hci_res['best_model']]
    root_path = "root_" + best_id + ".npy"
    root_path = os.path.join(wd, "proba_pred", root_path)
    root_proba_pred = hci_res['res']["root_proba_pred"]
    np.save(root_path, root_proba_pred)

    # node_path = "node_" + best_id + ".npy"
    # node_path = os.path.join(wd, "proba_pred", node_path)
    # node_proba_preds = hci_res['res']["node_proba_preds"]
    # np.save(node_path, node_proba_preds)

    full_path = "hc_" + best_id + ".npy"
    full_path = os.path.join(wd, "proba_pred", full_path)
    proba_pred = hci_res['res']["proba_pred"]
    np.save(full_path, proba_pred)

    # Save the good indices to disk so we can correct y_test later
    idx_path = "idx_" + best_id + ".npy"
    idx_path = os.path.join(wd, "proba_pred", idx_path)
    np.save(idx_path, hci_res['res']['good_idx'])

    return None


def run_model(args: dict, datapath: str):
    """
    Runs the experiment by fitting the model to data and saves results to disk
    for analysis at a later time

    Notes
    At minimum, we assume that args contains the following keys:

    run_num: the run number for the particular experiment
    method: the algorithmic method to employ for the experiment
    group_algo: grouping algorithm to use to find meta-classes
    estimator: the ML estimator to use for the experiment
    niter: number of values to search over during hyper-parameter search
    wd: working directory to save results to disk

    There could be others depending on the experiment settings for a particular
    data set
    """

    # First we need to read in and prepare the data
    wd = args["wd"]
    savepath = os.path.join(wd, "test_labels.csv")
    rng = np.random.RandomState(args["run_num"])
    if "downsample_prop" in args.keys():
        data_dict = prep_data(datapath, savepath, rng, args["downsample_prop"])
    else:
        data_dict = prep_data(datapath, savepath, rng)

    # Create the proba_pred directory if it does not already exist
    if not os.path.exists(os.path.join(args['wd'], 'proba_pred')):
        os.mkdir(os.path.join(args['wd'], 'proba_pred'))

    # Next we need to fit the model given the appropriate method
    if args["method"] == "f":
        res = fit_fc(data_dict, rng, args)
        save_fc_res(res, wd)

    else:
        if args['group_algo'] in ['kmm', 'lp', 'comm']:
            res = fit_hc(data_dict, rng, args)
        elif args['group_algo'] == 'spectral':
            res = fit_spectral(data_dict, rng, args)
        else:
            res = {}

        save_hc_res(res, wd)

    # Finally we need to record the various experimental settings for
    # further analysis at a later time
    exp_df = create_experiment_df(args, res['ids'])
    exp_path = os.path.join(wd, "experiment_settings.csv")
    if os.path.exists(exp_path):
        exp_df.to_csv(exp_path, mode="a", header=False, index=False)
    else:
        exp_df.to_csv(exp_path, index=False)

    # Return Nothing; we have already saved all that we need to disk
    return None

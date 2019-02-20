import pandas as pd
import os
import argparse


def find_bad_files(exp_ids: list, basefiles: list):
    """
    Finds and deletes the bad probability files
    """

    # Go through each of the files and experiment IDs and determine which
    # items need to be deleted
    bad_files = []
    for exp_id in exp_ids:
        for file in basefiles:
            if exp_id in file:
                bad_files.append(file)

    return bad_files


def delete_experiments():
    """
    Deletes the experiments from the appropriate sources given the
    command line arguments
    """

    # Parse the command line
    parser = argparse.ArgumentParser()
    parser.add_argument("--wd", type=str)
    parser.add_argument("--query", type=str)
    args = vars(parser.parse_args())

    # Get the relevant DataFrames
    exp_df = pd.read_csv(os.path.join(args['wd'], 'experiment_settings.csv'))
    search_df = pd.read_csv(os.path.join(args['wd'], 'search_res.csv'))
    group_df = pd.read_csv(os.path.join(args['wd'], 'group_res.csv'))

    # Infer the appropriate experiment IDs given the query
    exp_ids = exp_df.query(args['query'], engine='python').id.tolist()

    # Remove exp_ids from the exp_df, search_df, and group_df
    exp_df = exp_df[~exp_df['id'].isin(exp_ids)]
    search_df = search_df[~search_df['id'].isin(exp_ids)]
    group_df = group_df[~group_df['id'].isin(exp_ids)]

    # Save the updated DataFrames to disk
    exp_df.to_csv(os.path.join(args['wd'], 'experiment_settings.csv'),
                  index=False)
    search_df.to_csv(os.path.join(args['wd'], 'search_res.csv'), index=False)
    group_df.to_csv(os.path.join(args['wd'], 'group_res.csv'), index=False)

    # Finally we need to find all of the probability prediction files that
    # correspond with the experiment IDs and we need to delete them
    proba_path = os.path.join(args['wd'], 'proba_pred')
    basefiles = os.listdir(proba_path)
    bad_files = find_bad_files(exp_ids, basefiles)

    # Add the full file path to each of the bad files
    bad_files = [os.path.join(proba_path, file) for file in bad_files]

    # Go through each of the bad files and delete them
    [os.remove(file) for file in bad_files]
    return None


if __name__ == '__main__':
    delete_experiments()

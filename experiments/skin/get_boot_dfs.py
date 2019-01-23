from core.compare_experiments import compare_experiments
import argparse
from os import path


def main():
    # Parse the command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('exp_vars', nargs='*', type=str)
    parser.add_argument('wd', type=str, nargs='?',
                        default='/pool001/zblanks/label_reduction_data/skin')
    parser.add_argument('--bootstrap_samples', type=int, nargs='?',
                        default=1000)
    args = vars(parser.parse_args())

    # Get the expected data paths
    exp_path = path.join(args['wd'], 'experiment_settings.csv')
    group_path = path.join(args['wd'], 'group_res.csv')
    proba_path = path.join(args['wd'], 'proba_pred')
    label_path = path.join(args['wd'], 'test_labels.csv')

    # Perform the bootstrap analysis
    boot_df, pair_df = compare_experiments(exp_path, group_path, proba_path,
                                           label_path, args['exp_vars'],
                                           args['bootstrap_samples'])

    # Save the DataFrames to disk
    boot_df.to_csv(path.join(args['wd'], 'boot_res.csv'), index=False)
    pair_df.to_csv(path.join(args['wd'], 'exp_pairs.csv'), index=False)


if __name__ == '__main__':
    main()

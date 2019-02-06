from core.gen_boot_distns import gen_boot_df
import argparse
from os import path


def main():
    # Parse the command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_vars', nargs='*', type=str)
    parser.add_argument('--wd', type=str)
    parser.add_argument('--metrics', nargs='*', type=str,
                        default=['leaf_top1', 'leaf_top3'])
    parser.add_argument('--nsamples', type=int, nargs='?',
                        default=1000)
    args = vars(parser.parse_args())

    # Get the expected data paths
    exp_path = path.join(args['wd'], 'experiment_settings.csv')
    proba_path = path.join(args['wd'], 'proba_pred')
    label_path = path.join(args['wd'], 'test_labels.csv')

    # # Perform the bootstrap analysis
    boot_df, raw_df = gen_boot_df(exp_path, proba_path, label_path,
                                  args['exp_vars'], args['metrics'],
                                  args['nsamples'])

    # Save the DataFrames to disk
    boot_df.to_csv(path.join(args['wd'], 'boot_res.csv'), index=False)
    raw_df.to_csv(path.join(args['wd'], 'raw_res.csv'), index=False)


if __name__ == '__main__':
    main()

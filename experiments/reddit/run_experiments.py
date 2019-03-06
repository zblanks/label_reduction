from core.run_model import run_model
import argparse
from os import path


def main():
    # Parse the command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("run_num", type=int)
    parser.add_argument("method", type=str)
    parser.add_argument("group_algo", type=str)
    parser.add_argument("estimator", type=str)
    parser.add_argument("--downsample_prop", type=float, nargs="?",
                        default=0.2)
    parser.add_argument("--niter", type=int, nargs="?", default=10)
    parser.add_argument("--k_vals", type=int, nargs="*",
                        default=list(range(10, 110, 10)))
    parser.add_argument("--wd", type=str, nargs="?",
                        default="/pool001/zblanks/label_reduction_data/reddit")
    parser.add_argument('--metrics', type=str, nargs='*',
                        default='')
    parser.add_argument('--features', type=str, nargs='?', default='lda')
    args = vars(parser.parse_args())

    # Get the data path depending on whether we're using meta-data or not
    datapath = path.join(args['wd'], 'data.h5')

    # Run the model
    run_model(args, datapath)


if __name__ == '__main__':
    main()

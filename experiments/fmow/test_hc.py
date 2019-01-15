from core.run_model import run_model
import argparse


def main():
    # Parse the command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("run_num", type=int)
    parser.add_argument("method", type=str)
    parser.add_argument("group_algo", type=str)
    parser.add_argument("estimator", type=str)
    parser.add_argument("use_meta", type=int)
    parser.add_argument("--niter", type=int, nargs="?", default=10)
    parser.add_argument("--k_vals", type=int, nargs="*",
                        default=list(range(2, 62)))
    parser.add_argument("--wd", type=str, nargs="?",
                        default="/pool001/zblanks/label_reduction_data/fmow")
    args = vars(parser.parse_args())

    # Run the model
    run_model(args)


if __name__ == '__main__':
    main()

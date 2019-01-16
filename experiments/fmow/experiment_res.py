from core.compare_experiments import compare_experiments
import os


def main():
    wd = "/pool001/zblanks/label_reduction_data/fmow"
    exp_path = os.path.join(wd, "experiment_settings.csv")
    group_path = os.path.join(wd, "group_res.csv")
    proba_path = os.path.join(wd, "proba_pred")
    label_path = os.path.join(wd, "test_labels.csv")
    exp_vars = ["method"]
    bootstrap_samples = 1000

    # Get the experiment(s) results
    boot_df, pair_df = compare_experiments(exp_path, group_path, proba_path,
                                           label_path, exp_vars,
                                           bootstrap_samples)

    # Save to disk to check if it worked
    boot_df.to_csv(os.path.join(wd, "boot_res.csv"))
    pair_df.to_csv(os.path.join(wd, "pair_map.csv"))


if __name__ == '__main__':
    main()

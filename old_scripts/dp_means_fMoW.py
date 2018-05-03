from old_scripts.dp_means import parallel_dp_means
import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler


if __name__ == '__main__':
    np.random.seed(17)
    wd = '/pool001/zblanks/'

    # Get our data
    meta = pd.read_csv(os.path.join(wd, 'side_info/metadata_train.csv'))
    df = pd.read_csv(os.path.join(wd, 'experiment/train_conv_features.csv'),
                     header=None)
    idx = np.random.choice([True, False], size=meta.shape[0], p=[0.75, 0.25])
    meta = meta.loc[idx, :]
    df.loc[:, 'label'] = meta.target.tolist()
    targets = [0, 44, 61]
    sub_df = df.loc[(df.label.isin(targets)), :]
    sub_df = sub_df.drop(['label'], axis=1)
    scaler = StandardScaler()
    sub_df = scaler.fit_transform(sub_df)

    # Search over various eta values for DP-means clustering
    eta_vals = np.arange(start=5, stop=50)
    dp_res = parallel_dp_means(X=sub_df, eta_vals=eta_vals)

    # Save our results to disk
    np.savetxt(os.path.join(wd, 'experiment/dp_means_results.csv'),
               X=dp_res, delimiter=',')

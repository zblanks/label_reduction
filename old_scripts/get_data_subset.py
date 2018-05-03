import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import sys
import os


if __name__ == '__main__':
    data_loc = sys.argv[1]
    save_loc = sys.argv[2]
    labels = sys.argv[3].split(',')

    # Convert our labels to be numeric
    labels = list(map(int, labels))

    # Get the appropriate data files
    train_meta = pd.read_csv(os.path.join(data_loc, 'metadata_train.csv'))
    idx = np.loadtxt(os.path.join(data_loc, 'idx.csv'))
    df = pd.read_csv(os.path.join(data_loc, 'train_conv_features.csv'),
                     header=None)

    # Subset the image data on the appropriate indexes
    train_meta = train_meta.loc[idx.astype(bool), :]
    train_meta.reset_index(drop=True, inplace=True)
    df.loc[:, 'target'] = train_meta.target
    df2 = df.loc[df.target.isin(labels), :]
    targets = df2.target.as_matrix()
    df2.drop(['target'], axis=1, inplace=True)

    # Standardize the data
    scaler = StandardScaler()
    df2 = scaler.fit_transform(df2)

    # Get our data back into DataFrame form
    df2 = pd.DataFrame(df2)
    df2.loc[:, 'target'] = targets

    # Save the data to disk
    df2.to_csv(save_loc, index=False)

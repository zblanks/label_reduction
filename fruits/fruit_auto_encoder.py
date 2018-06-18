import sys
sys.path.insert(0, '/home/zblanks/label_reduction')
from encode_data import EncodeData
import argparse
import pandas as pd
import os


if __name__ == '__main__':
    # Get the script arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('data_path', help='Location for data', type=str)
    parser.add_argument('save_path', help='Location to save auto-encoder',
                        type=str)
    args = vars(parser.parse_args())

    # Get the auto-encoder object
    encoder = EncodeData(data_path=args['data_path'],
                         save_path=args['save_path'])
    encoder.encode()

    # Save the encoded data and the training history
    encoded_df = encoder.encoded_data
    encoded_df.to_csv(os.path.join(args['save_path'], 'encoded_data.csv'),
                      index=False)
    error_df = pd.DataFrame(encoder.train_history, columns=['train_loss',
                                                            'val_loss'])
    error_df.to_csv(os.path.join(args['save_path'], 'error.csv'), index=False)

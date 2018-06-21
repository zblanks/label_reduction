import sys
sys.path.insert(0, '/home/zblanks/label_reduction')
from encode_data import EncodeData
import argparse
import os


if __name__ == '__main__':
    # Get the script arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('data_path', help='Location for data', type=str)
    parser.add_argument('save_path', help='Location to save auto-encoder',
                        type=str)
    parser.add_argument('batch_size', nargs='?', type=int, default=4096)
    args = vars(parser.parse_args())

    # Get the auto-encoder object
    encoder = EncodeData(data_path=args['data_path'],
                         save_path=args['save_path'],
                         batch_size=args['batch_size'])
    encoder.encode()

    # Save the encoded data and the training history
    encoded_df = encoder.encoded_data
    encoded_df.to_csv(os.path.join(args['save_path'], 'encoded_data.csv'),
                      index=False)

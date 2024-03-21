from cdvae.common.data_utils import preprocess
import pandas as pd
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.core.structure import Structure
from pymatgen.core.lattice import Lattice
from pymatgen.analysis.diffraction.xrd import XRDCalculator, WAVELENGTHS
import torch
from tqdm import tqdm
import argparse
import os

def split_data(args):
    # list the directory
    files = os.listdir(args.init_data_dir)
    # filter the files
    files = [f for f in files if f.endswith('.csv')]
    print(f'Found {len(files)} files: {files}')
    assert len(files) == 3
    all_data = list()
    for file in files:
        print(f'Processing {file}')
        # Load the data
        curr_data = pd.read_pickle(os.path.join(args.init_data_dir, file))
        print(curr_data.head())
        all_data.append(curr_data)
        print(len(curr_data))
    all_data = pd.concat(all_data, axis=0, ignore_index=True)
    print(all_data.head())
    print(len(all_data))
    print(all_data.columns)

    new_train_len = int(args.train_percent * len(all_data))
    new_test_len = int(args.test_percent * len(all_data))

    train_data = all_data.iloc[0:new_train_len]
    test_data = all_data.iloc[new_train_len:new_train_len+new_test_len]
    val_data = all_data.iloc[new_train_len+new_test_len:]

    print(train_data.head(), len(train_data))
    print(val_data.head(), len(val_data))
    print(test_data.head(), len(test_data))

    train_data.to_pickle(os.path.join(args.new_data_dir, 'train.csv'))
    val_data.to_pickle(os.path.join(args.new_data_dir, 'val.csv'))
    test_data.to_pickle(os.path.join(args.new_data_dir, 'test.csv'))

    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Split data')
    parser.add_argument(
        '--init_data_dir',
        default = '/home/gabeguo/cdvae_xrd/data/mp_20',
        type=str,
        help='path to input CIF+XRD files'
    )
    parser.add_argument(
        '--new_data_dir',
        default = '/home/gabeguo/cdvae_xrd/data/mp_20_resplit',
        type=str,
        help='path to save re-split data'
    )
    parser.add_argument(
        '--train_percent',
        type=float,
        default=0.9
    )
    parser.add_argument(
        '--test_percent',
        type=float,
        default=0.025
    )
    args = parser.parse_args()
    os.makedirs(args.new_data_dir, exist_ok=True)
    split_data(args)

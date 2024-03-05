import pandas as pd
from pymatgen.io.cif import CifWriter
from gen_xrd import create_xrd_tensor
import argparse
import re
import os
from tqdm import tqdm
import random

MAX_ATOMS = 24

def good_file_format(struct_filename):
    format = r'^mp-\d+_Structure[.]pickle$'
    return re.match(format, struct_filename)

def struct2xrd_filename(struct_filename):
    assert good_file_format(struct_filename)
    return struct_filename.replace('Structure', 'XRD')

def extract_mp_from_filename(struct_filename):
    assert good_file_format(struct_filename)
    return struct_filename.split('_')[0]

def save_df_with_indices(mpids, cifs, xrds, indices, name):
    assert len(mpids) == len(cifs)
    assert len(cifs) == len(xrds)

    mpids = [mpids[i] for i in range(len(mpids)) if i in indices]
    cifs = [cifs[i] for i in range(len(cifs)) if i in indices]
    xrds = [xrds[i] for i in range(len(xrds)) if i in indices]

    the_df = pd.DataFrame(columns=['material_id', 'cif', 'xrd'], dtype=object)
    the_df['material_id'] = mpids
    the_df['cif'] = cifs
    the_df['xrd'] = xrds

    os.makedirs(args.save_filepath, exist_ok=True)
    the_df.to_pickle(os.path.join(args.save_filepath, f'{name}.csv'))

    return

def main(args):
    random.seed(args.seed)

    noshows = list()
    too_big = list()

    cifs = list()
    xrds = list()
    mpids = list()
    for struct_file in tqdm(os.listdir(args.struct_dir_pickled)):
        the_mpid = extract_mp_from_filename(struct_filename=struct_file)
        struct_filepath = os.path.join(args.struct_dir_pickled, struct_file)
        xrd_filepath = os.path.join(args.xrd_dir_pickled, struct2xrd_filename(struct_file))
        if not (os.path.exists(struct_filepath) and os.path.exists(xrd_filepath)):
            noshows.append(the_mpid)
            continue
        the_structure = pd.read_pickle(struct_filepath)
        if the_structure.num_sites >= MAX_ATOMS:
            too_big.append(the_mpid)
            continue
        the_xrd = create_xrd_tensor(args, pd.read_pickle(xrd_filepath))
        cif_writer = CifWriter(the_structure)
        cif_string = cif_writer.__str__()

        cifs.append(cif_string)
        xrds.append(the_xrd)
        mpids.append(the_mpid)

    indices = list(range(len(mpids)))
    random.shuffle(indices)

    assert args.train_ratio + args.val_ratio < 1
    train_end = int(len(mpids) * args.train_ratio)
    val_end = train_end + int(len(mpids) * args.val_ratio)
    
    train_indices = indices[:train_end]
    val_indices = indices[train_end:val_end]
    test_indices = indices[val_end:]

    for curr_indices, curr_name in zip([train_indices, val_indices, test_indices], ['train', 'val', 'test']):
        save_df_with_indices(mpids=mpids, cifs=cifs, xrds=xrds, indices=curr_indices, name=curr_name)

    print('noshows:', len(noshows), ' : ', noshows)
    print(f'too big: {len(too_big)} / {len(mpids) + len(too_big)}')

    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate XRD patterns from CIF descriptions')
    parser.add_argument(
        '--max_theta',
        default = 180,
        type=int,
    )
    parser.add_argument(
        '--min_theta',
        default = 0,
        type=int,
    )
    parser.add_argument(
        '--xrd_vector_dim',
        default = 512,
        type=int,
    )
    parser.add_argument(
        '--save_filepath',
        default='/home/gabeguo/cdvae_xrd/data/mp_trigonal',
        type=str,
    )
    parser.add_argument(
        '--struct_dir_pickled',
        default='/home/gabeguo/mp_dataset/updated_crystallography_data/pickled_positions/Trigonal',
        type=str
    )
    parser.add_argument(
        '--xrd_dir_pickled',
        default='/home/gabeguo/mp_dataset/updated_crystallography_data/pickled_xrds/Trigonal',
        type=str
    )
    parser.add_argument(
        '--seed',
        default=0,
        type=str
    )
    parser.add_argument(
        '--train_ratio',
        default=0.8,
        type=float
    )
    parser.add_argument(
        '--val_ratio',
        default=0.1,
        type=float
    )
    args = parser.parse_args()
    main(args)


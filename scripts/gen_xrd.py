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
import numpy as np
from pymatgen.io.cif import CifParser

def create_xrd_tensor(args, pattern):
    wavelength = WAVELENGTHS[args.wave_source]

    # takes in a pattern (in 2theta space) and converts it to a tensor in Q space
    peak_data = torch.zeros(args.xrd_vector_dim) # Q space
    peak_locations_2_theta = pattern.x.tolist()
    # convert 2theta to theta
    peak_locations_theta = [0.5 * theta for theta in peak_locations_2_theta]
    # convert theta to Q
    peak_locations_Q = [4 * np.pi * np.sin(np.radians(theta)) / wavelength for theta in peak_locations_theta]
    peak_values = pattern.y.tolist()

    # convert min and max theta to Q
    min_Q = 4 * np.pi * np.sin(np.radians(args.min_theta / 2)) / wavelength
    max_Q = 4 * np.pi * np.sin(np.radians(args.max_theta / 2)) / wavelength
    for i2 in range(len(peak_locations_Q)):
        q = peak_locations_Q[i2]
        height = peak_values[i2] / 100
        scaled_location = int(args.xrd_vector_dim * (q - min_Q) / (max_Q - min_Q))
        peak_data[scaled_location] = max(peak_data[scaled_location], height) # just in case really close

    return peak_data


def gen_xrd(args):
    data_dir = args.data_dir
    # list the directory
    files = os.listdir(data_dir)
    # filter the files
    files = [f for f in files if f.endswith('.csv')]
    print(f'Found {len(files)} files')
    for file in files:
        print(f'Processing {file}')
        # Load the data
        data = pd.read_csv(os.path.join(data_dir, file))
        # extract CIF
        cifs = data['cif'].values
        # iterate over the CIFs
        xrd_tensor_list = []
        for cif in tqdm(cifs, desc=f'Generating XRDs for file {file}'):
            # Create the structure

            parser = CifParser.from_str(cif)
            structure = parser.get_structures()[0]

            # crystal = Structure.from_str(cif, fmt='cif')
            # crystal = crystal.get_primitive_structure()
            # structure = Structure(
            #     lattice=Lattice.from_parameters(*crystal.lattice.parameters),
            #     species=crystal.species,
            #     coords=crystal.frac_coords,
            #     coords_are_cartesian=False,
            # )
            
            # important to use the conventional structure to ensure
            # that peaks are labelled with the conventional Miller indices
            sga = SpacegroupAnalyzer(structure)
            structure = sga.get_conventional_standard_structure()

            # wavelength
            curr_wavelength = WAVELENGTHS[args.wave_source]
            # Create the XRD calculator
            xrd_calc = XRDCalculator(wavelength=curr_wavelength)
            # Calculate the XRD pattern
            pattern = xrd_calc.get_pattern(structure)
            # Create the XRD tensor
            xrd_tensor = create_xrd_tensor(args, pattern)
            xrd_tensor_list.append(xrd_tensor)
        # Save the XRD tensor
        xrd_df = pd.DataFrame(columns=['xrd'], dtype=object)
        xrd_df['xrd'] = xrd_tensor_list
        data = pd.concat([data, xrd_df], axis=1)
        data.to_pickle(os.path.join(args.save_dir, file))
        # torch.save(xrd_tensor, os.path.join(args.save_dir, file.replace('.csv', '.pt')))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate XRD patterns from CIF descriptions')
    parser.add_argument(
        '--data_dir',
        default = '/home/gabeguo/cdvae_xrd/data/mp_20_oldSplit/no_xrd',
        type=str,
        help='path to input CIF files'
    )
    parser.add_argument(
        '--save_dir',
        default = '/home/gabeguo/cdvae_xrd/data/mp_20_oldSplit',
        type=str,
        help='path to save XRD patterns'
    )
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
        '--wave_source', 
        type=str, 
        default='CuKa',                
        help='What is the wave source?'
    )
    parser.add_argument(
        '--xrd_vector_dim',
        default = 4096,
        type=int,
    )
    args = parser.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    gen_xrd(args)

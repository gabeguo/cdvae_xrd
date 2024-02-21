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

def create_xrd_tensor(args, pattern):
    peak_data = torch.zeros(args.xrd_vector_dim)

    peak_locations = pattern.x.tolist()
    peak_values = pattern.y.tolist()

    for i2 in range(len(peak_locations)):
        theta = peak_locations[i2]
        height = peak_values[i2] / 100
        scaled_location = int(args.xrd_vector_dim * theta / (args.max_theta - args.min_theta))
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
            crystal = Structure.from_str(cif, fmt='cif')
            crystal = crystal.get_primitive_structure()
            structure = Structure(
                lattice=Lattice.from_parameters(*crystal.lattice.parameters),
                species=crystal.species,
                coords=crystal.frac_coords,
                coords_are_cartesian=False,
            )
            
            ### @Gabe Issue when these are uncommented ###
            # sga = SpacegroupAnalyzer(structure)
            # conventional_structure = sga.get_conventional_standard_structure()
            ### @Gabe Issue when these are uncommented ###


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
        xrd_tensor = torch.stack(xrd_tensor_list)
        torch.save(xrd_tensor, os.path.join(args.save_dir, file.replace('.csv', '.pt')))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate XRD patterns from CIF descriptions')
    parser.add_argument(
        '--data_dir',
        default = '/home/tsaidi/Research/cdvae_xrd/data/perov_5',
        type=str,
        help='path to input CIF files'
    )
    parser.add_argument(
        '--save_dir',
        default = '/home/tsaidi/Research/cdvae_xrd/data/perov_5/xrd',
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
        default = 512,
        type=int,
    )
    args = parser.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    gen_xrd(args)

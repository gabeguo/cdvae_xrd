import torch
import numpy as np
import argparse
import os
import matplotlib.pyplot as plt
from pymatgen.analysis.diffraction.xrd import XRDCalculator, WAVELENGTHS
from pymatgen.core.structure import Structure
from scripts.gen_xrd import create_xrd_tensor

def get_field_value(all_lines, desired_start):
    for i, the_line in enumerate(all_lines):
        if the_line.startswith(desired_start):
            split_line = the_line.split()
            if len(split_line) > 1:
                return split_line[-1]
            else:
                all_lines[i+1]
    return None

def main(args):
    sim_wavelength = WAVELENGTHS[args.desired_wavelength]
    assert os.path.exists(args.filepath)
    with open(args.filepath, 'r') as fin:
        all_lines = [x.strip() for x in fin.readlines()]
        xrd_loop_start_idx = all_lines.index('loop_')
        assert xrd_loop_start_idx >= 0
        expected_fields = ['_pd_meas_intensity_total',
                                    '_pd_proc_ls_weight',
                                    '_pd_proc_intensity_bkg_calc',
                                    '_pd_calc_intensity_total']
        for idx in range(len(expected_fields)):
            assert expected_fields[idx] == all_lines[xrd_loop_start_idx + (idx + 1)]
        for idx in range(xrd_loop_start_idx + (len(expected_fields)+1), len(all_lines)):
            if all_lines[idx].strip() != '':
                start_idx = idx
                break
        
        end_idx = all_lines.index('', start_idx)
        assert end_idx > start_idx

        _2theta_min_deg = float(get_field_value(all_lines, '_pd_meas_2theta_range_min'))
        _2theta_max_deg = float(get_field_value(all_lines, '_pd_meas_2theta_range_max'))
        _exp_wavelength = float(get_field_value(all_lines, '_diffrn_radiation_wavelength'))

        print(_2theta_min_deg)
        print(_2theta_max_deg)
        print(_exp_wavelength)

        theta_min_rad = (_2theta_min_deg / 2) * np.pi / 180
        theta_max_rad = (_2theta_max_deg / 2) * np.pi / 180

        xrd_intensities = all_lines[start_idx:end_idx]
        the_thetas_rad = np.linspace(theta_min_rad, theta_max_rad, len(xrd_intensities))
        print('theta range (rad):', np.min(the_thetas_rad), np.max(the_thetas_rad))
        converted_2thetas = 2 * np.arcsin(np.sin(the_thetas_rad) * sim_wavelength / _exp_wavelength) * 180 / np.pi
        print('2theta range (deg):', np.nanmin(converted_2thetas), np.nanmax(converted_2thetas))
        
        xrd_tensor = torch.zeros(args.xrd_vector_dim)

        _2thetas = np.linspace(args.min_2theta, args.max_2theta, args.xrd_vector_dim)

        for i in range(len(converted_2thetas)):
            curr_2theta = converted_2thetas[i]
            xrd_info = xrd_intensities[i]
            xrd_info = xrd_info.split()
            assert len(xrd_info) == len(expected_fields)
            intensity_mean, intensity_std = xrd_info[0].split('(')
            intensity_std = float(intensity_std[:-1])
            intensity_mean = float(intensity_mean)

            if np.isnan(curr_2theta):
                print(f'{curr_2theta} too big: {i} out of {len(converted_2thetas)}')
                break
            
            closest_tensor_idx = int((curr_2theta - args.min_2theta) / (args.max_2theta - args.min_2theta) * args.xrd_vector_dim)
            xrd_tensor[closest_tensor_idx] = max(xrd_tensor[closest_tensor_idx],
                                                 intensity_mean + np.random.rand() * intensity_std)   

        #print(xrd_tensor)
        xrd_tensor = (xrd_tensor - torch.min(xrd_tensor)) / (torch.max(xrd_tensor) - torch.min(xrd_tensor))
        plt.plot(_2thetas, xrd_tensor.detach().cpu().numpy(), alpha=0.6, label='converted')
        filename = args.filepath.split('/')[-1]
    
    # TODO: generate GT XRD for sanity check
    structure = Structure.from_file(args.filepath)
    # wavelength
    curr_wavelength = WAVELENGTHS[args.desired_wavelength]
    # Create the XRD calculator
    xrd_calc = XRDCalculator(wavelength=curr_wavelength)
    # Calculate the XRD pattern
    pattern = xrd_calc.get_pattern(structure)
    simulated_xrd_tensor = create_xrd_tensor(args, pattern)
    plt.plot(_2thetas, simulated_xrd_tensor.detach().cpu().numpy(), alpha=0.6, label='simulated')
    
    plt.legend()
    plt.savefig(f'{filename}.png')

    return



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--filepath',
                        type=str,
                        default='/home/gabeguo/experimental_cif/zm5036Isup2.rtv.combined.cif')
    parser.add_argument('--desired_wavelength',
                        type=str,
                        default='CuKa')
    parser.add_argument('--xrd_vector_dim',
                        type=int,
                        default=4096)
    parser.add_argument('--min_2theta',
                        type=int,
                        default=0)
    parser.add_argument('--max_2theta',
                        type=int,
                        default=180)
    args = parser.parse_args()

    setattr(args, 'min_theta', args.min_2theta)
    setattr(args, 'max_theta', args.max_2theta)

    main(args)
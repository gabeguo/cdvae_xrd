import torch
import numpy as np
import argparse
import os
import matplotlib.pyplot as plt
from pymatgen.analysis.diffraction.xrd import XRDCalculator, WAVELENGTHS

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
        start_idx = xrd_loop_start_idx + (4+1)
        end_idx = all_lines.index('', start_idx)
        assert end_idx > start_idx

        xrd_tensor = torch.zeros(args.xrd_dim)

        _2theta_min_deg = float(get_field_value(all_lines, '_pd_meas_2theta_range_min'))
        _2theta_max_deg = float(get_field_value(all_lines, '_pd_meas_2theta_range_max'))
        _wavelength = float(get_field_value(all_lines, '_diffrn_radiation_wavelength'))

        print(_2theta_min_deg)
        print(_2theta_max_deg)
        print(_wavelength)

        theta_min_rad = (_2theta_min_deg / 2) * np.pi / 180
        theta_max_rad = (_2theta_max_deg / 2) * np.pi / 180

        xrd_intensities = all_lines[start_idx:end_idx]
        the_thetas = np.linspace(theta_min_rad, theta_max_rad, len(xrd_intensities))

        for i in range(len(the_thetas)):
            curr_theta = the_thetas[i]
            xrd_info = xrd_intensities[i]
            xrd_info = xrd_info.split()
            assert len(xrd_info) == len(expected_fields)
            intensity_mean, intensity_std = xrd_info[0].split('(')
            intensity_std = float(intensity_std[:-1])
            intensity_mean = float(intensity_mean)
            intensity_mean, intensity_std = xrd_info[0].split('(')
            intensity_std = float(intensity_std[:-1])
            intensity_mean = float(intensity_mean)

            q = 4 * np.pi * np.sin(curr_theta) / _wavelength
            if q  > args.max_Q:
                print(f'{q} exceeds max q {args.max_Q}: {i}/{len(the_thetas)}')
                break
            
            closest_tensor_idx = int((q - args.min_Q) / (args.max_Q - args.min_Q) * args.xrd_dim)
            xrd_tensor[closest_tensor_idx] = max(xrd_tensor[closest_tensor_idx],
                                                 intensity_mean)            

        #print(xrd_tensor)
        xrd_tensor = (xrd_tensor - torch.min(xrd_tensor)) / (torch.max(xrd_tensor) - torch.min(xrd_tensor))
        plt.plot(np.linspace(args.min_Q, args.max_Q, len(xrd_tensor)), xrd_tensor.detach().cpu().numpy())
        filename = args.filepath.split('/')[-1]
        plt.savefig(f'{filename}.png')

    return



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--filepath',
                        type=str,
                        default='/home/gabeguo/experimental_cif/zm5036Isup2.rtv.combined.cif')
    parser.add_argument('--xrd_dim',
                        type=int,
                        default=4096)
    parser.add_argument('--min_Q',
                        type=int,
                        default=0)
    parser.add_argument('--max_Q',
                        type=int,
                        default=10)
    args = parser.parse_args()
    main(args)
import torch
import numpy as np
import argparse
import os
import matplotlib.pyplot as plt
from pymatgen.analysis.diffraction.xrd import XRDCalculator, WAVELENGTHS
from pymatgen.core.structure import Structure
from scripts.gen_xrd import create_xrd_tensor

import warnings
warnings.filterwarnings("ignore")

def get_field_value(all_lines, desired_start):
    for i, the_line in enumerate(all_lines):
        if the_line.startswith(desired_start):
            split_line = the_line.split()
            if len(split_line) > 1:
                return split_line[-1]
            else:
                ret_val = all_lines[i+1]
                tokens = ret_val.split()
                for the_token in tokens:
                    try:
                        return float(the_token)
                    except ValueError:
                        continue
                raise ValueError(f'invalid field value for {desired_start}')
    raise ValueError(f'could not find field {desired_start}')

def find_index_of_xrd_loop(all_lines):
    for i in range(len(all_lines) - 1):
        if all_lines[i] == 'loop_' and '_pd_' in all_lines[i+1]:
            return i
    raise ValueError('could not find XRD loop')

def get_file_format(args):
    if 'zm5036' in args.filepath:
        expected_fields = ['_pd_meas_intensity_total', # '_pd_meas_intensity_total'
                        '_pd_proc_ls_weight',
                        '_pd_proc_intensity_bkg_calc',
                        '_pd_calc_intensity_total']
        intensity_idx = 0
        correction_idx = 2
        _2theta_idx = None
    elif 'os0043108' in args.filepath:
        expected_fields = ['_pd_meas_intensity_total', # '_pd_meas_intensity_total'
                        '_pd_proc_ls_weight',
                        '_pd_proc_intensity_calc_bkg',
                        '_pd_calc_intensity_total']
        intensity_idx = 0
        correction_idx = 2
        _2theta_idx = None
    elif any([x in args.filepath for x in ['av5088sup4', 'wm6137', 'wm2446', 'zb5035']]):
        expected_fields = ['_pd_meas_counts_total',
                        '_pd_proc_ls_weight',
                        '_pd_proc_intensity_bkg_calc',
                        '_pd_calc_intensity_total']  
        intensity_idx = 0
        correction_idx = 2
        _2theta_idx = None
    elif 'wm2699' in args.filepath:
        expected_fields = ['_pd_meas_counts_total', '_pd_calc_intensity_total']
        intensity_idx = 0
        correction_idx = None
        _2theta_idx = None
    elif 'wm2731' in args.filepath:
        expected_fields = ['_pd_proc_point_id',
                        '_pd_proc_2theta_corrected',
                        '_pd_proc_intensity_net',                   
                        '_pd_calc_intensity_net']
        intensity_idx = 2
        correction_idx = None
        _2theta_idx = 1
    elif 'kd5052' in args.filepath:
        expected_fields = ['_pd_proc_point_id',
                        '_pd_proc_d_spacing',
                        '_pd_proc_intensity_net',                   
                        '_pd_calc_intensity_net']  
        intensity_idx = 2
        correction_idx = None
        _2theta_idx = 1      
    elif 'wm2324' in args.filepath:
        expected_fields = ['_pd_proc_point_id',
                        '_pd_proc_2theta_corrected',  
                        '_pd_proc_intensity_total',
                        '_pd_calc_intensity_total',
                        '_pd_proc_intensity_bkg_calc']
        intensity_idx = 2
        correction_idx = 4
        _2theta_idx = 1
    elif 'iz1026' in args.filepath:
        expected_fields = ['_pd_proc_point_id',                         
                        '_pd_proc_2theta_corrected',            
                        '_pd_proc_energy_incident',            
                        '_pd_proc_d_spacing',                  
                        '_pd_proc_intensity_net',                     
                        '_pd_calc_intensity_net',                     
                        '_pd_proc_ls_weight'] 
        intensity_idx = 4
        correction_idx = None
        _2theta_idx = 1
    elif any([x in args.filepath for x in ['he5606', 'br1322', 'ck5030', 'gw5052']]):
        expected_fields = ['_pd_proc_point_id',
                        '_pd_proc_2theta_corrected',
                        '_pd_proc_intensity_net',
                        '_pd_calc_intensity_net',
                        '_pd_proc_ls_weight']
        intensity_idx = 2
        correction_idx = None
        _2theta_idx = 1
    elif 'dk5084' in args.filepath:
        expected_fields = ['_pd_proc_point_id',
                        '_pd_proc_2theta_corrected',
                        '_pd_proc_d_spacing',
                        '_pd_proc_intensity_net',
                        '_pd_calc_intensity_net',
                        '_pd_proc_ls_weight']
        intensity_idx = 3
        correction_idx = None
        _2theta_idx = 1
    elif 'ra5050' in args.filepath:
        expected_fields = ['_pd_meas_2theta_scan',
                        '_pd_proc_2theta_corrected',
                        '_pd_meas_counts_total',
                        '_pd_proc_ls_weight',
                        '_pd_proc_intensity_bkg_calc',
                        '_pd_calc_intensity_total']
        intensity_idx = 2
        correction_idx = 4
        _2theta_idx = 0
    elif 'hw5008PST-Ba_0.59GPasup15' in args.filepath:
        expected_fields = ['_pd_meas_time_of_flight',
                        '_pd_proc_d_spacing',
                        '_pd_meas_intensity_total',
                        '_pd_proc_ls_weight',
                        '_pd_proc_intensity_bkg_calc',
                        '_pd_calc_intensity_total']
        intensity_idx = 2
        correction_idx = None
        _2theta_idx = None
    
    return expected_fields, intensity_idx, correction_idx, _2theta_idx

def find_end_of_xrd(all_lines, start_idx):
    for i in range(start_idx + 1, len(all_lines)):
        if '_pd' and 'number_of_points' in all_lines[i]:
            return i
        if all_lines[i].strip() == '':
            return i
    raise ValueError('could not find end of xrd')

def main(args):
    sim_wavelength = WAVELENGTHS[args.desired_wavelength]
    assert os.path.exists(args.filepath)
    with open(args.filepath, 'r') as fin:
        all_lines = [x.strip() for x in fin.readlines()]
        xrd_loop_start_idx = find_index_of_xrd_loop(all_lines)
        assert xrd_loop_start_idx >= 0
        expected_fields, intensity_idx, correction_idx, _2theta_idx = get_file_format(args)
        
        for idx in range(len(expected_fields)):
            assert expected_fields[idx] == all_lines[xrd_loop_start_idx + (idx + 1)].strip().split()[0]
        for idx in range(xrd_loop_start_idx + (len(expected_fields)+1), len(all_lines)):
            if all_lines[idx].strip() != '':
                start_idx = idx
                break
        
        end_idx = min(all_lines.index('', start_idx), find_end_of_xrd(all_lines, start_idx))
        assert end_idx > start_idx

        xrd_intensities = all_lines[start_idx:end_idx]
        print(xrd_intensities[0])

        _exp_wavelength = float(get_field_value(all_lines, '_diffrn_radiation_wavelength'))
        if _2theta_idx is not None:
            _2theta_min_deg = float(xrd_intensities[0].split()[_2theta_idx])
            _2theta_max_deg = float(xrd_intensities[-1].split()[_2theta_idx])
        else:
            _2theta_min_deg = float(get_field_value(all_lines, '_pd_meas_2theta_range_min'))
            _2theta_max_deg = float(get_field_value(all_lines, '_pd_meas_2theta_range_max'))

        print(_2theta_min_deg)
        print(_2theta_max_deg)
        print(_exp_wavelength)

        theta_min_rad = (_2theta_min_deg / 2) * np.pi / 180
        theta_max_rad = (_2theta_max_deg / 2) * np.pi / 180
        the_thetas_rad = np.linspace(theta_min_rad, theta_max_rad, len(xrd_intensities))
        print('theta range (rad):', np.min(the_thetas_rad), np.max(the_thetas_rad))
        converted_2thetas = 2 * np.arcsin(np.sin(the_thetas_rad) * sim_wavelength / _exp_wavelength) * 180 / np.pi
        print('2theta range (deg):', np.nanmin(converted_2thetas), np.nanmax(converted_2thetas))
        
        xrd_tensor = torch.zeros(args.xrd_vector_dim)

        _2thetas = np.linspace(args.min_2theta, args.max_2theta, args.xrd_vector_dim)

        min_val = np.inf
        max_val = -np.inf
        for i in range(len(converted_2thetas)):
            curr_2theta = converted_2thetas[i]
            xrd_info = xrd_intensities[i]
            xrd_info = xrd_info.split()
            #print(xrd_info, expected_fields)
            if len(xrd_info) != len(expected_fields):
                break
            try:
                intensity_mean = float(xrd_info[intensity_idx])
            except ValueError:
                intensity_mean, intensity_std = xrd_info[intensity_idx].split('(')
                intensity_std = float(intensity_std[:-1])
                intensity_mean = float(intensity_mean)
            
            if correction_idx is not None:
                try:
                    intensity_mean -= float(xrd_info[correction_idx])
                except ValueError:
                    pass

            if np.isnan(curr_2theta):
                print(f'{curr_2theta} too big: {i} out of {len(converted_2thetas)}')
                break
            
            closest_tensor_idx = int((curr_2theta - args.min_2theta) / (args.max_2theta - args.min_2theta) * args.xrd_vector_dim)
            xrd_tensor[closest_tensor_idx] = max(xrd_tensor[closest_tensor_idx],
                                                 intensity_mean)

            min_val = min(min_val, intensity_mean)
            max_val = max(max_val, intensity_mean)   

        #print(xrd_tensor)
        min_val = max(min_val, 0)
        xrd_tensor = torch.maximum((xrd_tensor - min_val) / (max_val - min_val), torch.zeros_like(xrd_tensor))
        plt.plot(_2thetas, xrd_tensor.detach().cpu().numpy(), alpha=0.6, label='converted')
        filename = args.filepath.split('/')[-1]
    
    # TODO: generate GT XRD for sanity check
    structure = Structure.from_file(args.filepath)
    print(len(structure.sites))
    # wavelength
    curr_wavelength = WAVELENGTHS[args.desired_wavelength]
    # Create the XRD calculator
    xrd_calc = XRDCalculator(wavelength=curr_wavelength)
    # Calculate the XRD pattern
    pattern = xrd_calc.get_pattern(structure)
    simulated_xrd_tensor = create_xrd_tensor(args, pattern)
    plt.plot(_2thetas, simulated_xrd_tensor.detach().cpu().numpy(), alpha=0.6, label='simulated')
    
    plt.legend()
    plt.savefig(os.path.join(args.output_dir, f'{filename}.png'))

    return



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--filepath',
                        type=str,
                        default='/home/gabeguo/experimental_cif/he5606SrLaMgRuO6_300Ksup2.rtv.combined.cif')
                        #default='/home/gabeguo/experimental_cif/br1322Isup2.rtv.combined.cif')
                        #default='/home/gabeguo/experimental_cif/av5088sup4.rtv.combined.cif')
                        #default='/home/gabeguo/experimental_cif/ra5050Isup2.rtv.combined.cif')
                        #default='/home/gabeguo/experimental_cif/os0043108Ksup3.rtv.combined.cif')
                        #default='/home/gabeguo/experimental_cif/dk5084I_2sup8.rtv.combined.cif')
                        #default='/home/gabeguo/experimental_cif/ck5030Vsup6.rtv.combined.cif')
                        #default='/home/gabeguo/experimental_cif/gw5052Mg2Sn_100K_LTsup23.rtv.combined.cif')
                        #default='/home/gabeguo/experimental_cif/iz1026Isup2.rtv.combined.cif')
                        #default='/home/gabeguo/experimental_cif/wm2446Isup2.rtv.combined.cif')
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
    parser.add_argument('--output_dir',
                        type=str,
                        default='xrd_images')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    setattr(args, 'min_theta', args.min_2theta)
    setattr(args, 'max_theta', args.max_2theta)

    main(args)
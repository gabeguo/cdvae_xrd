import torch
import numpy as np
import argparse
import os
import matplotlib.pyplot as plt
from pymatgen.analysis.diffraction.xrd import XRDCalculator, WAVELENGTHS
from pymatgen.core.structure import Structure
import pandas as pd
from pymatgen.core import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.io.cif import CifWriter, CifParser
from tqdm import tqdm

# import warnings
# warnings.filterwarnings("ignore")

def get_field_value(all_lines, desired_start, is_num=True):
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
                if not is_num:
                    return ret_val
                raise ValueError(f'invalid field value for {desired_start}')
    raise ValueError(f'could not find field {desired_start}')

def find_index_of_xrd_loop(all_lines):
    for i in range(len(all_lines) - 1):
        if all_lines[i] == 'loop_' and ('_pd_' in all_lines[i+1]):
            return i
    raise ValueError('could not find XRD loop')

def get_file_format(args, filepath):
    if 'zm5036' in filepath:
        expected_fields = ['_pd_meas_intensity_total',
                        '_pd_proc_ls_weight',
                        '_pd_proc_intensity_bkg_calc',
                        '_pd_calc_intensity_total']
        intensity_idx = 0
        correction_idx = 2
        _2theta_idx = None
    elif any([x in filepath for x in ['os0043108', 'sn0038Isup']]):
        expected_fields = ['_pd_meas_intensity_total',
                        '_pd_proc_ls_weight',
                        '_pd_proc_intensity_calc_bkg',
                        '_pd_calc_intensity_total']
        intensity_idx = 0
        correction_idx = 2
        _2theta_idx = None
    elif any([x in filepath for x in ['av5088sup4', 'wm6137', 'wm2446', 'zb5035']]):
        expected_fields = ['_pd_meas_counts_total',
                        '_pd_proc_ls_weight',
                        '_pd_proc_intensity_bkg_calc',
                        '_pd_calc_intensity_total']  
        intensity_idx = 0
        correction_idx = 2
        _2theta_idx = None
    elif 'wm2699' in filepath:
        expected_fields = ['_pd_meas_counts_total', '_pd_calc_intensity_total']
        intensity_idx = 0
        correction_idx = None
        _2theta_idx = None
    elif any([x in filepath for x in ['sq1033', 'wm2731', 'wf5122']]):
        expected_fields = ['_pd_proc_point_id',
                        '_pd_proc_2theta_corrected',
                        '_pd_proc_intensity_net',                   
                        '_pd_calc_intensity_net']
        intensity_idx = 2
        correction_idx = None
        _2theta_idx = 1
    elif any([x in filepath for x in ['kd5052']]):
        expected_fields = ['_pd_proc_point_id',
                        '_pd_proc_d_spacing',
                        '_pd_proc_intensity_net',                   
                        '_pd_calc_intensity_net']  
        intensity_idx = 2
        correction_idx = None
        _2theta_idx = 1      
    elif any([x in filepath for x in ['ks5409', 'sq3214', 'wm2324', 'wm2097']]):
        expected_fields = ['_pd_proc_point_id',
                        '_pd_proc_2theta_corrected',  
                        '_pd_proc_intensity_total',
                        '_pd_calc_intensity_total',
                        '_pd_proc_intensity_bkg_calc']
        intensity_idx = 2
        correction_idx = 4
        _2theta_idx = 1
    elif any(x in filepath for x in ['br1340', 'iz1026']):
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
    elif any([x in filepath for x in ['ck5018', 'sn5085', 'br1322', 'dk5008', 'he5606', 'br1322', 'ck5030', 'gw5052']]):
        expected_fields = ['_pd_proc_point_id',
                        '_pd_proc_2theta_corrected',
                        '_pd_proc_intensity_net',
                        '_pd_calc_intensity_net',
                        '_pd_proc_ls_weight']
        intensity_idx = 2
        correction_idx = None
        _2theta_idx = 1
    elif 'wn6225' in filepath:
        expected_fields = ['_pd_proc_point_id',
                        '_pd_proc_2theta_corrected',
                        '_pd_proc_energy_incident',
                        '_pd_proc_d_spacing',
                        '_pd_proc_intensity_net',
                        '_pd_calc_intensity_net']
        intensity_idx = 4
        correction_idx = None
        _2theta_idx = 1
    elif 'dk5084' in filepath:
        expected_fields = ['_pd_proc_point_id',
                        '_pd_proc_2theta_corrected',
                        '_pd_proc_d_spacing',
                        '_pd_proc_intensity_net',
                        '_pd_calc_intensity_net',
                        '_pd_proc_ls_weight']
        intensity_idx = 3
        correction_idx = None
        _2theta_idx = 1
    elif 'ra5050' in filepath:
        expected_fields = ['_pd_meas_2theta_scan',
                        '_pd_proc_2theta_corrected',
                        '_pd_meas_counts_total',
                        '_pd_proc_ls_weight',
                        '_pd_proc_intensity_bkg_calc',
                        '_pd_calc_intensity_total']
        intensity_idx = 2
        correction_idx = 4
        _2theta_idx = 0
    elif 'hw5008PST-Ba_0.59GPasup15' in filepath:
        expected_fields = ['_pd_meas_time_of_flight',
                        '_pd_proc_d_spacing',
                        '_pd_meas_intensity_total',
                        '_pd_proc_ls_weight',
                        '_pd_proc_intensity_bkg_calc',
                        '_pd_calc_intensity_total']
        intensity_idx = 2
        correction_idx = 4
        _2theta_idx = None
    elif 'ko5041YMO-1400Ksup4' in filepath:
        expected_fields = ['_pd_meas_time_of_flight',
                        '_pd_proc_d_spacing',
                        '_pd_meas_intensity_total',
                        '_pd_proc_intensity_total',
                        '_pd_proc_ls_weight',
                        '_pd_proc_intensity_bkg_calc',
                        '_pd_calc_intensity_total']
        intensity_idx = 3
        correction_idx = None
        _2theta_idx = None
    elif 'wh5012' in filepath:
        expected_fields = ['_pd_proc_2theta_corrected',
                           '_pd_proc_intensity_net']
        intensity_idx = 1
        correction_idx = None
        _2theta_idx = 0
    elif 'sh0123Xray' in filepath:
        expected_fields = ['_pd_peak_id',
                        '_pd_peak_2theta_centroid\\',
                        '_pd_peak_d_spacing',
                        '_pd_peak_pk_height',
                        '_pd_peak_width_2theta',
                        '_pd_peak_wavelaength_id',
                        '_refln_index_h',  
                        '_refln_index_k',
                        '_refln_index_l'] 
        intensity_idx = 3
        correction_idx = None
        _2theta_idx = 1
    return expected_fields, intensity_idx, correction_idx, _2theta_idx

def find_end_of_xrd(all_lines, start_idx):
    for i in range(start_idx + 1, len(all_lines)):
        if '_pd' and 'number_of_points' in all_lines[i]:
            return i
        if all_lines[i].strip() == '':
            return i
    raise ValueError('could not find end of xrd')

def create_xrd_tensor(args, pattern, wavelength, min_Q, max_Q):
    peak_data = torch.zeros(args.xrd_vector_dim)
    peak_locations_theta_deg = pattern.x / 2
    peak_locations_theta_rad = np.radians(peak_locations_theta_deg)
    peak_locations_Q = 4 * np.pi * np.sin(peak_locations_theta_rad) / wavelength
    peak_values = pattern.y.tolist()
    for i2 in range(len(peak_locations_Q)):
        curr_Q = peak_locations_Q[i2]
        height = peak_values[i2] / 100
        scaled_location = int(args.xrd_vector_dim * curr_Q / (max_Q - min_Q))
        peak_data[scaled_location] = max(peak_data[scaled_location], height) # just in case really close

    return peak_data

def create_data(args, filepath):
    filename = filepath.split('/')[-1]
    print(filename)

    sim_wavelength = WAVELENGTHS[args.desired_wavelength]
    assert os.path.exists(filepath)
    with open(filepath, 'r') as fin:
        print(filepath)
        all_lines = [x.strip() for x in fin.readlines()]
        xrd_loop_start_idx = find_index_of_xrd_loop(all_lines)
        assert xrd_loop_start_idx >= 0
        expected_fields, intensity_idx, correction_idx, _2theta_idx = get_file_format(args, filepath)
        
        for idx in range(len(expected_fields)):
            assert expected_fields[idx] == all_lines[xrd_loop_start_idx + (idx + 1)].strip().split()[0]
        for idx in range(xrd_loop_start_idx + (len(expected_fields)+1), len(all_lines)):
            if all_lines[idx].strip() != '':
                start_idx = idx
                break
        
        end_idx = min(all_lines.index('', start_idx), find_end_of_xrd(all_lines, start_idx))
        assert end_idx > start_idx

        xrd_intensities = all_lines[start_idx:end_idx]
        print('\t', xrd_intensities[0])
        
        assert 'neutron' not in get_field_value(all_lines, '_diffrn_radiation_type', is_num=False).lower()
        try:
            _exp_wavelength = float(get_field_value(all_lines, '_diffrn_radiation_wavelength'))
        except ValueError as e:
            print(f'\tcatch {e}; trying _diffrn_radiation_type')
            exp_radiation_type = get_field_value(all_lines, '_diffrn_radiation_type', is_num=False)
            if exp_radiation_type == "'Cu K\\a'":
                _exp_wavelength = WAVELENGTHS['CuKa']
            elif exp_radiation_type == "'MoK\\a'":
                _exp_wavelength = WAVELENGTHS['MoKa']
        if _2theta_idx is not None:
            _2theta_min_deg = float(xrd_intensities[0].split()[_2theta_idx])
            _2theta_max_deg = float(xrd_intensities[-1].split()[_2theta_idx])
        else:
            _2theta_min_deg = float(get_field_value(all_lines, '_pd_meas_2theta_range_min'))
            _2theta_max_deg = float(get_field_value(all_lines, '_pd_meas_2theta_range_max'))

        print('\tmin 2 theta', _2theta_min_deg)
        print('\tmax 2 theta', _2theta_max_deg)
        print('\texperimental wavlength', _exp_wavelength)

        theta_min_rad = (_2theta_min_deg / 2) * np.pi / 180
        theta_max_rad = (_2theta_max_deg / 2) * np.pi / 180
        the_thetas_rad = np.linspace(theta_min_rad, theta_max_rad, len(xrd_intensities))
        print('\ttheta range (rad):', np.min(the_thetas_rad), np.max(the_thetas_rad))
        experimental_Qs = 4 * np.pi * np.sin(the_thetas_rad) / _exp_wavelength
        # converted_2thetas = 2 * np.arcsin(np.sin(the_thetas_rad) * sim_wavelength / _exp_wavelength) * 180 / np.pi
        # print('\t2theta range (deg):', np.nanmin(converted_2thetas), np.nanmax(converted_2thetas))
        print(f'\tQ range: {np.min(experimental_Qs)}, {np.max(experimental_Qs)}')
        
        # TODO: write function for 2theta to Q
        xrd_tensor = torch.zeros(args.xrd_vector_dim)
        min_Q = 4 * np.pi * np.sin(np.radians(args.min_2theta / 2)) / sim_wavelength
        max_Q = 4 * np.pi * np.sin(np.radians(args.max_2theta / 2)) / sim_wavelength
        desired_Qs = np.linspace(min_Q, max_Q, args.xrd_vector_dim)
        #_2thetas = np.linspace(args.min_2theta, args.max_2theta, args.xrd_vector_dim)

        min_val = np.inf
        max_val = -np.inf
        for i in range(len(experimental_Qs)):
            xrd_info = xrd_intensities[i]
            xrd_info = xrd_info.split()
            if _2theta_idx is not None:
                curr_2theta_unconverted_deg = float(xrd_info[_2theta_idx])
                curr_2theta_unconverted_rad = curr_2theta_unconverted_deg * np.pi / 180
                curr_theta_unconverted_rad = curr_2theta_unconverted_rad / 2
                curr_Q = 4 * np.pi * np.sin(curr_theta_unconverted_rad) / _exp_wavelength
                #curr_2theta = 2 * np.arcsin(np.sin(curr_theta_unconverted_rad) * sim_wavelength / _exp_wavelength) * 180 / np.pi
            else:
                #curr_2theta = converted_2thetas[i]
                curr_Q = experimental_Qs[i]
            
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
            
            closest_tensor_idx = int((curr_Q - min_Q) / (max_Q - min_Q) * args.xrd_vector_dim)
            if closest_tensor_idx >= xrd_tensor.shape[0]:
                print(f'\t{curr_Q} too large: {i} out of {len(experimental_Qs)}')
                break
            xrd_tensor[closest_tensor_idx] = max(xrd_tensor[closest_tensor_idx],
                                                 intensity_mean)

            min_val = min(min_val, intensity_mean)
            max_val = max(max_val, intensity_mean)   

        #print(xrd_tensor)
        min_val = max(min_val, 0) 
        xrd_tensor = torch.maximum((xrd_tensor - min_val) / (max_val - min_val), torch.zeros_like(xrd_tensor))
    
    cif_parser = CifParser(filepath)
    structure = cif_parser.get_structures()[0]
    #Structure.from_file(filepath)
    print(f'\t{len(structure.sites)} sites')

    if args.plot_img:
        # wavelength
        curr_wavelength = WAVELENGTHS[args.desired_wavelength]
        # Create the XRD calculator
        xrd_calc = XRDCalculator(wavelength=curr_wavelength)

        # Calculate the XRD pattern
        pattern = xrd_calc.get_pattern(structure)
        simulated_xrd_tensor = create_xrd_tensor(args, pattern, wavelength=sim_wavelength, min_Q=min_Q, max_Q=max_Q)
        plt.plot(desired_Qs, simulated_xrd_tensor.detach().cpu().numpy(), alpha=0.6, label='simulated')
        plt.plot(desired_Qs, xrd_tensor.detach().cpu().numpy(), alpha=0.6, label='converted')

        # Sanity check spacegroups
        sga = SpacegroupAnalyzer(structure)
        print(sga.get_crystal_system())
        conventional_structure = sga.get_conventional_standard_structure()

        alt_pattern = xrd_calc.get_pattern(conventional_structure)
        alt_simulated_xrd_tensor = create_xrd_tensor(args, alt_pattern, wavelength=sim_wavelength, min_Q=min_Q, max_Q=max_Q)
        plt.plot(desired_Qs, alt_simulated_xrd_tensor.detach().cpu().numpy(), alpha=0.6, label='simulated (spacegroup recalc)')

        plt.legend()
        vis_filepath = os.path.join(args.output_dir, 'vis')
        os.makedirs(vis_filepath, exist_ok=True)
        plt.savefig(os.path.join(vis_filepath, f'{filename}.png'))
        plt.cla()

    # Pretty formula
    pretty_formula = structure.composition.reduced_formula

    # Elements
    #all_atoms = [element.symbol for site in structure.sites for element in site.species.elements]
    #print(f'\tnum atoms: {len(all_atoms)}')
    #unique_elements = list(set([element.symbol for element in structure.composition.elements]))
    unique_elements = list(set([str(element) for element in structure.species]))

    # Space Group Number
    spacegroup_analyzer = SpacegroupAnalyzer(structure)
    spacegroup_number = spacegroup_analyzer.get_space_group_number()

    # sanity check that CiF works
    temp_cif_path = os.path.join(args.output_dir, 'temp.cif')
    structure.to(filename=temp_cif_path)
    dummy = Structure.from_file(temp_cif_path)
    cif_writer = CifWriter(structure)

    # Print the extracted information
    print(f"\tPretty Formula: {pretty_formula}")
    print(f"\tElements: {unique_elements}")
    print(f"\tSpace Group Number: {spacegroup_number}")
    print(f"\tXRD tensor shape: {xrd_tensor.shape}")

    return {
        'material_id': filename,
        'pretty_formula': pretty_formula,
        'elements': unique_elements,
        'cif': cif_writer.__str__(),
        'spacegroup.number': spacegroup_number,
        'xrd': xrd_tensor
    }

if __name__ == "__main__":
    FILEPATHS = [
        '/home/gabeguo/experimental_cif/av5088sup4.rtv.combined.cif',
        '/home/gabeguo/experimental_cif/br1322Isup2.rtv.combined.cif',
        '/home/gabeguo/experimental_cif/br1340Isup2.rtv.combined.cif',
        '/home/gabeguo/experimental_cif/ck5030Vsup6.rtv.combined.cif',
        '/home/gabeguo/experimental_cif/gw5052Mg2Sn_100K_LTsup23.rtv.combined.cif',
        '/home/gabeguo/experimental_cif/gw5052Mg2Si_100K_LTsup2.rtv.combined.cif',
        '/home/gabeguo/experimental_cif/ks5409BTsup2.rtv.combined.cif',
        '/home/gabeguo/experimental_cif/sh0123Xraysup5.rtv.combined.cif',
        '/home/gabeguo/experimental_cif/sq1033Isup2.rtv.combined.cif',
        '/home/gabeguo/experimental_cif/sq3214Isup2.rtv.combined.cif',
        '/home/gabeguo/experimental_cif/iz1026Isup2.rtv.combined.cif',
        '/home/gabeguo/experimental_cif/wh5012phaseIIsup2.rtv.combined.cif',
        '/home/gabeguo/experimental_cif/wh5012phaseIIIsup3.rtv.combined.cif',
        '/home/gabeguo/experimental_cif/wm2446Isup2.rtv.combined.cif',
        '/home/gabeguo/experimental_cif/wn6225Isup2.rtv.combined.cif',
    ]
    parser = argparse.ArgumentParser()
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
                        default='/home/gabeguo/cdvae_xrd/data/experimental_xrd')
    parser.add_argument('--plot_img',
                        action='store_true')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    setattr(args, 'min_theta', args.min_2theta)
    setattr(args, 'max_theta', args.max_2theta)

    output_df = pd.DataFrame({
        'material_id': list(),
        'pretty_formula': list(),
        'elements': list(),
        'cif': list(),
        'spacegroup.number': list(),
        'xrd': list()
    })
    failed_materials = 0
    for filepath in tqdm(FILEPATHS):
        try:
            curr_data = create_data(args, filepath)
            output_df = output_df.append(curr_data, ignore_index=True)
        except AttributeError as e:
            print(e)
            print('abort element')
            failed_materials += 1
    
    print(f'{failed_materials} out of {len(FILEPATHS)} materials failed')
    
    print(output_df)

    output_df.to_pickle(os.path.join(args.output_dir, 'test.csv'))

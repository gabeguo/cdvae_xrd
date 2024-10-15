import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
from pymatgen.core import Lattice, Structure, Molecule
from pymatgen.io.cif import CifWriter
import torch
from tqdm import tqdm
from pymatgen.analysis.diffraction.xrd import XRDCalculator, WAVELENGTHS

WAVELENGTH=0.7294437

def standardize_pattern(x):
    return (np.array(x) - np.min(x)) / (np.max(x) - np.min(x))

def convert_to_q(x, wavelength=WAVELENGTH):
    x = np.array(x)
    peak_locations_theta_deg = x / 2
    peak_locations_theta_rad = np.radians(peak_locations_theta_deg)
    peak_locations_Q = 4 * np.pi * np.sin(peak_locations_theta_rad) / wavelength

    return peak_locations_Q

def get_2theta_intensity(data_path):
    _2thetas = list()
    _intensities_a = list()
    _intensities_b = list()
    with open(data_path, 'r') as fin:
        the_lines = [x.strip() for x in fin.readlines()]
        first_line_idx = the_lines.index('*/') + 1
        the_lines = the_lines[first_line_idx:]
        for curr_line in the_lines:
            curr_2theta, curr_intensity_a, curr_intensity_b = curr_line.split()
            _2thetas.append(float(curr_2theta))
            _intensities_a.append(float(curr_intensity_a))
            _intensities_b.append(float(curr_intensity_b))
    _intensities_a = standardize_pattern(_intensities_a)
    _intensities_b = standardize_pattern(_intensities_b)

    # plot 2-theta
    plt.xlabel(r'2\theta')
    plt.plot(_2thetas, _intensities_a, alpha=0.7, label='background removed')
    plt.plot(_2thetas, _intensities_b, alpha=0.7, label='raw')
    plt.legend()
    save_dir = os.path.dirname(data_path)
    img_filename = os.path.basename(data_path)[:-4] + ".png"
    plt.savefig(os.path.join(save_dir, img_filename))
    plt.close()

    q_values = convert_to_q(_2thetas)

    # plot Q
    plt.xlabel("Q")
    plt.plot(q_values, _intensities_a)
    plt.savefig(os.path.join(save_dir, f"q_{img_filename}"))
    plt.close()

    return q_values, _intensities_a

def regrid_Q_xrd_pattern(experimental_Qs, xrd_intensities, min_2theta=0, max_2theta=180, xrd_vector_dim=4096, wavelength=WAVELENGTHS['CuKa']):
    xrd_tensor = torch.zeros(xrd_vector_dim)
    min_Q = 4 * np.pi * np.sin(np.radians(min_2theta / 2)) / wavelength
    max_Q = 4 * np.pi * np.sin(np.radians(max_2theta / 2)) / wavelength
    desired_Qs = np.linspace(min_Q, max_Q, xrd_vector_dim)
    #_2thetas = np.linspace(args.min_2theta, args.max_2theta, args.xrd_vector_dim)

    min_val = np.inf
    max_val = -np.inf
    for i in range(len(experimental_Qs)):
        curr_Q = experimental_Qs[i]
        curr_intensity = xrd_intensities[i]
                
        closest_tensor_idx = int((curr_Q - min_Q) / (max_Q - min_Q) * xrd_vector_dim)
        if closest_tensor_idx >= xrd_tensor.shape[0]:
            print(f'\t{curr_Q} too large: {i} out of {len(experimental_Qs)}')
            break
        xrd_tensor[closest_tensor_idx] = max(xrd_tensor[closest_tensor_idx], curr_intensity)

        min_val = min(min_val, curr_intensity)
        max_val = max(max_val, curr_intensity)   

    min_val = max(min_val, 0) 
    xrd_tensor = torch.maximum((xrd_tensor - min_val) / (max_val - min_val), torch.zeros_like(xrd_tensor))

    return desired_Qs, xrd_tensor

def adjust_background_crude(desired_Qs, xrd_tensor, Q_thresh=2.5):
    max_index = len([x for x in desired_Qs if x < Q_thresh])
    min_index = None
    start_value = None
    for i in range(len(xrd_tensor)):
        if xrd_tensor[i] > 0:
            min_index = i
            start_value = xrd_tensor[i]
            break
    assert start_value is not None
    delta_y = start_value / (max_index - min_index)
    adjusted_xrd_tensor = torch.clone(xrd_tensor)
    for i in range(min_index, max_index):
        adjusted_xrd_tensor[i] -= delta_y * (max_index - i)
        adjusted_xrd_tensor[i] = max(0, adjusted_xrd_tensor[i])
    mini_peak = torch.max(adjusted_xrd_tensor[min_index:max_index])
    for i in range(min_index, max_index):
        adjusted_xrd_tensor[i] *= (2 * (adjusted_xrd_tensor[i] / mini_peak - 0.5))**2
    adjusted_xrd_tensor /= torch.max(adjusted_xrd_tensor)
    
    return adjusted_xrd_tensor

def plot_xrd(q, regridded_xrd, adjusted_xrd, filepath):
    if isinstance(q, torch.Tensor):
        q = q.numpy()
    if isinstance(regridded_xrd, torch.Tensor):
        regridded_xrd = regridded_xrd.numpy()
    if isinstance(adjusted_xrd, torch.Tensor):
        adjusted_xrd = adjusted_xrd.numpy()
    
    plt.plot(q, regridded_xrd, label='Non-background adjusted', alpha=0.6)
    plt.plot(q, adjusted_xrd, label="Background adjusted", alpha=0.6)
    plt.xlabel("Q")
    plt.ylabel("Intensity")
    plt.legend()
    save_dir = os.path.dirname(filepath)
    img_filename = os.path.basename(filepath)[:-4] + "_adjusted_v_orig.png"
    plt.savefig(os.path.join(save_dir, img_filename))
    plt.close()

    return

def get_data(filepath, composition, idx):
    total_num_atoms = sum([x for x in composition.values()])
    element_list = list()
    for elem in composition:
        for _ in range(composition[elem]):
            element_list.append(elem)
    assert len(element_list) == total_num_atoms

    coords = np.random.uniform(size=(total_num_atoms, 3))
    
    lattice = Lattice.from_parameters(a=3, b=3, c=3, alpha=90, beta=90, gamma=90)
    struct = Structure(lattice, element_list, coords)

    cif_writer = CifWriter(struct)

    orig_q, orig_grid_xrd_intensity = get_2theta_intensity(filepath)
    regridded_q, regridded_xrd = regrid_Q_xrd_pattern(orig_q, orig_grid_xrd_intensity)
    adjusted_xrd = adjust_background_crude(desired_Qs=regridded_q, xrd_tensor=regridded_xrd, Q_thresh=2.5)

    plot_xrd(regridded_q, regridded_xrd, adjusted_xrd, filepath)

    return {
        "material_id": f"mp-{idx}",
        "pretty_formula": "".join([f"{a[0]}{a[1]}" for a in composition.items()]),
        "elements": element_list,
        "cif": cif_writer.__str__(),
        "spacegroup.number": 0,
        "xrd": adjusted_xrd
    }

if __name__ == "__main__":
    output_df = pd.DataFrame({
        'material_id': list(),
        'pretty_formula': list(),
        'elements': list(),
        'cif': list(),
        'spacegroup.number': list(),
        'xrd': list()
    })
    running_count = 0
    for multiple in tqdm(range(1, 5+1)):
        output_df = output_df._append(get_data('/home/gabeguo/cdvae_xrd/real_data/BL21Robot_0321-2023-09-02-1241_scan2_Mn3Ge.xye', composition={'Mn':3*multiple, 'Ge':1*multiple}, idx=running_count), ignore_index=True)
        running_count += 1
    for Ge_multiple in tqdm(range(1, 1+1)):
        for overall_multiple in range(1, 5+1):
            mn_quantity = 3*overall_multiple
            ge_quantity = Ge_multiple*overall_multiple
            if mn_quantity + ge_quantity > 20:
                continue
            output_df = output_df._append(get_data('/home/gabeguo/cdvae_xrd/real_data/BL21Robot_0327-2023-09-02-1312_scan2_Mn3GeN.xye', composition={'Mn':mn_quantity, 'Ge':ge_quantity}, idx=running_count), ignore_index=True)
            running_count += 1

    output_df.to_pickle('/home/gabeguo/cdvae_xrd/real_data/test.csv')

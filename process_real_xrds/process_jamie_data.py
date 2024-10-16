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

def read_xy_q_vs_intensity(filepath):
    assert os.path.exists(filepath)
    with open(filepath, 'r') as fin:
        Qs = list()
        intensities = list()
        all_lines = fin.readlines()[1:]
        for curr_line in all_lines:
            curr_q, curr_i = curr_line.split(',')
            Qs.append(float(curr_q.strip()))
            intensities.append(float(curr_i.strip()))

    return Qs, intensities

def plot_xrd(Qs, intensities, filepath, name):
    # plot Q
    plt.xlabel("Q")
    plt.plot(Qs, intensities)
    save_dir = os.path.dirname(filepath)
    img_filename = os.path.basename(filepath)[:-3] + ".png"
    plt.savefig(os.path.join(save_dir, f"{name}_{img_filename}"))
    plt.close()

def regrid_Q_xrd_pattern(experimental_Qs, xrd_intensities, min_2theta=0, max_2theta=180, xrd_vector_dim=4096, wavelength=WAVELENGTHS['CuKa']):
    xrd_tensor = torch.zeros(xrd_vector_dim)
    min_Q = 4 * np.pi * np.sin(np.radians(min_2theta / 2)) / wavelength
    max_Q = 4 * np.pi * np.sin(np.radians(max_2theta / 2)) / wavelength
    desired_Qs = np.linspace(min_Q, max_Q, xrd_vector_dim)
    #_2thetas = np.linspace(args.min_2theta, args.max_2theta, args.xrd_vector_dim)

    min_val = None
    max_val = -np.inf
    for i in range(len(experimental_Qs)):
        curr_Q = experimental_Qs[i]
        curr_intensity = xrd_intensities[i]
                
        closest_tensor_idx = int((curr_Q - min_Q) / (max_Q - min_Q) * xrd_vector_dim)
        if closest_tensor_idx >= xrd_tensor.shape[0]:
            print(f'\t{curr_Q} too large: {i} out of {len(experimental_Qs)}')
            break
        xrd_tensor[closest_tensor_idx] = max(xrd_tensor[closest_tensor_idx], curr_intensity)

        if curr_intensity > 0 and min_val is None:
            min_val = min(xrd_intensities[i:i+10])
        max_val = max(max_val, curr_intensity)   

    min_val = max(min_val, 0) 
    print(min_val)
    xrd_tensor = torch.maximum((xrd_tensor - min_val) / (max_val - min_val), torch.zeros_like(xrd_tensor))

    return desired_Qs, xrd_tensor

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

    orig_q, orig_grid_xrd_intensity = read_xy_q_vs_intensity(filepath) #get_2theta_intensity(filepath)
    regridded_q, regridded_xrd = regrid_Q_xrd_pattern(orig_q, orig_grid_xrd_intensity)
    plot_xrd(orig_q, orig_grid_xrd_intensity, filepath=filepath, name="orig_grid")
    plot_xrd(regridded_q, regridded_xrd, filepath=filepath, name="regridded")

    return {
        "material_id": f"mp-{idx}",
        "pretty_formula": "".join([f"{a[0]}{a[1]}" for a in composition.items()]),
        "elements": element_list,
        "cif": cif_writer.__str__(),
        "spacegroup.number": 0,
        "xrd": regridded_xrd
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
        output_df = output_df._append(get_data('/home/gabeguo/cdvae_xrd/real_data/Mn3Ge_backsub.xy', composition={'Mn':3*multiple, 'Ge':1*multiple}, idx=running_count), ignore_index=True)
        running_count += 1
    for Ge_multiple in tqdm(range(1, 1+1)):
        for overall_multiple in range(1, 5+1):
            mn_quantity = 3*overall_multiple
            ge_quantity = Ge_multiple*overall_multiple
            if mn_quantity + ge_quantity > 20:
                continue
            output_df = output_df._append(get_data('/home/gabeguo/cdvae_xrd/real_data/Mn3GeNx_backsub.xy', composition={'Mn':mn_quantity, 'Ge':ge_quantity}, idx=running_count), ignore_index=True)
            running_count += 1

    output_df.to_pickle('/home/gabeguo/cdvae_xrd/real_data/test.csv')

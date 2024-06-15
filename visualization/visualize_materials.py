import torch
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from pymatgen.core.periodic_table import Element
from pymatgen.core.structure import Structure
from pymatgen.core.lattice import Lattice
from pymatgen.analysis.diffraction.xrd import XRDCalculator, WAVELENGTHS
from scipy.ndimage import gaussian_filter1d
from scripts.gen_xrd import create_xrd_tensor
from scripts.eval_utils import get_crystals_list
import warnings
import os
import argparse
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

# Thanks ChatGPT!
# Thanks https://www.umass.edu/microbio/chime/pe_beta/pe/shared/cpk-rgb.htm
CPK_COLORS = {
    "C": [200, 200, 200],  # Carbon
    "O": [240, 0, 0],      # Oxygen
    "H": [248, 248, 248],  # Hydrogen
    "N": [143, 143, 255],  # Nitrogen
    "S": [255, 200, 50],   # Sulphur
    "Cl": [0, 255, 0],     # Chlorine
    "B": [0, 255, 0],      # Boron
    "P": [255, 165, 0],    # Phosphorus
    "Fe": [255, 165, 0],   # Iron
    "Ba": [255, 165, 0],   # Barium
    "Na": [0, 0, 255],     # Sodium
    "Mg": [34, 139, 34],   # Magnesium
    "Zn": [165, 42, 42],   # Zinc
    "Cu": [165, 42, 42],   # Copper
    "Ni": [165, 42, 42],   # Nickel
    "Br": [165, 42, 42],   # Bromine
    "Ca": [128, 128, 144], # Calcium
    "Mn": [128, 128, 144], # Manganese
    "Al": [128, 128, 144], # Aluminum
    "Ti": [128, 128, 144], # Titanium
    "Cr": [128, 128, 144], # Chromium
    "Ag": [128, 128, 144], # Silver
    "F": [218, 165, 32],   # Fluorine
    "Si": [218, 165, 32],  # Silicon
    "Au": [218, 165, 32],  # Gold
    "I": [160, 32, 240],   # Iodine
    "Li": [178, 34, 34],   # Lithium
    "He": [255, 192, 203], # Helium
}
DEFAULT_COLOR = [255, 20, 147] # Default
DEFAULT_RADIUS = 0.1

def create_materials(args, frac_coords, num_atoms, atom_types, lengths, angles, create_xrd=False, symprec=0.01):
    # wavelength
    curr_wavelength = WAVELENGTHS[args.wave_source]
    # Create the XRD calculator
    xrd_calc = XRDCalculator(wavelength=curr_wavelength)
    # get the crystals
    crystals_list = get_crystals_list(frac_coords=frac_coords, atom_types=atom_types, lengths=lengths, angles=angles, num_atoms=num_atoms)
    # ret vals
    all_coords = list()
    all_atom_types = list()
    all_xrds = list()
    # loop through and process the crystals
    for i in tqdm(range(min(args.num_materials, len(crystals_list)))):
        curr_crystal = crystals_list[i]
        curr_structure = Structure(
            lattice=Lattice.from_parameters(
                *(curr_crystal['lengths'].tolist() + curr_crystal['angles'].tolist())),
            species=curr_crystal['atom_types'], coords=curr_crystal['frac_coords'], coords_are_cartesian=False)
        

        curr_coords = list()
        curr_atom_types = list()

        for site in curr_structure:
            curr_coords.append([site.x, site.y, site.z])
            curr_atom_types.append(Element(site.species_string))

        if create_xrd:
            try:
                sga = SpacegroupAnalyzer(curr_structure, symprec=symprec)
                conventional_structure = sga.get_conventional_standard_structure()
            except:
                warnings.warn(f"Failed to get conventional standard structure for material {i}")
                conventional_structure = curr_structure
            # Calculate the XRD pattern
            try:
                pattern = xrd_calc.get_pattern(conventional_structure)
                # Create the XRD tensor
                xrd_tensor = create_xrd_tensor(args, pattern)
            except: 
                warnings.warn(f"Failed to get XRD pattern for material {i}")
                xrd_tensor = torch.zeros(args.xrd_vector_dim)
            all_xrds.append(xrd_tensor)
        
        all_coords.append(np.array(curr_coords))
        all_atom_types.append(curr_atom_types)
    
    truncated_crystals_list = crystals_list[:args.num_materials]
    assert len(all_coords) == len(all_atom_types)
    assert len(all_coords) == min(len(num_atoms), args.num_materials)
    assert len(truncated_crystals_list) == len(all_coords)

    if create_xrd:
        assert len(all_coords) == len(all_xrds)
        all_xrds = torch.stack(all_xrds, dim=0).numpy()
        assert all_xrds.shape == (len(all_coords), 4096)

    return all_coords, all_atom_types, all_xrds, truncated_crystals_list


def sinc_filter(x, sinc_filt):
    filtered = np.convolve(x, sinc_filt, mode='same')
    return filtered

def gaussian_filter(x, n_presubsample, horizontal_noise_range):
    filtered = gaussian_filter1d(x,
                sigma=np.random.uniform(
                    low=n_presubsample * horizontal_noise_range[0], 
                    high=n_presubsample * horizontal_noise_range[1]
                ), 
                mode='constant', cval=0)    
    return filtered

def sample(x, n_postsubsample):
    step_size = int(np.ceil(len(x) / n_postsubsample))
    x_subsample = [np.max(x[i:i+step_size]) for i in range(0, len(x), step_size)]
    return np.array(x_subsample)

def augment_xrdStrip(curr_xrdStrip, sinc_filt, n_presubsample=4096, n_postsubsample=512, horizontal_noise_range=(1e-2, 1.1e-2), vertical_noise=1e-3, xrd_filter='both'):
        """
        Augments curr_xrdStrip via:
        -> Adding peak broadening (horizontal)
        -> Adding small Gaussian perturbations to peaks (vertical)
        """
        xrd = curr_xrdStrip.numpy()
        assert xrd.shape == (n_presubsample,)
        # Peak broadening
        if xrd_filter == 'both':
            sinc_filtered = sinc_filter(xrd, sinc_filt)
            filtered = gaussian_filter(sinc_filtered, n_presubsample, horizontal_noise_range)
            assert filtered.shape == xrd.shape
        elif xrd_filter == 'sinc':
            filtered = sinc_filter(xrd)
            assert filtered.shape == xrd.shape
        else:
            raise ValueError("Invalid filter requested")

        # scale
        filtered = filtered / np.max(filtered)
        filtered = np.maximum(filtered, np.zeros_like(filtered))
        # sample it
        assert filtered.shape == (n_presubsample,)
        assert filtered.shape == curr_xrdStrip.shape
        filtered = sample(filtered)
        # convert to torch
        filtered = torch.from_numpy(filtered)
        assert filtered.shape == (n_postsubsample,)
        # Perturbation
        perturbed = filtered + torch.normal(mean=0, std=vertical_noise, size=filtered.size())
        perturbed = torch.maximum(perturbed, torch.zeros_like(perturbed))
        perturbed = torch.minimum(perturbed, torch.ones_like(perturbed)) # band-pass filter
        return perturbed

# Thanks ChatGPT!
# Function to generate sphere coordinates
def generate_sphere_coordinates(center, radius, n_points=100):
    phi = np.linspace(0, 2 * np.pi, n_points)
    theta = np.linspace(0, np.pi, n_points)
    phi, theta = np.meshgrid(phi, theta)

    x = center[0] + radius * np.sin(theta) * np.cos(phi)
    y = center[1] + radius * np.sin(theta) * np.sin(phi)
    z = center[2] + radius * np.cos(theta)
    
    return x, y, z

# https://stackoverflow.com/a/71053527
def ms(center, radius, n_points=20):
    x, y, z = center
    """Return the coordinates for plotting a sphere centered at (x,y,z)"""
    u, v = np.mgrid[0:2*np.pi:n_points*2j, 0:np.pi:n_points*1j]
    X = radius * np.cos(u)*np.sin(v) + x
    Y = radius * np.sin(u)*np.sin(v) + y
    Z = radius * np.cos(v) + z
    return (X, Y, Z)

def plot_materials(args, the_coords, atom_types, output_dir, batch_idx):
    for i in range(min(len(the_coords), args.num_materials)):
        curr_coords = the_coords[i]
        curr_atom_types = atom_types[i]

        plot_material_single(curr_coords, curr_atom_types, output_dir, idx=i, batch_idx=batch_idx)
    
    return

def plot_xrds(args, xrds, output_dir):
    for i in range(min(args.num_materials, xrds.shape[0])):
        curr_xrd = xrds[i]
        assert curr_xrd.shape == (512,)
        thetas = [pos * 180 / len(curr_xrd) for pos in range(len(curr_xrd))]
        plt.plot(thetas, curr_xrd)
        plt.savefig(os.path.join(output_dir, f'material{i}.png'))
        plt.close()
    return

def plot_xrd_single(args, curr_xrd, output_dir, idx, filename=None, x_axis=None, x_label='2 Theta (degrees)'):
    plt.figure()
    assert curr_xrd.shape == (512,)
    if x_axis is None:
        x_axis = [pos * 180 / len(curr_xrd) for pos in range(len(curr_xrd))]
    plt.figure()
    plt.plot(x_axis, curr_xrd)
    plt.xlabel(x_label)
    plt.ylabel('Scaled Intensity')
    filename = filename if filename is not None else f'material{idx}.png'
    img_path = os.path.join(output_dir, filename)
    plt.savefig(img_path)
    plt.savefig(img_path.replace('.png', '.pdf'))
    plt.close()
    return img_path

def plot_material_single(curr_coords, curr_atom_types, output_dir, idx=0, batch_idx=0, filename=None):
    assert len(curr_atom_types) == len(curr_coords)
    assert len(curr_coords.shape) == 2 and curr_coords.shape[1] == 3

    x = curr_coords[:,0].tolist()
    y = curr_coords[:,1].tolist()
    z = curr_coords[:,2].tolist()

    plot_data = list()
    shown_elements = set()

    elemental_names = [el.symbol for el in curr_atom_types]
    atomic_radii = [float(el.atomic_radius) if el.atomic_radius else DEFAULT_RADIUS for el in curr_atom_types]

    curr_coords = curr_coords.tolist()
    for i in range(len(curr_coords)):
        curr_center = curr_coords[i]
        x, y, z = ms(center=curr_center, radius=atomic_radii[i], n_points=25)
        curr_color = tuple(CPK_COLORS[elemental_names[i]] if elemental_names[i] in CPK_COLORS else DEFAULT_COLOR)
        plot_data.append(
            go.Surface(
                x=x, y=y, z=z,
                opacity=1,
                lighting=dict(ambient=0.9, diffuse=0.5, roughness = 0.5, specular=0.1, fresnel=3),
                showscale=False,
                colorscale=[[0, f'rgb{curr_color}'],[1, f'rgb{curr_color}']],
                name=elemental_names[i],
                showlegend=elemental_names[i] not in shown_elements
            )
        )

    # Plot spheres using Mesh3d
    fig = go.Figure(
        data=plot_data
    )

    camera = dict(
        up=dict(x=0, y=0, z=1),
        center=dict(x=0, y=0, z=0),
        eye=dict(x=1.5, y=1.5, z=1.5)
    )

    # Customize layout
    fig.update_layout(
        title=os.path.split(output_dir)[-1],
        scene=dict(
            xaxis_title='X (Å)',
            yaxis_title='Y (Å)',
            zaxis_title='Z (Å)',
        ),
        margin=dict(l=10, r=10, t=30, b=10),
        scene_camera=camera,
        scene_aspectmode='data'
    )

    filename = filename if filename is not None else f'material{idx}_sample{batch_idx}.png'
    img_path = os.path.join(output_dir, filename)
    fig.write_image(img_path)
    fig.write_image(img_path.replace('.png', '.pdf'))

    return img_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate XRD patterns from CIF descriptions')
    parser.add_argument('--filepath', type=str, help='the file with the predictions from evaluate.py',
                        default='/home/gabeguo/hydra/singlerun/2024-03-01/perov_smoothScaledXRD/eval_recon.pt')
    parser.add_argument('--results_folder', type=str, help='where to save the visualizations',
                        default='material_vis')
    parser.add_argument('--xrd_vector_dim', type=int, help='what dimension are the xrds? (should be 512)',
                        default=512)
    parser.add_argument('--min_theta', type=int,
                        default=0)
    parser.add_argument('--max_theta', type=int, 
                        default=180)
    parser.add_argument('--num_materials', type=int, help='how many materials to visualize?',
                        default=10)
    parser.add_argument('--wave_source', type=str, help='What is the wave source?',
                        default='CuKa')
    parser.add_argument('--task', choices=['recon', 'gen', 'opt'], help='What is the task?',
                        default='recon')

    args = parser.parse_args()

    results = torch.load(args.filepath)

    if args.task == 'recon':
        print([x for x in results])
        for the_dataset, the_name in zip([results, results['input_data_batch']], 
                                        ['pred', 'gt']):

            is_pred = 'pred' in the_name

            batched_frac_coords = the_dataset['frac_coords']
            batched_num_atoms = the_dataset['num_atoms']
            batched_atom_types = the_dataset['atom_types']
            batched_lengths = the_dataset['lengths']
            batched_angles = the_dataset['angles']
            if not is_pred:
                batched_frac_coords = batched_frac_coords.unsqueeze(0)
                batched_num_atoms = batched_num_atoms.unsqueeze(0)
                batched_atom_types = batched_atom_types.unsqueeze(0)
                batched_lengths = batched_lengths.unsqueeze(0)
                batched_angles = batched_angles.unsqueeze(0)

            num_batches = batched_frac_coords.shape[0]
            assert num_batches == batched_num_atoms.shape[0]

            curr_folder = os.path.join(args.results_folder, the_name)

            os.makedirs(curr_folder, exist_ok=True)

            for i in range(num_batches):
                frac_coords = batched_frac_coords[i]
                num_atoms = batched_num_atoms[i]
                atom_types = batched_atom_types[i]
                lengths = batched_lengths[i]
                angles = batched_angles[i]

                the_coords, atom_types, generated_xrds, crystals_list = create_materials(args, 
                        frac_coords, num_atoms, atom_types, lengths, angles, create_xrd=True)
                plot_materials(args, the_coords, atom_types, curr_folder, i)
                xrd_folder = os.path.join(args.results_folder, f"{'pred' if is_pred else 'gt'}_xrds")
                os.makedirs(xrd_folder, exist_ok=True)
                if i == 0:
                    plot_xrds(args, generated_xrds, xrd_folder)

    elif args.task == 'opt':
        the_dataset = results
        # fetch base truth materials
        base_truth = the_dataset['data']
        base_truth_frac_coords = base_truth['frac_coords']
        base_truth_num_atoms = base_truth['num_atoms']
        base_truth_atom_types = base_truth['atom_types']
        base_truth_lengths = base_truth['lengths']
        base_truth_angles = base_truth['angles']
        
        # fetch the optimized materials
        batched_frac_coords = the_dataset['frac_coords']
        batched_num_atoms = the_dataset['num_atoms']
        batched_atom_types = the_dataset['atom_types']
        batched_lengths = the_dataset['lengths']
        batched_angles = the_dataset['angles']

        num_batches = batched_frac_coords.shape[0]
        assert num_batches == batched_num_atoms.shape[0]
        print('num_batches', num_batches)
        
        # create the folders - optimization
        opt_materials_folder = os.path.join(args.results_folder, 'opt_materials')
        os.makedirs(opt_materials_folder, exist_ok=True)
        pred_xrd_folder = os.path.join(args.results_folder, 'opt_xrds')
        os.makedirs(pred_xrd_folder, exist_ok=True)

        # create the folders - base truth
        base_truth_folder = os.path.join(args.results_folder, 'base_truth_materials')
        os.makedirs(base_truth_folder, exist_ok=True)
        base_truth_xrd_folder = os.path.join(args.results_folder, 'base_truth_xrds')
        os.makedirs(base_truth_xrd_folder, exist_ok=True)
        xrds = results['xrds'].cpu().squeeze().numpy()
        os.makedirs(base_truth_xrd_folder, exist_ok=True)
        plot_xrds(args, xrds, base_truth_xrd_folder)

        for i in range(num_batches):
            frac_coords = batched_frac_coords[i]
            num_atoms = batched_num_atoms[i]
            atom_types = batched_atom_types[i]
            lengths = batched_lengths[i]
            angles = batched_angles[i]

            # predictions
            the_coords, atom_types, generated_xrds, crystals_list = create_materials(args, 
                    frac_coords, num_atoms, atom_types, lengths, angles, create_xrd=True)

            plot_materials(args, the_coords, atom_types, opt_materials_folder, i)
            # apply gaussian smoothing to the XRDs (to match base-truth gaussian smoothed xrds)
            smoothed_xrds = list()
            for i in range(generated_xrds.shape[0]):
                smoothed_xrd = augment_xrdStrip(torch.tensor(generated_xrds[i,:]))
                smoothed_xrds.append(smoothed_xrd)
            generated_xrds = torch.stack(smoothed_xrds, dim=0).numpy()
            plot_xrds(args, generated_xrds, pred_xrd_folder)

            # ground truth
            the_coords, atom_types, generated_xrds, crystals_list = create_materials(args, 
                    base_truth_frac_coords, base_truth_num_atoms, base_truth_atom_types, base_truth_lengths, base_truth_angles, create_xrd=True)
            plot_materials(args, the_coords, atom_types, base_truth_folder, i)
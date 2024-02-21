import torch
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from pymatgen.core.periodic_table import Element
from pymatgen.core.structure import Structure
from pymatgen.core.lattice import Lattice
from scripts.eval_utils import get_crystals_list
import warnings
import os

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
RESULTS_FOLDER = 'dummy_vis'

def create_materials(frac_coords, num_atoms, atom_types, lengths, angles):
    crystals_list = get_crystals_list(frac_coords=frac_coords, atom_types=atom_types, lengths=lengths, angles=angles, num_atoms=num_atoms)
    all_coords = list()
    all_atom_types = list()
    for curr_crystal in tqdm(crystals_list):
        curr_structure = Structure(
            lattice=Lattice.from_parameters(
                *(curr_crystal['lengths'].tolist() + curr_crystal['angles'].tolist())),
            species=curr_crystal['atom_types'], coords=curr_crystal['frac_coords'], coords_are_cartesian=False)
        print(curr_crystal['angles'].tolist())
        curr_coords = list()
        curr_atom_types = list()

        for site in curr_structure:
            curr_coords.append([site.x, site.y, site.z])
            curr_atom_types.append(Element(site.species_string))
        
        all_coords.append(np.array(curr_coords))
        all_atom_types.append(curr_atom_types)
    
    assert len(all_coords) == len(all_atom_types)
    assert len(all_coords) == len(num_atoms)

    return all_coords, all_atom_types

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

def plot_materials(the_coords, atom_types, output_dir, num_materials=5):
    for i in range(min(len(the_coords), num_materials)):
        curr_coords = the_coords[i]
        curr_atom_types = atom_types[i]

        plot_material_single(curr_coords, curr_atom_types, output_dir, idx=i)
    
    return

def plot_material_single(curr_coords, curr_atom_types, output_dir, idx=0):
    print(curr_coords)
    print(curr_atom_types)
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
        title='moo',
        scene=dict(
            xaxis_title='X (Å)',
            yaxis_title='Y (Å)',
            zaxis_title='Z (Å)',
        ),
        margin=dict(l=10, r=10, t=30, b=10),
        scene_camera=camera,
        scene_aspectmode='data'
    )

    print('moo')

    fig.write_image(os.path.join(output_dir, f'dummy{idx}.png'))

    return

if __name__ == "__main__":
    filepath = '/home/gabeguo/hydra/singlerun/2024-02-16/mp_20/eval_recon.pt'

    results = torch.load(filepath)

    print(len(results))

    for item in results:
        print(item)

    print('frac_coords', results['frac_coords'].shape)
    print('num_atoms', results['num_atoms'].shape)
    print('atom_types', results['atom_types'].shape)
    print('lengths', results['lengths'].shape)
    print('angles', results['angles'].shape)

    frac_coords = results['frac_coords'].squeeze()
    num_atoms = results['num_atoms'].squeeze()
    atom_types = results['atom_types'].squeeze()
    lengths = results['lengths'].squeeze()
    angles = results['angles'].squeeze()

    os.makedirs(RESULTS_FOLDER, exist_ok=True)

    the_coords, atom_types = create_materials(frac_coords, num_atoms, atom_types, lengths, angles)
    plot_materials(the_coords, atom_types, RESULTS_FOLDER)
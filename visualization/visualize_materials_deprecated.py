import torch
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from pymatgen.core.periodic_table import Element


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

# Thanks https://chemistry.stackexchange.com/questions/136836/converting-fractional-coordinates-into-cartesian-coordinates-for-crystallography
# Thanks https://www.ucl.ac.uk/~rmhajc0/frorth.pdf
def generate_transform_matrix(a, b, c, alpha, beta, gamma):
    alpha *= np.pi / 180
    beta *= np.pi / 180
    gamma *= np.pi / 180
    n2 = (np.cos(alpha)-np.cos(gamma)*np.cos(beta))/np.sin(gamma)
    M = np.array([
        [a,0,0],
        [b*np.cos(gamma),b*np.sin(gamma),0], 
        [c*np.cos(beta),c*n2,c*np.sqrt(np.sin(beta)**2-n2**2)]
    ])
    return M # left-multiply coordinates

def create_materials(frac_coords, num_atoms, atom_types, lengths, angles):
    print('creating materials')
    the_coords = list()
    the_atom_types = list()

    num_atoms = num_atoms.tolist()
    atom_types = [Element.from_Z(el) for el in atom_types.tolist()]
    print(len(atom_types))
    start_idx = 0

    nan_count = 0
    for i in tqdm(range(len(num_atoms))):
        curr_num_atoms = num_atoms[i]
        # take these atoms
        low = start_idx
        high = start_idx + curr_num_atoms
        curr_coords = frac_coords[low:high]
        assert curr_coords.shape == (curr_num_atoms, 3)
        curr_elements = atom_types[low:high]
        
        # change start idx
        start_idx += curr_num_atoms

        # calculate cartesian coordinates
        a, b, c = tuple(lengths[i].tolist())
        alpha, beta, gamma = tuple(angles[i].tolist())
        transform_matrix = generate_transform_matrix(a=a, b=b, c=c, alpha=alpha, beta=beta, gamma=gamma)
        curr_coords = curr_coords.numpy() @ transform_matrix
        if np.any(np.isnan(curr_coords)):
            nan_count += 1
        curr_coords = np.nan_to_num(curr_coords)

        # add materials
        the_coords.append(curr_coords)
        the_atom_types.append(curr_elements)

        assert len(curr_coords) == len(curr_elements)

    print(f'{nan_count} out of {len(the_coords)} nan')
    assert len(the_coords) == len(the_atom_types)

    return the_coords, the_atom_types

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

def plot_materials(the_coords, atom_types, num_materials=5):
    for i in range(min(len(the_coords), num_materials)):
        curr_coords = the_coords[i]
        curr_atom_types = atom_types[i]

        plot_material_single(curr_coords, curr_atom_types, idx=i)
    
    return

def plot_material_single(curr_coords, curr_atom_types, idx=0):
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

    fig.write_image(f'/home/gabeguo/cdvae/dummy_vis/dummy_alt{idx}.png')

    return

if __name__ == "__main__":
    filepath = '/home/gabeguo/hydra/singlerun/2024-02-16/carbon/eval_recon.pt'

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

    the_coords, atom_types = create_materials(frac_coords, num_atoms, atom_types, lengths, angles)
    plot_materials(the_coords, atom_types)
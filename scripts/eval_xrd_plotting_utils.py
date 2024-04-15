import matplotlib.pyplot as plt
import plotly.graph_objects as go
import numpy as np
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

def xrd_torch_to_np(the_xrd):
    if not isinstance(the_xrd, np.ndarray):
        the_xrd = the_xrd.detach().cpu().numpy()
        assert the_xrd.shape == (512, 1)
        the_xrd = the_xrd.reshape((512,))
    assert the_xrd.shape == (512,)
    return the_xrd

"""
Plotting Code for XRD
"""
# Thanks ChatGPT!
# If you want to change the colors of the lines and shades, simply modify in the ax.fill_between() and ax.plot() functions
# A list of possible colors can be found at: https://matplotlib.org/stable/gallery/color/named_colors.html
def plot_overlaid_graphs(xrd_a, xrd_b, xrd_a_label, xrd_b_label, Qs, savepath):
    fig, ax = plt.subplots()

    xrd_a = xrd_torch_to_np(xrd_a)
    xrd_b = xrd_torch_to_np(xrd_b)

    # Plot and fill the area under the first curve
    ax.fill_between(Qs, xrd_a, color="mistyrose", alpha=0.2)
    ax.plot(Qs, xrd_a, color="red", alpha=0.6, linestyle='dotted', linewidth=2, label=xrd_a_label)  # Dotted curve line with increased linewidth

    # Plot and fill the area under the second curve
    ax.fill_between(Qs, xrd_b, color="lightgreen", alpha=0.2)
    ax.plot(Qs, xrd_b, color="green", alpha=0.6, linestyle='dashed', linewidth=2, label=xrd_b_label)  # Dotted curve line with increased linewidth

    # Customizing the plot
    ax.set_title("XRD Patterns")
    ax.set_xlabel(r'$Q (\mathring A^{-1})$')
    ax.set_ylabel("Scaled Intensity")
    # ax.set_xlim(0, 180)  # Set x-axis limits
    ax.set_ylim(0, 1)  # Set y-axis limits
    # ax.set_xticks(np.arange(0, 181, 10)) 
    # ax.set_xticklabels(ax.get_xticks(), rotation=70)  # Rotate x-axis labels by 70 degrees
    ax.set_yticks(np.arange(0, 1.1, 0.1))  # Set horizontal gridlines every 0.1 from 0 to 1
    ax.grid(True)  # Show gridlines
    ax.legend()

    # Display the plot
    #plt.show()
    plt.tight_layout()
    plt.savefig(savepath)
    plt.savefig(savepath.replace('.png', '.pdf'))
    plt.close()

    return
   
"""
Plotting code for material
"""
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

# https://stackoverflow.com/a/71053527
def ms(center, radius, n_points=20):
    x, y, z = center
    """Return the coordinates for plotting a sphere centered at (x,y,z)"""
    u, v = np.mgrid[0:2*np.pi:n_points*2j, 0:np.pi:n_points*1j]
    X = radius * np.cos(u)*np.sin(v) + x
    Y = radius * np.sin(u)*np.sin(v) + y
    Z = radius * np.cos(v) + z
    return (X, Y, Z)


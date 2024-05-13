import numpy as np
import os
from scripts.gen_xrd import create_xrd_tensor
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.core.structure import Structure
from scipy.ndimage import gaussian_filter1d
from pymatgen.analysis.diffraction.xrd import XRDCalculator, WAVELENGTHS
import torch
from types import SimpleNamespace
import matplotlib.pyplot as plt
from pymatgen.io.cif import CifParser
from tqdm import tqdm

def create_sinc_filter(wave_source, min_2_theta, max_2_theta, xrd_vector_dim, nanomaterial_size):
    wavelength = WAVELENGTHS[wave_source]

    min_theta = min_2_theta / 2
    max_theta = max_2_theta / 2
    Q_min = 4 * np.pi * np.sin(np.radians(min_theta)) / wavelength
    Q_max = 4 * np.pi * np.sin(np.radians(max_theta)) / wavelength

    # phase shift for sinc filter = half of the signed Q range
    phase_shift = (Q_max - Q_min) / 2

    # compute Qs
    Qs = np.linspace(Q_min, Q_max, xrd_vector_dim)
    Qs_shifted = Qs - phase_shift   
    the_filter = nanomaterial_size * np.sinc((np.pi * nanomaterial_size * Qs_shifted)/np.pi)

    return Qs, the_filter

def apply_filter(x, the_filter):
    return np.convolve(x, the_filter, mode='same')

def post_process_filtered_xrd(filtered):
    # scale
    filtered = filtered / np.max(filtered)
    filtered = np.maximum(filtered, np.zeros_like(filtered))
    filtered = np.minimum(filtered, np.ones_like(filtered)) # band-pass filter
    return filtered

def create_xrd_args(wave_source, xrd_vector_dim, min_2_theta, max_2_theta):
    alt_args = SimpleNamespace()
    alt_args.wave_source = wave_source
    alt_args.xrd_vector_dim = xrd_vector_dim
    alt_args.max_theta = max_2_theta # misnomer
    alt_args.min_theta = min_2_theta # misnomer

    return alt_args

def generate_xrd(wave_source, curr_structure, symprec, xrd_vector_dim, min_2_theta, max_2_theta):
    curr_wavelength = WAVELENGTHS[wave_source]
    # Create the XRD calculator
    xrd_calc = XRDCalculator(wavelength=curr_wavelength)
    try:
        sga = SpacegroupAnalyzer(curr_structure, symprec=symprec)
        conventional_structure = sga.get_conventional_standard_structure()
    except:
        print(f"Failed to get conventional standard structure for material")
        conventional_structure = curr_structure
    # Calculate the XRD pattern
    try:
        pattern = xrd_calc.get_pattern(conventional_structure)
        # Create the XRD tensor
        xrd_tensor = create_xrd_tensor(create_xrd_args(wave_source=wave_source, 
                                                       xrd_vector_dim=xrd_vector_dim,
                                                       min_2_theta=min_2_theta,
                                                       max_2_theta=max_2_theta), pattern)
    except: 
        print(f"Failed to get XRD pattern for material")
        xrd_tensor = torch.zeros(xrd_vector_dim)
    return xrd_tensor

def read_structure(filepath):
    parser = CifParser(filepath)
    structure = parser.get_structures()[0]
    return structure

def convert_and_save_q_to_2theta(q_xrd, Qs, output_filepath_without_ext,
                        wavelength, plot=False):
    assert len(q_xrd.shape) == 1
    _2thetas = np.degrees(2 * np.arcsin(Qs * wavelength / (4 * np.pi)))
    intensities = q_xrd

    with open(f"{output_filepath_without_ext}.xy", 'w') as fout:
        for i in range(len(intensities)):
            fout.write(f'{_2thetas[i]} {intensities[i]}\n')

    plt.grid()
    plt.ylim(0, 1)
    plt.yticks(np.linspace(0, 1, 11))
    plt.plot(_2thetas, intensities)
    plt.savefig(f'{output_filepath_without_ext}_2theta.png')
    plt.close()

    plt.grid()
    plt.ylim(0, 1)
    plt.yticks(np.linspace(0, 1, 11))
    plt.plot(Qs, intensities)
    plt.savefig(f'{output_filepath_without_ext}_q.png')
    plt.close()

    return

def generate_and_save_xrd_for_cif_file(cif_filepath, output_filepath_without_ext,
                                       wave_source, xrd_vector_dim, min_2_theta, max_2_theta,
                                       nanomaterial_size, plot_xrd,
                                       broadening='sinc',
                                       gaussian_sigma_frac=5e-3):
    curr_structure = read_structure(cif_filepath)
    ideal_xrd_tensor = generate_xrd(wave_source=wave_source, 
                                    curr_structure=curr_structure, 
                                    symprec=0.01, 
                                    xrd_vector_dim=xrd_vector_dim, 
                                    min_2_theta=min_2_theta, 
                                    max_2_theta=max_2_theta)
    Qs, sinc_filter = create_sinc_filter(wave_source=wave_source, 
                                        min_2_theta=min_2_theta, 
                                        max_2_theta=max_2_theta, 
                                        xrd_vector_dim=xrd_vector_dim, 
                                        nanomaterial_size=nanomaterial_size)
    if broadening == 'sinc':
        noised_xrd_array = apply_filter(ideal_xrd_tensor.numpy(), sinc_filter)
    elif broadening == 'gaussian':
        noised_xrd_array = gaussian_filter1d(ideal_xrd_tensor.numpy(),
                    sigma=int(len(Qs) * gaussian_sigma_frac), 
                    mode='constant', cval=0)    
    else:
        raise ValueError(f'{broadening} invalid')
    noised_xrd_array = post_process_filtered_xrd(noised_xrd_array)
    noised_xrd_tensor = torch.tensor(noised_xrd_array)

    torch.save(noised_xrd_tensor, output_filepath_without_ext + '.pt')
    convert_and_save_q_to_2theta(q_xrd=noised_xrd_array, Qs=Qs, 
                                 output_filepath_without_ext=output_filepath_without_ext,
                                 wavelength=WAVELENGTHS[wave_source],
                                 plot=plot_xrd)
    
    return

def save_xrds_for_pregenerated_results(input_dirpath, sinc_size, filter='sinc', 
                                            gaussian_sigma_frac=0.01):
    materials = [x for x in os.listdir(input_dirpath) if 'material' in x]
    for curr_material in tqdm(materials):
        material_num = curr_material.split('_')[0]
        assert 'material' in material_num

        curr_gt_cif_path = os.path.join(input_dirpath, curr_material, 'gt',
            'cif', f'noSpacegroup_{curr_material}.cif')
        assert os.path.exists(curr_gt_cif_path)
        numerical_descriptor = sinc_size if filter == 'sinc' else gaussian_sigma_frac
        desired_gt_xrd_filepath_without_ext = os.path.join(input_dirpath, curr_material, 'gt',
            'xrd', 
            f'hiRes_{filter}_{numerical_descriptor}_{curr_material}')
        generate_and_save_xrd_for_cif_file(
            cif_filepath=curr_gt_cif_path, 
            output_filepath_without_ext=desired_gt_xrd_filepath_without_ext,
            wave_source='CuKa', 
            xrd_vector_dim=8192, 
            min_2_theta=0, 
            max_2_theta=180,
            nanomaterial_size=sinc_size, 
            plot_xrd=True,
            broadening=filter,
            gaussian_sigma_frac=gaussian_sigma_frac
        )

        for pred_idx in range(5):
            curr_pred_cif_path = os.path.join(input_dirpath, curr_material, 
                'pred', f'candidate{pred_idx}', 'cif', f'noSpacegroup_{material_num}_candidate{pred_idx}.cif')
            assert os.path.exists(curr_pred_cif_path)
            desired_pred_filepath_without_ext = os.path.join(input_dirpath, curr_material, 
                'pred', f'candidate{pred_idx}', 'xrd_opt_gen', 
                f'{filter}_{numerical_descriptor}_{curr_material}')
            generate_and_save_xrd_for_cif_file(
                cif_filepath=curr_pred_cif_path, 
                output_filepath_without_ext=desired_pred_filepath_without_ext,
                wave_source='CuKa', 
                xrd_vector_dim=8192, 
                min_2_theta=0, 
                max_2_theta=180,
                nanomaterial_size=sinc_size, 
                plot_xrd=True,
                broadening=filter,
                gaussian_sigma_frac=gaussian_sigma_frac
            )
    return

def save_sinc_xrds_for_curated_results(input_dirpath):
    for root, dirs, files in os.walk(input_dirpath):
        for file in files:
            filepath = os.path.join(root, file)
            if '.cif' not in filepath:
                continue
            if 'gt' in filepath:
                for sinc_size in [10, 100]:
                    generate_and_save_xrd_for_cif_file(
                        cif_filepath=filepath, 
                        output_filepath_without_ext=os.path.join(root, f'sinc{sinc_size}'),
                        wave_source='CuKa', 
                        xrd_vector_dim=8092, 
                        min_2_theta=0, 
                        max_2_theta=180,
                        nanomaterial_size=sinc_size, 
                        plot_xrd=True
                    )
    return

if __name__ == "__main__":
    for path in [
        '/home/gabeguo/cdvae_xrd/paper_results_PRELIM/_random_baseline_sinc10_',
        '/home/gabeguo/cdvae_xrd/paper_results_PRELIM/_sinc10_baseline_noOpt',
        '/home/gabeguo/cdvae_xrd/paper_results_PRELIM/_sinc10_replicate',
        '/home/gabeguo/cdvae_xrd/paper_results_PRELIM/_random_baseline_EXPERIMENTAL_',
        '/home/gabeguo/cdvae_xrd/paper_results_PRELIM/experimental_sinc10Filter_',
        '/home/gabeguo/cdvae_xrd/paper_results_PRELIM/experimental_baseline_noOpt'
    ]:
        save_xrds_for_pregenerated_results(
            input_dirpath=path,
            sinc_size=10,
            filter='gaussian',
            gaussian_sigma_frac=5e-3
        )

    for path in [
        '/home/gabeguo/cdvae_xrd/paper_results_PRELIM/_random_baseline_sinc100_',
        '/home/gabeguo/cdvae_xrd/paper_results_PRELIM/_sinc100_baseline_noOpt',
        '/home/gabeguo/cdvae_xrd/paper_results_PRELIM/_sinc100_'
    ]:
        save_xrds_for_pregenerated_results(
            input_dirpath=path,
            sinc_size=100,
            filter='gaussian',
            gaussian_sigma_frac=5e-3
        )
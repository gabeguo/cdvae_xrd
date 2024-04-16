import time
import argparse
import torch
import os

from tqdm import tqdm
from torch.optim import Adam
import torch.nn.functional as F
from pathlib import Path
from types import SimpleNamespace
from torch_geometric.data import Batch
from torch_geometric.data import DataLoader

from eval_utils import load_model, get_crystals_list
from cdvae.pl_data.dataset import CrystDataset
from cdvae.common.data_utils import get_scaler_from_data_list
from compute_metrics import Crystal, RecEval, GenEval

from pymatgen.core.periodic_table import Element
from pymatgen.core.structure import Structure
from pymatgen.core.lattice import Lattice
from pymatgen.analysis.diffraction.xrd import XRDCalculator, WAVELENGTHS
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.io.cif import CifParser
from pymatgen.io.cif import CifWriter


from scripts.gen_xrd import create_xrd_tensor
import numpy as np

import warnings
from tqdm import tqdm

import eval_xrd_plotting_utils as pu

"""
Reconstruction code
"""

def reconstruct_all(args, loader, model, ld_kwargs, num_evals,
                    down_sample_traj_step=1, model_path=None):
    for idx, batch in tqdm(enumerate(loader)):
        ####
        # Set Up
        ####     
        assert len(batch.mpid) == 1
        curr_mpid = batch.mpid[0]
        curr_formula = batch.pretty_formula[0]

        curr_folder = os.path.join(args.output_dir, f'material{idx}_{curr_mpid}_{curr_formula}')
    
        ####
        # Base Truth
        ####
        gt_frac_coords = batch.frac_coords
        gt_num_atoms = batch.num_atoms
        gt_atom_types = batch.atom_types
        gt_lengths = batch.lengths
        gt_angles = batch.angles
        noisy_given_xrd = batch.y
        gt_raw_xrd = batch.raw_xrd

        assert len(batch.cif) == 1
        gt_cif_parser = CifParser.from_str(batch.cif[0])
        gt_cif = gt_cif_parser.get_structures()
        assert len(gt_cif) == 1
        gt_cif = gt_cif[0]

        gt_cif_folder = os.path.join(curr_folder, 'gt', 'cif')
        os.makedirs(gt_cif_folder, exist_ok=True)
        gt_cif_writer = CifWriter(gt_cif, symprec=1e-4)
        gt_cif_writer.write_file(filename=f'{gt_cif_folder}/material{idx}_{curr_mpid}_{curr_formula}_gt.cif')

        assert gt_num_atoms.shape == (1,)
        assert len(gt_frac_coords.shape) == 2
        assert gt_frac_coords.shape[0] == gt_atom_types.shape[0]

        # Save XRD image
        sample_factor = loader.dataset.n_presubsample // loader.dataset.n_postsubsample
        downsampled_Qs = loader.dataset.Qs[::sample_factor]
        gt_xrd_folder = os.path.join(curr_folder, 'gt', 'xrd')
        os.makedirs(gt_xrd_folder, exist_ok=True)
        curr_xrd_folder = os.path.join(gt_xrd_folder, f'xrd{idx}_{curr_mpid}_{curr_formula}')
        os.makedirs(curr_xrd_folder, exist_ok=True)
        pu.plot_overlaid_graphs(xrd_a=noisy_given_xrd, xrd_b=gt_raw_xrd, 
                                xrd_a_label='GT XRD (noised)', xrd_b_label='GT XRD (no noise)', 
                                Qs=downsampled_Qs, savepath=f'{curr_xrd_folder}/xrd{idx}_{curr_mpid}_{curr_formula}_gt.png')
        torch.save(noisy_given_xrd, f'{curr_xrd_folder}/xrd{idx}_{curr_mpid}_{curr_formula}_gtNoisy.pt')
        torch.save(gt_raw_xrd, f'{curr_xrd_folder}/xrd{idx}_{curr_mpid}_{curr_formula}_gtNoiseless.pt')
        # Save structure image
        print('shape of coords:', gt_frac_coords.shape)
        print('shape of atom types:', gt_atom_types.shape)
        print('shape of raw xrd:', gt_raw_xrd.shape)
        print('shape of noised xrd:', noisy_given_xrd.shape)
        gt_img_folder = os.path.join(curr_folder, 'gt', 'vis')
        os.makedirs(gt_img_folder, exist_ok=True)
        gt_cart_coords, gt_str_atom_types, _, _ = create_materials(args=args, 
                frac_coords=gt_frac_coords, num_atoms=gt_num_atoms, atom_types=gt_atom_types, 
                lengths=gt_lengths, angles=gt_angles, create_xrd=False, symprec=1e-3)
        # TODO: compare this XRD to GT
        # TODO: check create_materials
        cart_gt_coords = np.array(gt_cart_coords)[0]
        str_gt_atom_types = np.array(gt_str_atom_types)[0]
        pu.plot_material_single(curr_coords=cart_gt_coords, curr_atom_types=str_gt_atom_types, output_dir=gt_img_folder, 
                                filename=f'structureVis{idx}_{curr_mpid}_{curr_formula}_gt.png')

        ####
        # Predicted materials
        ####
        (pred_frac_coords, pred_num_atoms, pred_atom_types, pred_lengths, pred_angles) = reconstruction(
                idx=idx, batch=batch, model=model, ld_kwargs=ld_kwargs, num_evals=args.num_evals,
                down_sample_traj_step=args.down_sample_traj_step, model_path=args.model_path)
        pred_coords, pred_atom_types, pred_generated_xrds, pred_crystal_list = create_materials(
            args=args, frac_coords=pred_frac_coords, num_atoms=pred_num_atoms,
            atom_types=pred_atom_types, lengths=pred_lengths, angles=pred_angles,
            create_xrd=True, symprec=1e-3)
        # See candidates
        assert len(pred_crystal_list) == num_evals
        for eval_idx, the_sample in enumerate(pred_crystal_list):
            the_crystal = Crystal(the_sample)
            the_generated_xrd = pred_generated_xrds[eval_idx]
            pred_cif_folder = os.path.join(curr_folder, 'pred', f'candidate{eval_idx}', 'cif')
            os.makedirs(pred_cif_folder, exist_ok=True)
            pred_cif_writer = CifWriter(the_crystal.structure, symprec=1e-3)
            pred_cif_writer.write_file(filename=f'{pred_cif_folder}/material{idx}_{curr_mpid}_{curr_formula}_pred{eval_idx}.cif')
            pred_xrd_folder = os.path.join(curr_folder, 'pred', f'candidate{eval_idx}', 'xrd')
            os.makedirs(pred_xrd_folder, exist_ok=True)
            pu.plot_overlaid_graphs(xrd_a=the_generated_xrd, xrd_b=gt_raw_xrd, 
                xrd_a_label='Predicted XRD', xrd_b_label='GT XRD (no noise)', 
                Qs=downsampled_Qs, savepath=f'{pred_xrd_folder}/xrd{idx}_{curr_mpid}_{curr_formula}_pred{eval_idx}.png')
            torch.save(the_generated_xrd, f'{pred_xrd_folder}/xrd{idx}_{curr_mpid}_{curr_formula}_pred{eval_idx}.pt')
            pred_img_folder = os.path.join(curr_folder, 'pred', f'candidate{eval_idx}', 'vis')
            os.makedirs(pred_img_folder, exist_ok=True)
            pu.plot_material_single(curr_coords=pred_coords[eval_idx], curr_atom_types=pred_atom_types[eval_idx], output_dir=pred_img_folder, 
                filename=f'structureVis{idx}_{curr_mpid}_{curr_formula}_pred{eval_idx}.png')
        
def reconstruction(idx, batch, model, ld_kwargs, num_evals,
                  down_sample_traj_step=1, model_path=None):
    """
    reconstruct the crystals in <loader>.
    """
    if torch.cuda.is_available():
        batch.cuda()

    # Note that the z comes from XRD
    # only sample one z, multiple evals for stoichaticity in langevin dynamics
    _, _, z = model.encode(batch)

    print('latent shape', z.shape)
    z = z.expand(num_evals, -1)
    assert z.shape == (num_evals, 256)
    init_num_atoms = batch.num_atoms.repeat(num_evals) # PyG wants them sequentially
    init_atom_types = batch.atom_types.repeat(num_evals) # PyG wants the sequentially
    assert len(init_num_atoms.shape) == 1
    assert len(init_atom_types.shape) == 1
    outputs = model.langevin_dynamics(z, ld_kwargs, init_num_atoms, init_atom_types)

    # collect sampled crystals in this batch.
    frac_coords = outputs['frac_coords'].detach().cpu()
    num_atoms = outputs['num_atoms'].detach().cpu()
    atom_types = outputs['atom_types'].detach().cpu()
    lengths = outputs['lengths'].detach().cpu()
    angles = outputs['angles'].detach().cpu()

    print('frac coords shape', frac_coords.shape) # (num_crystals * atoms_per_crystal, 3)
    print('num atoms shape:', num_atoms.shape) # (num_crystals,)
    print('atom types shape:', atom_types.shape) # (num_crystals * atoms_per_crystal,)
    print('lengths shape:', lengths.shape) # (num_crystals, 3)
    print('angles shape:', angles.shape) # (num_crystals, 3)

    ret_val = [
        frac_coords, num_atoms, atom_types, lengths, angles]
    return ret_val

def create_materials(args, frac_coords, num_atoms, atom_types, lengths, angles, create_xrd=False, symprec=0.01):
    # wavelength
    curr_wavelength = WAVELENGTHS[args.wave_source]
    # Create the XRD calculator
    xrd_calc = XRDCalculator(wavelength=curr_wavelength)
    # get the crystals
    print(frac_coords.shape, atom_types.shape, num_atoms)
    crystals_list = get_crystals_list(frac_coords=frac_coords, atom_types=atom_types, lengths=lengths, angles=angles, num_atoms=num_atoms)
    # ret vals
    all_coords = list()
    all_atom_types = list()
    all_xrds = list()
    # loop through and process the crystals
    for i in tqdm(range(len(crystals_list))):
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
            pattern = xrd_calc.get_pattern(conventional_structure)
            # Create the XRD tensor
            xrd_tensor = create_xrd_tensor(args, pattern)
            all_xrds.append(xrd_tensor)
        
        all_coords.append(np.array(curr_coords))
        all_atom_types.append(curr_atom_types)
    
    assert len(all_coords) == len(all_atom_types)
    assert len(all_coords) == len(num_atoms)
    assert len(crystals_list) == len(all_coords)

    if create_xrd:
        assert len(all_coords) == len(all_xrds)
        all_xrds = torch.stack(all_xrds, dim=0).numpy()
        assert all_xrds.shape == (len(all_coords), 512)

    return np.array(all_coords), np.array(all_atom_types), all_xrds, crystals_list

# TODO: metrics

def main(args):
    # load_data if do reconstruction.
    model_path = Path(args.model_path)
    model, test_loader, cfg = load_model(
        model_path, load_data=True, batch_size=1)
    model.eval()
    ld_kwargs = SimpleNamespace(n_step_each=args.n_step_each,
                                step_lr=args.step_lr,
                                min_sigma=args.min_sigma,
                                save_traj=args.save_traj,
                                disable_bar=args.disable_bar)
    
    if torch.cuda.is_available():
        model.to('cuda')

    print('Evaluate model on the reconstruction task.')

    reconstruct_all(args=args, loader=test_loader, model=model,
                    ld_kwargs=ld_kwargs, num_evals=args.num_evals,
                    down_sample_traj_step=args.down_sample_traj_step, model_path=args.model_path)

    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--output_dir', default='eval_xrd_output', type=str)
    parser.add_argument('--wave_source', default='CuKa', type=str)
    parser.add_argument('--xrd_vector_dim', default=512, type=int)
    parser.add_argument('--min_theta', default=0, type=int, help='Actually 2 theta, but for legacy reasons, do not change')
    parser.add_argument('--max_theta', default=180, type=int, help='Actually 2 theta, but for legacy reasons, do not change')
    parser.add_argument('--data_dir', default='data', type=str)
    parser.add_argument('--n_step_each', default=100, type=int)
    parser.add_argument('--step_lr', default=1e-4, type=float)
    parser.add_argument('--min_sigma', default=0, type=float)
    parser.add_argument('--save_traj', default=False, type=bool)
    parser.add_argument('--disable_bar', default=False, type=bool)
    parser.add_argument('--num_evals', default=2, type=int)
    parser.add_argument('--start_from', default='data', type=str)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--down_sample_traj_step', default=10, type=int)

    args = parser.parse_args()
    
    assert args.batch_size == 1

    print('starting eval', args)
    main(args)

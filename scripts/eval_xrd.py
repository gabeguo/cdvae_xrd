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
        assert batch.shape[0] == 1
        (frac_coords, num_atoms, atom_types, lengths, angles,
            all_frac_coords_stack, all_atom_types_stack, input_data_batch) = reconstruction(
            loader, model, ld_kwargs, args.num_evals,
            args.down_sample_traj_step, args.model_path)
        
        curr_mpid = batch.mpid[0]
        curr_formula = batch.pretty_formula[0]

        curr_folder = os.path.join(args.output_dir, f'material{idx}_{curr_mpid}_{curr_formula}')
    
        ####
        # Base Truth
        ####
        gt_coords = batch.frac_coords
        gt_num_atoms = batch.num_atoms
        gt_atom_types = batch.atom_types
        gt_lengths = batch.lengths
        gt_angles = batch.angles
        noisy_given_xrd = batch.y
        gt_raw_xrd = batch.raw_xrd
        gt_cif = batch.cif

        gt_cif_folder = os.path.join(curr_folder, 'gt', 'cif')
        os.makedirs(gt_cif_folder, exist_ok=True)
        gt_cif_writer = CifWriter(gt_cif, symprec=1e-4)
        gt_cif_writer.write_file(filename=f'{gt_cif_folder}/material{idx}_{curr_mpid}_{curr_formula}_gt.cif')

        assert gt_num_atoms.shape[0] == 1
        assert gt_coords.shape[0] == gt_atom_types.shape[0]

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
        print('shape of coords:', gt_coords.shape)
        print('shape of atom types:', gt_atom_types.shape)
        gt_img_folder = os.path.join(curr_folder, 'gt', 'vis')
        os.makedirs(gt_img_folder, exist_ok=True)
        pu.plot_material_single(curr_coords=gt_coords, curr_atom_types=gt_atom_types, output_dir=gt_img_folder, 
                                filename=f'structureVis{idx}_{curr_mpid}_{curr_formula}_gt.png')

        ####
        # Predicted materials
        ####
        pred_coords, pred_atom_types, pred_generated_xrds, pred_crystal_list = create_materials(
            args=args, frac_coords=frac_coords, num_atoms=num_atoms,
            atom_types=atom_types, lengths=lengths, angles=angles,
            create_xrd=True, symprec=0.01)
        
        assert len(pred_crystal_list) == num_evals
        for eval_idx, the_sample in enumerate(num_evals):
            the_crystal = Crystal(the_sample)
            the_generated_xrd = pred_generated_xrds[eval_idx]
            pred_cif_folder = os.path.join(curr_folder, 'pred', f'candidate{eval_idx}', 'cif')
            os.makedirs(pred_cif_folder)
            pred_cif_writer = CifWriter(the_crystal.structure, symprec=1e-3)
            pred_cif_writer.write_file(filename=f'{pred_cif_folder}/material{idx}_{curr_mpid}_{curr_formula}_pred{eval_idx}.cif')
            pred_xrd_folder = os.path.join(curr_folder, 'pred', f'candidate{eval_idx}', 'xrd')
            os.makedirs(pred_xrd_folder)
            pu.plot_overlaid_graphs(xrd_a=the_generated_xrd, xrd_b=gt_raw_xrd, 
                xrd_a_label='Predicted XRD', xrd_b_label='GT XRD (no noise)', 
                Qs=downsampled_Qs, savepath=f'{pred_xrd_folder}/xrd{idx}_{curr_mpid}_{curr_formula}_pred{eval_idx}.png')
            torch.save(the_generated_xrd, f'{pred_xrd_folder}/xrd{idx}_{curr_mpid}_{curr_formula}_pred{eval_idx}.pt')
            pred_img_folder = os.path.join(curr_folder, 'pred', f'candidate{eval_idx}', 'vis')
            os.makedirs(pred_img_folder)
            pu.plot_material_single(curr_coords=pred_coords, curr_atom_types=pred_atom_types, output_dir=pred_img_folder, 
                filename=f'structureVis{idx}_{curr_mpid}_{curr_formula}_pred{eval_idx}.png')
        
def reconstruction(idx, batch, model, ld_kwargs, num_evals,
                  down_sample_traj_step=1, model_path=None):
    """
    reconstruct the crystals in <loader>.
    """
    all_frac_coords_stack = []
    all_atom_types_stack = []
    frac_coords = []
    num_atoms = []
    atom_types = []
    lengths = []
    angles = []
    input_data_list = []
    
    assert batch.shape[0] == 1
    if torch.cuda.is_available():
        batch.cuda()
    batch_all_frac_coords = []
    batch_all_atom_types = []
    batch_frac_coords, batch_num_atoms, batch_atom_types = [], [], []
    batch_lengths, batch_angles = [], []

    # Note that the z comes from XRD
    # only sample one z, multiple evals for stoichaticity in langevin dynamics
    _, _, z = model.encode(batch)

    for eval_idx in range(num_evals):
        # TODO: speed up by parallel
        gt_num_atoms = batch.num_atoms
        gt_atom_types = batch.atom_types
        outputs = model.langevin_dynamics(
            z, ld_kwargs, gt_num_atoms, gt_atom_types)

        # collect sampled crystals in this batch.
        batch_frac_coords.append(outputs['frac_coords'].detach().cpu())
        batch_num_atoms.append(outputs['num_atoms'].detach().cpu())
        batch_atom_types.append(outputs['atom_types'].detach().cpu())
        batch_lengths.append(outputs['lengths'].detach().cpu())
        batch_angles.append(outputs['angles'].detach().cpu())
        if ld_kwargs.save_traj:
            batch_all_frac_coords.append(
                outputs['all_frac_coords'][::down_sample_traj_step].detach().cpu())
            batch_all_atom_types.append(
                outputs['all_atom_types'][::down_sample_traj_step].detach().cpu())
    # collect sampled crystals for this z.
    frac_coords.append(torch.stack(batch_frac_coords, dim=0))
    num_atoms.append(torch.stack(batch_num_atoms, dim=0))
    atom_types.append(torch.stack(batch_atom_types, dim=0))
    lengths.append(torch.stack(batch_lengths, dim=0))
    angles.append(torch.stack(batch_angles, dim=0))
    if ld_kwargs.save_traj:
        all_frac_coords_stack.append(
            torch.stack(batch_all_frac_coords, dim=0))
        all_atom_types_stack.append(
            torch.stack(batch_all_atom_types, dim=0))
    # Save the ground truth structure
    input_data_list = input_data_list + batch.to_data_list()

    frac_coords = torch.cat(frac_coords, dim=1)
    num_atoms = torch.cat(num_atoms, dim=1)
    atom_types = torch.cat(atom_types, dim=1)
    lengths = torch.cat(lengths, dim=1)
    angles = torch.cat(angles, dim=1)
    if ld_kwargs.save_traj:
        all_frac_coords_stack = torch.cat(all_frac_coords_stack, dim=2)
        all_atom_types_stack = torch.cat(all_atom_types_stack, dim=2)
    input_data_batch = Batch.from_data_list(input_data_list)

    ret_val = [
        frac_coords, num_atoms, atom_types, lengths, angles,
        all_frac_coords_stack, all_atom_types_stack, input_data_batch]
    return ret_val

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
        assert all_xrds.shape == (len(all_coords), 4096)

    return all_coords, all_atom_types, all_xrds, crystals_list

# TODO: metrics

def main(args):
    # load_data if do reconstruction.
    model_path = Path(args.model_path)
    model, test_loader, cfg = load_model(
        model_path, load_data=('recon' in args.tasks) or
        ('opt' in args.tasks and args.start_from == 'data'))
    ld_kwargs = SimpleNamespace(n_step_each=args.n_step_each,
                                step_lr=args.step_lr,
                                min_sigma=args.min_sigma,
                                save_traj=args.save_traj,
                                disable_bar=args.disable_bar)

    test_dataset = CrystDataset(
        args.data_dir,
        filename='test.csv',
        prop='xrd'
    )
    test_dataset.lattice_scaler = torch.load(
        Path(model_path) / 'lattice_scaler.pt')
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
    ) 
    
    if torch.cuda.is_available():
        model.to('cuda')

    print('Evaluate model on the reconstruction task.')
    start_time = time.time()

    reconstruct_all(test_loader, model, ld_kwargs, args.num_evals,
        args.down_sample_traj_step, args.model_path)

    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # TODO: add args
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--xrd', action='store_true') # TODO: deprecate option
    parser.add_argument('--data_dir', default='data', type=str)
    parser.add_argument('--tasks', nargs='+', default=['recon', 'gen', 'opt'])
    parser.add_argument('--n_step_each', default=100, type=int)
    parser.add_argument('--step_lr', default=1e-4, type=float)
    parser.add_argument('--min_sigma', default=0, type=float)
    parser.add_argument('--save_traj', default=False, type=bool)
    parser.add_argument('--disable_bar', default=False, type=bool)
    parser.add_argument('--num_evals', default=1, type=int)
    parser.add_argument('--num_batches_to_samples', default=20, type=int)
    parser.add_argument('--start_from', default='data', type=str)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--force_num_atoms', action='store_true')
    parser.add_argument('--force_atom_types', action='store_true')
    parser.add_argument('--down_sample_traj_step', default=10, type=int)
    parser.add_argument('--label', default='')

    args = parser.parse_args()
    
    assert args.batch_size == 1

    print('starting eval', args)
    main(args)

import time
import argparse
import torch
import os

from tqdm import tqdm
from torch.optim import Adam
import torch.nn.functional as F
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from torch_geometric.data import Batch
from torch_geometric.data import DataLoader
from torch.distributions import MultivariateNormal


from eval_utils import load_model
from cdvae.pl_modules.xrd import XRDEncoder
from cdvae.pl_data.dataset import CrystXRDDataset
from cdvae.common.data_utils import get_scaler_from_data_list
from visualization.visualize_materials import create_materials, augment_xrdStrip, plot_material_single, plot_xrd_single

def optimization(model, ld_kwargs, data_loader,
                 num_starting_points=1000, num_gradient_steps=5000,
                 lr=1e-3, k=10, l2_penalty=1e-5):
    assert data_loader is not None

    opt_material_folder = f'materials_viz/test/opt_material'
    opt_xrd_folder = f'materials_viz/test/opt_xrd'
    gt_material_folder = f'materials_viz/test/base_truth_material'
    gt_xrd_folder = f'materials_viz/test/base_truth_xrd'
    os.makedirs(opt_material_folder, exist_ok=True)
    os.makedirs(opt_xrd_folder, exist_ok=True)
    os.makedirs(gt_material_folder, exist_ok=True)
    os.makedirs(gt_xrd_folder, exist_ok=True)

    m = MultivariateNormal(torch.zeros(model.hparams.hidden_dim).cuda(), torch.eye(model.hparams.hidden_dim).cuda())

    for j, batch in enumerate(data_loader):
        if j == k:
            break
        batch = batch.to(model.device)
        # Initialize random latent codes! (Nonsensical to encode, then decode)
        
        z = torch.randn(num_starting_points, model.hparams.hidden_dim,
                        device=model.device)
        
        z.requires_grad = True
        target_noisy_xrd = batch.y.reshape(1, 512)
        opt = Adam([z], lr=lr)
        model.freeze()
        with tqdm(total=num_gradient_steps, desc="Property opt", unit="steps") as pbar:
            for i in range(num_gradient_steps):
                opt.zero_grad()
                xrd_loss = F.mse_loss(model.fc_property(z), target_noisy_xrd.broadcast_to(z.shape[0], 512))
                prob = m.log_prob(z).mean()
                pbar.set_postfix(loss=f"XRD loss: {xrd_loss.item():.3e}; Gaussian log PDF: {prob.item():.3e}", refresh=True)
                # Update the progress bar by one step
                pbar.update(1)
                # calculate total loss: minimize XRD loss, maximize latent code probability (min neg prob)
                total_loss = xrd_loss - l2_penalty * prob
                # backprop through total loss
                total_loss.backward()
                opt.step()
                if i == (num_gradient_steps-1):
                    # TODO: speed this one up
                    crystals = model.langevin_dynamics(z, ld_kwargs)
                    crystals = {k: crystals[k] for k in ['frac_coords', 'atom_types', 'num_atoms', 'lengths', 'angles']}

        # convert crystals to xrds
        frac_coords = crystals['frac_coords']
        num_atoms = crystals['num_atoms']
        atom_types = crystals['atom_types']
        lengths = crystals['lengths']
        angles = crystals['angles']

        args = SimpleNamespace()
        args.wave_source = 'CuKa'
        args.num_materials = num_starting_points
        args.xrd_vector_dim = 512
        args.max_theta = 180
        args.min_theta = 0

        # predictions
        the_coords, atom_types, generated_xrds = create_materials(args, 
                frac_coords, num_atoms, atom_types, lengths, angles, create_xrd=True)

        # apply smoothing to the XRD patterns
        smoothed_xrds = list()
        for i in range(generated_xrds.shape[0]):
            smoothed_xrd = augment_xrdStrip(torch.tensor(generated_xrds[i,:]))
            smoothed_xrds.append(smoothed_xrd)
        generated_xrds = torch.stack(smoothed_xrds, dim=0).numpy()

        # compute loss on desired and generated xrds
        target = target_noisy_xrd.broadcast_to(generated_xrds.shape[0], 512)
        input = torch.tensor(generated_xrds).to(model.device)
        loss = F.mse_loss(input, target, reduction='none').mean(dim=-1)
        # find the minimum loss element
        min_loss_idx = torch.argmin(loss)
        # construct the corresponding crystal
        opt_coords = the_coords[min_loss_idx]
        opt_atom_types = atom_types[min_loss_idx]
        opt_xrd = input[min_loss_idx, :].cpu().numpy()
        # save the optimal crystal and its xrd
        plot_material_single(opt_coords, opt_atom_types, opt_material_folder, idx=j)
        plot_xrd_single(args, opt_xrd, opt_xrd_folder, idx=j)

        # plot base truth
        frac_coords = batch.frac_coords
        num_atoms = batch.num_atoms
        atom_types = batch.atom_types
        lengths = batch.lengths
        angles = batch.angles

        the_coords, atom_types, generated_xrds = create_materials(args, 
                frac_coords, num_atoms, atom_types, lengths, angles, create_xrd=True)
        the_coords = np.array(the_coords)[0]
        atom_types = np.array(atom_types)[0]

        plot_material_single(the_coords, atom_types, gt_material_folder, idx=j)
        plot_xrd_single(args, target_noisy_xrd.squeeze().cpu().numpy(), gt_xrd_folder, idx=j)

        # metrics
        assert target_noisy_xrd.squeeze().shape == input[min_loss_idx].squeeze().shape
        xrd_error = F.mse_loss(target_noisy_xrd.squeeze(), input[min_loss_idx].squeeze()).item()
        print(f'xrd_error: {xrd_error}')

        composition_error = compare_composition(atom_types, opt_atom_types)
        print(f'composition error: {composition_error}')

def get_elemental_ratios(atom_types):
    one_hot = np.zeros(119+1)
    for atom in atom_types:
        one_hot[atom.Z] += 1
    one_hot /= np.sum(one_hot)
    return one_hot
    
def compare_composition(gt_atom_types, pred_atom_types):
    gt_atom_types = get_elemental_ratios(gt_atom_types)
    pred_atom_types = get_elemental_ratios(pred_atom_types)
    result = np.sum(np.abs(gt_atom_types - pred_atom_types)) / 2
    assert result >= -1e-5 and result <= 1 + 1e-5
    return result

def main(args):
    # load_data if do reconstruction.
    model_path = Path(args.model_path)    
    model, test_loader, cfg = load_model(
        model_path, load_data=True, batch_size=1)
    ld_kwargs = SimpleNamespace(n_step_each=args.n_step_each,
                                step_lr=args.step_lr,
                                min_sigma=args.min_sigma,
                                save_traj=args.save_traj,
                                disable_bar=args.disable_bar)
    
    if torch.cuda.is_available():
        model.to('cuda')

    print('Evaluate model on the property optimization task.')
    start_time = time.time()
    if args.start_from == 'data':
        loader = test_loader
    else:
        loader = None
    optimization(model, ld_kwargs, loader, 
                 l2_penalty=args.l2_penalty, 
                 num_starting_points=args.num_starting_points,
                 num_gradient_steps=args.num_gradient_steps,
                 lr=args.lr,
                 k=args.num_tested_materials)    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--data_dir', default='data', type=str)
    parser.add_argument('--n_step_each', default=100, type=int)
    parser.add_argument('--step_lr', default=1e-4, type=float)
    parser.add_argument('--save_traj', default=False, type=bool)
    parser.add_argument('--min_sigma', default=0, type=float)
    parser.add_argument('--disable_bar', default=False, type=bool)
    parser.add_argument('--num_evals', default=1, type=int)
    parser.add_argument('--start_from', default='data', type=str)
    parser.add_argument('--l2_penalty', default=1e-5, type=float)
    parser.add_argument('--num_starting_points', default=1000, type=int)
    parser.add_argument('--num_gradient_steps', default=5000, type=int)
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--num_tested_materials', default=10, type=int)
    parser.add_argument('--label', default='')

    args = parser.parse_args()

    print('starting eval', args)
    main(args)

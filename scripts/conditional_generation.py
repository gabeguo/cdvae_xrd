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
from visualization.visualize_materials import create_materials, augment_xrdStrip, plot_material_single, plot_xrds

def optimization(model, ld_kwargs, data_loader,
                 num_starting_points=1000, num_gradient_steps=5000,
                 lr=1e-3, k=10):
    assert data_loader is not None

    m = MultivariateNormal(torch.zeros(model.hparams.hidden_dim).cuda(), torch.eye(model.hparams.hidden_dim).cuda())

    j = 0
    for batch in data_loader:
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
                loss = F.mse_loss(model.fc_property(z), target_noisy_xrd.broadcast_to(z.shape[0], 512))
                prob = m.log_prob(z).mean()
                pbar.set_postfix(loss=f"XRD loss: {loss.item():.3e}; Gaussian log PDF: {prob.item():.3e}", refresh=True)
                # Update the progress bar by one step
                pbar.update(1)
                # save elementwise mse
                loss.backward()
                opt.step()
                if i == (num_gradient_steps-1):
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
        opt_xrd = input[min_loss_idx, :].unsqueeze(0).cpu().numpy()
        # save the optimal crystal and its xrd
        material_folder = f'materials_viz/test/opt_material_{j}'
        xrd_folder = f'materials_viz/test/opt_xrd_{j}'
        os.makedirs(material_folder, exist_ok=True)
        os.makedirs(xrd_folder, exist_ok=True)
        plot_material_single(opt_coords, opt_atom_types, material_folder)
        plot_xrds(args, opt_xrd, xrd_folder)

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

        os.makedirs(f'materials_viz/test/base_truth_material_{j}', exist_ok=True)
        os.makedirs(f'materials_viz/test/base_truth_xrd_{j}', exist_ok=True)
        plot_material_single(the_coords, atom_types, f'materials_viz/test/base_truth_material_{j}')
        plot_xrds(args, target_noisy_xrd.cpu().numpy(), f'materials_viz/test/base_truth_xrd_{j}')

        j += 1

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
    optimization(model, ld_kwargs, loader)    


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
    parser.add_argument('--label', default='')

    args = parser.parse_args()

    print('starting eval', args)
    main(args)

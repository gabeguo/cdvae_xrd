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


from eval_utils import load_model
from cdvae.pl_modules.xrd import XRDEncoder
from cdvae.pl_data.dataset import CrystXRDDataset
from cdvae.common.data_utils import get_scaler_from_data_list
from visualization.visualize_materials import create_materials, augment_xrdStrip

def optimization(model, ld_kwargs, data_loader,
                 num_starting_points=1000, k=3, num_gradient_steps=20000,
                 lr=1e-2):
    assert data_loader is not None

    batch = next(iter(data_loader)).to(model.device)
    # Initialize random latent codes! (Nonsensical to encode, then decode)
    
    z = torch.randn(num_starting_points, model.hparams.hidden_dim,
                    device=model.device)
    
    z.requires_grad = True
    target_noisy_xrd = batch.y.reshape(1, 512)
    opt = Adam([z], lr=lr)
    model.freeze()

    #### CHANGE BACK
    num_gradient_steps = 5
    for i in range(num_gradient_steps):
    #### CHANGE BACK
        opt.zero_grad()
        loss = F.mse_loss(model.fc_property(z), target_noisy_xrd.broadcast_to(z.shape[0], 512))
        print(f'predicted property loss: {loss.item()}')
        # save elementwise mse
        loss.backward()
        opt.step()
        if i == (num_gradient_steps-1):
            #### CHANGE BACK
            ld_kwargs.n_step_each = 1
            #### CHANGE BACK
            crystals = model.langevin_dynamics(z, ld_kwargs)
            crystals = {k: crystals[k] for k in ['frac_coords', 'atom_types', 'num_atoms', 'lengths', 'angles']}
    print("")
    # convert crystals to xrds
    frac_coords = crystals['frac_coords']
    num_atoms = crystals['num_atoms']
    atom_types = crystals['atom_types']
    lengths = crystals['lengths']
    angles = crystals['angles']

    args = SimpleNamespace()
    args.wave_source = 'CuKa'
    args.num_materials = k
    args.xrd_vector_dim = 512
    args.max_theta = 180
    args.min_theta = 0

    # predictions
    the_coords, atom_types, generated_xrds = create_materials(args, 
            frac_coords, num_atoms, atom_types, lengths, angles, create_xrd=True)

    print(generated_xrds.shape)    

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
    optimized_crystals, data = optimization(model, ld_kwargs, loader)
    optimized_crystals.update({'data': data,
                                'eval_setting': args,
                                'time': time.time() - start_time})

    


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

import time
import argparse
import torch
import os

from tqdm import tqdm
from torch.optim import Adam
from pathlib import Path
from types import SimpleNamespace
from torch_geometric.data import Batch
from torch_geometric.data import DataLoader


from eval_utils import load_model
from cdvae.pl_modules.xrd_encoder import XRDEncoder
from cdvae.pl_data.dataset import CrystXRDDataset
from cdvae.common.data_utils import get_scaler_from_data_list

def reconstructon(loader, model, ld_kwargs, num_evals,
                  force_num_atoms=False, force_atom_types=False, down_sample_traj_step=1, xrd=False, model_path=None):
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

    if xrd:
        xrd_encoder = XRDEncoder().to('cuda' if torch.cuda.is_available() else 'cpu')
        xrd_encoder.load_state_dict(torch.load(os.path.join(model_path, 'xrd_enc.pt')))
        all_noised_xrds = list()
    
    for idx, batch in enumerate(loader):
        if xrd:
            batch, xrd_data = batch 
        if torch.cuda.is_available():
            batch.cuda()
        print(f'batch {idx} in {len(loader)}')
        batch_all_frac_coords = []
        batch_all_atom_types = []
        batch_frac_coords, batch_num_atoms, batch_atom_types = [], [], []
        batch_lengths, batch_angles = [], []

        # only sample one z, multiple evals for stoichaticity in langevin dynamics
        if xrd:
            z = xrd_encoder(xrd_data.cuda().unsqueeze(1))
            all_noised_xrds.append(xrd_data)
        else:
            _, _, z = model.encode(batch)

        for eval_idx in range(num_evals):
            gt_num_atoms = batch.num_atoms if force_num_atoms else None
            gt_atom_types = batch.atom_types if force_atom_types else None
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
    if xrd:
        all_noised_xrds = torch.cat(all_noised_xrds, dim=0)
        assert all_noised_xrds.shape == (len(loader.dataset), 256)
        ret_val.append(all_noised_xrds)
    else:
        ret_val.append(None)
    return ret_val


def generation(model, ld_kwargs, num_batches_to_sample, num_samples_per_z,
               batch_size=512, down_sample_traj_step=1):
    all_frac_coords_stack = []
    all_atom_types_stack = []
    frac_coords = []
    num_atoms = []
    atom_types = []
    lengths = []
    angles = []

    for z_idx in range(num_batches_to_sample):
        batch_all_frac_coords = []
        batch_all_atom_types = []
        batch_frac_coords, batch_num_atoms, batch_atom_types = [], [], []
        batch_lengths, batch_angles = [], []

        z = torch.randn(batch_size, model.hparams.hidden_dim,
                        device=model.device)

        for sample_idx in range(num_samples_per_z):
            samples = model.langevin_dynamics(z, ld_kwargs)

            # collect sampled crystals in this batch.
            batch_frac_coords.append(samples['frac_coords'].detach().cpu())
            batch_num_atoms.append(samples['num_atoms'].detach().cpu())
            batch_atom_types.append(samples['atom_types'].detach().cpu())
            batch_lengths.append(samples['lengths'].detach().cpu())
            batch_angles.append(samples['angles'].detach().cpu())
            if ld_kwargs.save_traj:
                batch_all_frac_coords.append(
                    samples['all_frac_coords'][::down_sample_traj_step].detach().cpu())
                batch_all_atom_types.append(
                    samples['all_atom_types'][::down_sample_traj_step].detach().cpu())

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

    frac_coords = torch.cat(frac_coords, dim=1)
    num_atoms = torch.cat(num_atoms, dim=1)
    atom_types = torch.cat(atom_types, dim=1)
    lengths = torch.cat(lengths, dim=1)
    angles = torch.cat(angles, dim=1)
    if ld_kwargs.save_traj:
        all_frac_coords_stack = torch.cat(all_frac_coords_stack, dim=2)
        all_atom_types_stack = torch.cat(all_atom_types_stack, dim=2)
    return (frac_coords, num_atoms, atom_types, lengths, angles,
            all_frac_coords_stack, all_atom_types_stack)


def optimization(model, ld_kwargs, data_loader,
                 num_starting_points=100, num_gradient_steps=5000,
                 lr=1e-3, num_saved_crys=10):
    if data_loader is not None:
        batch = next(iter(data_loader)).to(model.device)
        _, _, z = model.encode(batch)
        z = z[:num_starting_points].detach().clone()
        z.requires_grad = True
    else:
        z = torch.randn(num_starting_points, model.hparams.hidden_dim,
                        device=model.device)
        z.requires_grad = True

    opt = Adam([z], lr=lr)
    model.freeze()

    all_crystals = []
    interval = num_gradient_steps // (num_saved_crys-1)
    for i in tqdm(range(num_gradient_steps)):
        opt.zero_grad()
        loss = model.fc_property(z).mean()
        loss.backward()
        opt.step()

        if i % interval == 0 or i == (num_gradient_steps-1):
            crystals = model.langevin_dynamics(z, ld_kwargs)
            all_crystals.append(crystals)
    return {k: torch.cat([d[k] for d in all_crystals]).unsqueeze(0) for k in
            ['frac_coords', 'atom_types', 'num_atoms', 'lengths', 'angles']}


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
    # overwrite
    if args.xrd:
        dataset_to_prop = {
            'perov_5': 'heat_ref',
            'mp_20': 'formation_energy_per_atom',
            'carbon_24': 'energy_per_atom'
        }
        # test loader
        test_dataset = CrystXRDDataset(
            args.data_dir,
            filename='test.csv',
            prop=dataset_to_prop[args.data_dir.split('/')[-1]]
        )
        test_dataset.lattice_scaler = torch.load(
            Path(model_path) / 'lattice_scaler.pt')
        test_dataset.scaler = torch.load(Path(model_path) / 'prop_scaler.pt')
        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=2,
        ) 
    
    if torch.cuda.is_available():
        model.to('cuda')

    if 'recon' in args.tasks:
        print('Evaluate model on the reconstruction task.')
        start_time = time.time()
        (frac_coords, num_atoms, atom_types, lengths, angles,
         all_frac_coords_stack, all_atom_types_stack, input_data_batch, noised_xrds) = reconstructon(
            test_loader, model, ld_kwargs, args.num_evals,
            args.force_num_atoms, args.force_atom_types, args.down_sample_traj_step, args.xrd, args.model_path)

        if args.label == '':
            recon_out_name = 'eval_recon.pt'
        else:
            recon_out_name = f'eval_recon_{args.label}.pt'

        torch.save({
            'eval_setting': args,
            'input_data_batch': input_data_batch,
            'frac_coords': frac_coords,
            'num_atoms': num_atoms,
            'atom_types': atom_types,
            'lengths': lengths,
            'angles': angles,
            'all_frac_coords_stack': all_frac_coords_stack,
            'all_atom_types_stack': all_atom_types_stack,
            'time': time.time() - start_time,
            'xrds': noised_xrds
        }, model_path / recon_out_name)

    if 'gen' in args.tasks:
        print('Evaluate model on the generation task.')
        start_time = time.time()

        (frac_coords, num_atoms, atom_types, lengths, angles,
         all_frac_coords_stack, all_atom_types_stack) = generation(
            model, ld_kwargs, args.num_batches_to_samples, args.num_evals,
            args.batch_size, args.down_sample_traj_step)

        if args.label == '':
            gen_out_name = 'eval_gen.pt'
        else:
            gen_out_name = f'eval_gen_{args.label}.pt'

        torch.save({
            'eval_setting': args,
            'frac_coords': frac_coords,
            'num_atoms': num_atoms,
            'atom_types': atom_types,
            'lengths': lengths,
            'angles': angles,
            'all_frac_coords_stack': all_frac_coords_stack,
            'all_atom_types_stack': all_atom_types_stack,
            'time': time.time() - start_time
        }, model_path / gen_out_name)

    if 'opt' in args.tasks:
        print('Evaluate model on the property optimization task.')
        start_time = time.time()
        if args.start_from == 'data':
            loader = test_loader
        else:
            loader = None
        optimized_crystals = optimization(model, ld_kwargs, loader)
        optimized_crystals.update({'eval_setting': args,
                                   'time': time.time() - start_time})

        if args.label == '':
            gen_out_name = 'eval_opt.pt'
        else:
            gen_out_name = f'eval_opt_{args.label}.pt'
        torch.save(optimized_crystals, model_path / gen_out_name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--xrd', action='store_true')
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
    parser.add_argument('--batch_size', default=500, type=int)
    parser.add_argument('--force_num_atoms', action='store_true')
    parser.add_argument('--force_atom_types', action='store_true')
    parser.add_argument('--down_sample_traj_step', default=10, type=int)
    parser.add_argument('--label', default='')

    args = parser.parse_args()

    main(args)

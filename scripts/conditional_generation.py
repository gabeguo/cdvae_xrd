import time
import argparse
import torch
import os
import json

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
from compute_metrics import Crystal, RecEval, GenEval
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts

import wandb

from PIL import Image

AVG_COMPOSITION_ERROR = 'composition error rate'
AVG_XRD_MSE = 'Scaled XRD mean squared error'
AVG_XRD_L1 = 'Scaled XRD mean absolute error'
BEST_XRD_MSE = 'Average best scaled XRD mean squared error among candidates'
BEST_XRD_L1 = 'Average best scaled XRD mean absolute error among candidates'

MATCH_RATE = 'match_rate'
RMS_DIST = 'rms_dist'
COMPOSITION_VALIDITY = 'comp_valid'
STRUCTURE_VALIDITY = 'struct_valid'
VALIDITY = 'valid'
NUM_ATOM_ACCURACY = '% materials w/ # atoms pred correctly'

EPS = 1e-10

# Thanks ChatGPT!
def resize_image_to_same_width(image, width):
    """Resize an image to the same width, maintaining the aspect ratio."""
    ratio = width / float(image.width)
    new_height = int(image.height * ratio)
    return image.resize((width, new_height), Image.LANCZOS)

def collate_images(gt_material, gt_xrd, pred_material, pred_xrd, width):
    gt_material = resize_image_to_same_width(Image.open(gt_material), width)
    gt_xrd = resize_image_to_same_width(Image.open(gt_xrd), width)
    pred_material = resize_image_to_same_width(Image.open(pred_material), width)
    pred_xrd = resize_image_to_same_width(Image.open(pred_xrd), width)

    assert gt_material.height == pred_material.height
    assert gt_xrd.height == pred_xrd.height
    
    total_height = gt_material.height + gt_xrd.height
    combined_image = Image.new('RGB', (width * 2, total_height))
    
    combined_image.paste(gt_material, (0, 0))
    combined_image.paste(gt_xrd, (0, gt_material.height))
    combined_image.paste(pred_material, (width, 0))
    combined_image.paste(pred_xrd, (width, pred_material.height))
    
    return combined_image

# Thanks ChatGPT!
def calculate_accuracy(probabilities, labels):
    """
    Calculate accuracy given the softmax probabilities and true labels.
    
    :param probabilities: Softmax probabilities of shape (N, C) where N is the number of samples and C is the number of classes.
    :param labels: True class labels of shape (N,).
    :return: Accuracy as a Python float.
    """
    # Step 1: Convert softmax probabilities to predicted class indices
    _, predicted_classes = torch.max(probabilities, dim=1)
    
    # Step 2: Compare with true class labels
    correct_predictions = (predicted_classes == labels).float()  # Convert boolean tensor to float for sum operation
    
    # Step 3: Calculate accuracy
    accuracy = correct_predictions.sum() / labels.size(0)
    
    return accuracy.item()  # Convert to Python float for readability

def optimization(args, model, ld_kwargs, data_loader,
                 num_starting_points=1000, num_gradient_steps=5000,
                 lr=1e-3, k=10, num_candidates=5, l2_penalty=1e-5, label=''):
    assert data_loader is not None

    base_output_dir = f'materials_viz/{label}'
    os.makedirs(base_output_dir, exist_ok=True)
    with open(os.path.join(base_output_dir, 'parameters.json'), 'w') as fout:
        json.dump(vars(args), fout, indent=4)

    opt_material_folder = f'{base_output_dir}/opt_material'
    opt_xrd_folder = f'{base_output_dir}/opt_xrd'
    gt_material_folder = f'{base_output_dir}/base_truth_material'
    gt_xrd_folder = f'{base_output_dir}/base_truth_xrd'
    metrics_folder = f'{base_output_dir}/metrics'
    os.makedirs(opt_material_folder, exist_ok=True)
    os.makedirs(opt_xrd_folder, exist_ok=True)
    os.makedirs(gt_material_folder, exist_ok=True)
    os.makedirs(gt_xrd_folder, exist_ok=True)
    os.makedirs(metrics_folder, exist_ok=True)

    m = MultivariateNormal(torch.zeros(model.hparams.hidden_dim).cuda(), torch.eye(model.hparams.hidden_dim).cuda())

    all_gt_crystals = list()
    all_bestPred_crystals = list()

    all_composition_errors = list()
    all_xrd_l1_errors = list()
    all_xrd_l2_errors = list()
    total_correct_num_atoms = 0

    for j, batch in enumerate(data_loader):
        wandb.init(config=args, project='conditional generation', name=label, group=f'crystal {j}')
        if j == k:
            break
        batch = batch.to(model.device)
        # Initialize random latent codes! (Nonsensical to encode, then decode)
        
        z = torch.randn(num_starting_points, model.hparams.hidden_dim,
                        device=model.device)
        
        z.requires_grad = True
        # get xrd
        target_noisy_xrd = batch.y.reshape(1, 512)
        opt = Adam([z], lr=lr)
        total_gradient_steps = num_gradient_steps * (1+2+4) - 1
        scheduler = CosineAnnealingWarmRestarts(opt, num_gradient_steps, T_mult=2, eta_min=args.min_lr)
        model.freeze()
        with tqdm(total=total_gradient_steps, desc="Property opt", unit="steps") as pbar:
            # TODO: add model.fc_num_atoms, model.fc_composition, model.fc_lattice
            for i in range(total_gradient_steps):
                opt.zero_grad()
                if args.l1_loss:
                    xrd_loss = F.l1_loss(model.fc_property(z), target_noisy_xrd.broadcast_to(z.shape[0], 512))
                else:
                    xrd_loss = F.mse_loss(model.fc_property(z), target_noisy_xrd.broadcast_to(z.shape[0], 512))
                prob = m.log_prob(z).mean()
                # predict the number of atoms, lattice, composition
                (pred_num_atoms, pred_lengths_and_angles, pred_lengths, pred_angles,
                pred_composition_per_atom) = model.decode_stats(
                    z, batch.num_atoms, batch.lengths, batch.angles, teacher_forcing=False)
                # print('pred vs gt num atoms', pred_num_atoms.shape, 
                #       batch.num_atoms.repeat(num_starting_points).shape)
                # print('pred vs gt lattice', pred_lengths_and_angles.shape, torch.cat((batch.lengths, batch.angles), dim=1).repeat(num_starting_points, 1).shape)
                # print('pred vs gt composition', pred_composition_per_atom.shape, 
                #       batch.atom_types.repeat(num_starting_points).shape)
                # Need to repeat, because we have multiple starting points
                num_atom_loss = F.cross_entropy(pred_num_atoms, 
                                                batch.num_atoms.repeat(num_starting_points))
                # lattice_loss = model.lattice_loss(pred_lengths_and_angles, batch)
                # TODO: they do some weird stuff with composition loss: double check (I think it was inconsequential, but idk)
                composition_loss = F.cross_entropy(pred_composition_per_atom, 
                                                   (batch.atom_types - 1).repeat(num_starting_points))
                num_atom_accuracy = calculate_accuracy(pred_num_atoms, batch.num_atoms.repeat(num_starting_points))
                composition_accuracy = calculate_accuracy(pred_composition_per_atom, (batch.atom_types - 1).repeat(num_starting_points))
                pbar.set_postfix_str(f"XRD loss: {xrd_loss.item():.3e}; Gaussian log PDF: {prob.item():.3e}; " + 
                                     f"Num atom loss: {num_atom_loss.item():.3e}; Composition loss: {composition_loss.item():.3e}", 
                                     refresh=True)
                # Update the progress bar by one step
                pbar.update(1)
                # calculate total loss: minimize XRD loss, maximize latent code probability (min neg prob)
                total_loss = xrd_loss - l2_penalty * prob \
                    + args.num_atom_lambda * num_atom_loss \
                    + args.composition_lambda * composition_loss

                if i % 100 == 0 and i > 0:
                    wandb.log(
                        {
                            "total_loss":total_loss,
                            "lr":scheduler.get_last_lr()[0],
                            "xrd_loss":xrd_loss,
                            "log_prob":prob,
                            "num_atom_loss":num_atom_loss,
                            "composition_loss":composition_loss,
                            "num_atom_accuracy":num_atom_accuracy,
                            "composition_accuracy":composition_accuracy
                        },
                    step=i)

                # backprop through total loss
                total_loss.backward()
                opt.step()
                scheduler.step()

        # TODO: speed this one up
        init_num_atoms = batch.num_atoms.repeat(num_starting_points) if args.num_atom_lambda > EPS \
            else None
        init_atom_types = batch.atom_types.repeat(num_starting_points) if args.composition_lambda > EPS \
            else None

        print('know num atoms:', init_num_atoms is not None)
        print('know atom types:', init_atom_types is not None)

        crystals = model.langevin_dynamics(z, ld_kwargs, gt_num_atoms=init_num_atoms, gt_atom_types=init_atom_types)
        crystals = {k: crystals[k] for k in ['frac_coords', 'atom_types', 'num_atoms', 'lengths', 'angles']}

        alt_args = SimpleNamespace()
        alt_args.wave_source = 'CuKa'
        alt_args.num_materials = num_starting_points
        alt_args.xrd_vector_dim = 512
        alt_args.max_theta = 180
        alt_args.min_theta = 0
            
        # predictions
        frac_coords = crystals['frac_coords']
        num_atoms = crystals['num_atoms']
        atom_types = crystals['atom_types']
        lengths = crystals['lengths']
        angles = crystals['angles']

        all_opt_coords, all_opt_atom_types, opt_generated_xrds, curr_gen_crystals_list = create_materials(alt_args, 
                frac_coords, num_atoms, atom_types, lengths, angles, create_xrd=True)

        # plot base truth
        frac_coords = batch.frac_coords
        num_atoms = batch.num_atoms
        atom_types = batch.atom_types
        lengths = batch.lengths
        angles = batch.angles

        assert num_atoms.shape[0] == 1
        assert frac_coords.shape[0] == atom_types.shape[0]

        the_coords, atom_types, bt_generated_xrds, singleton_gt_crystal_list = create_materials(alt_args, 
                frac_coords, num_atoms, atom_types, lengths, angles, create_xrd=True)
        the_coords = np.array(the_coords)[0]
        atom_types = np.array(atom_types)[0]

        assert len(singleton_gt_crystal_list) == 1

        curr_gt_crystal = Crystal(singleton_gt_crystal_list[0])
        # log crystal
        all_gt_crystals.append(curr_gt_crystal)


        gt_material_filepath = plot_material_single(the_coords, atom_types, gt_material_folder, idx=j)
        gt_xrd_filepath = plot_xrd_single(alt_args, target_noisy_xrd.squeeze().cpu().numpy(), gt_xrd_folder, idx=j)

        # apply smoothing to the XRD patterns
        smoothed_xrds = list()
        for i in range(opt_generated_xrds.shape[0]):
            smoothed_xrd = augment_xrdStrip(torch.tensor(opt_generated_xrds[i,:]))
            smoothed_xrds.append(smoothed_xrd)
        opt_generated_xrds = torch.stack(smoothed_xrds, dim=0).numpy()

        # compute loss on desired and generated xrds
        target = target_noisy_xrd.broadcast_to(bt_generated_xrds.shape[0], 512)
        input = torch.tensor(opt_generated_xrds).to(model.device)
        print("GENERATED_XRDS shape:", input.shape)
        if args.l1_loss:
            loss = F.l1_loss(input, target, reduction='none').mean(dim=-1)
        else:
            loss = F.mse_loss(input, target, reduction='none').mean(dim=-1)
        
        # find the (num_candidates) minimum loss elements
        min_loss_indices = torch.argsort(loss).squeeze(0)[:num_candidates].tolist()
        
        candidate_xrd_l1_errors = list()
        candidate_xrd_l2_errors = list()

        # create material subdir
        subdir = f'material_{j}'
        opt_material_folder_cand = f'{opt_material_folder}/{subdir}'
        opt_xrd_folder_cand = f'{opt_xrd_folder}/{subdir}'
        
        os.makedirs(opt_material_folder_cand, exist_ok=True)
        os.makedirs(opt_xrd_folder_cand, exist_ok=True)

        print(f'crystal {j} has {len(min_loss_indices)} candidates')
        for i, min_loss_idx in enumerate(min_loss_indices): # for each candidate

            filename = f'candidate_{i}.png'
            # construct the corresponding crystal
            opt_coords = all_opt_coords[min_loss_idx]
            opt_atom_types = all_opt_atom_types[min_loss_idx]
            opt_xrd = input[min_loss_idx, :].cpu().numpy()
            curr_pred_crystal = Crystal(curr_gen_crystals_list[min_loss_idx])

            # log best (lowest loss) crystal for metrics
            if i == 0:
                all_bestPred_crystals.append(curr_pred_crystal)

            # save the optimal crystal and its xrd
            pred_material_filepath = plot_material_single(opt_coords, opt_atom_types, opt_material_folder_cand, idx=j, filename=filename)
            pred_xrd_filepath = plot_xrd_single(alt_args, opt_xrd, opt_xrd_folder_cand, idx=j, filename=filename)
               
            # Log image
            log_img = collate_images(gt_material=gt_material_filepath, gt_xrd=gt_xrd_filepath,
                                    pred_material=pred_material_filepath, pred_xrd=pred_xrd_filepath,
                                    width=600)
            wandb.log({"prediction": wandb.Image(log_img)})  

            # metrics
            assert target_noisy_xrd.squeeze().shape == input[min_loss_idx].squeeze().shape
            xrd_l1_error = F.l1_loss(target_noisy_xrd.squeeze(), input[min_loss_idx].squeeze()).item()
            xrd_l2_error = F.mse_loss(target_noisy_xrd.squeeze(), input[min_loss_idx].squeeze()).item()
            candidate_xrd_l1_errors.append(xrd_l1_error)
            candidate_xrd_l2_errors.append(xrd_l2_error)
            print(f'xrd l1 error: {xrd_l1_error}')
            print(f'xrd l2 error: {xrd_l2_error}')

            composition_error = compare_composition(atom_types, opt_atom_types)
            all_composition_errors.append(composition_error)
            print(f'composition error: {composition_error}')

            is_num_atoms_correct = compare_num_atoms(gt_atom_types=atom_types, 
                                                    pred_atom_types=opt_atom_types)
            if is_num_atoms_correct:
                total_correct_num_atoms += 1
            print(f'num atoms: {len(atom_types)} (gt) vs {len(opt_atom_types)} (pred)')

        
        all_xrd_l1_errors.append(candidate_xrd_l1_errors)
        all_xrd_l2_errors.append(candidate_xrd_l2_errors)

        wandb.finish() 
        
    # average xrd errors
    avg_xrd_mse = np.mean([list for list in all_xrd_l2_errors])
    avg_xrd_l1 = np.mean([list for list in all_xrd_l1_errors])

    # best of candidate xrd errors
    best_xrd_mse = np.mean([np.min(list) for list in all_xrd_l2_errors])
    best_xrd_l1 = np.mean([np.min(list) for list in all_xrd_l1_errors])

    ret_val = {
        AVG_COMPOSITION_ERROR: np.mean(all_composition_errors),
        AVG_XRD_MSE: avg_xrd_mse,
        AVG_XRD_L1: avg_xrd_l1,
        BEST_XRD_MSE: best_xrd_mse,
        BEST_XRD_L1: best_xrd_l1
    }

    ret_val.update(check_structure_match(gt_structures=all_gt_crystals, 
                                         pred_structures=all_bestPred_crystals))
    ret_val.update(check_validity(gt_structures=all_gt_crystals,
                                  pred_structures=all_bestPred_crystals))
    ret_val[NUM_ATOM_ACCURACY] = total_correct_num_atoms / len(all_gt_crystals)

    with open(f'{metrics_folder}/aggregate_metrics.json', 'w') as fout:
        json.dump(ret_val, fout, indent=4)
    
    print(json.dumps(ret_val, indent=4))

    return ret_val

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

def compare_num_atoms(gt_atom_types, pred_atom_types):
    if isinstance(gt_atom_types, list):
        assert not isinstance(gt_atom_types[0], list)
    else:
        assert len(gt_atom_types.shape) == 1
    if isinstance(pred_atom_types, list):
        assert not isinstance(pred_atom_types[-1], list)
    else:
        assert len(pred_atom_types.shape) == 1
    return len(gt_atom_types) == len(pred_atom_types)

def check_structure_match(gt_structures, pred_structures):
    """
    Input: lists of Crystal() objects
    return {'match_rate': match_rate,
            'rms_dist': mean_rms_dist}
    """
    structure_evaluator = RecEval(pred_crys=pred_structures, gt_crys=gt_structures)
    match_rate_rms_dict = structure_evaluator.get_match_rate_and_rms()
    return match_rate_rms_dict

def check_validity(gt_structures, pred_structures):
    """
    Input: lists of Crystal() objects
    return {'comp_valid': comp_valid,
            'struct_valid': struct_valid,
            'valid': valid}
    """
    validity_checker = GenEval(pred_crys=pred_structures, gt_crys=gt_structures, n_samples=len(gt_structures))
    return validity_checker.get_validity()

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
    optimization(args, model, ld_kwargs, loader, 
                 l2_penalty=args.l2_penalty, 
                 num_starting_points=args.num_starting_points,
                 num_gradient_steps=args.num_gradient_steps,
                 lr=args.lr,
                 k=args.num_tested_materials,
                 num_candidates=args.num_candidates,
                 label=args.label)    

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
    parser.add_argument('--num_atom_lambda', default=1e-3, type=float)
    parser.add_argument('--lattice_lambda', default=1e-3, type=float)
    parser.add_argument('--composition_lambda', default=1e-3, type=float)
    parser.add_argument('--num_starting_points', default=1000, type=int)
    parser.add_argument('--num_gradient_steps', default=5000, type=int)
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--min_lr', default=1e-5, type=float)
    parser.add_argument('--num_tested_materials', default=10, type=int)
    parser.add_argument('--l1_loss', action='store_true')
    parser.add_argument('--label', default='')
    parser.add_argument('--num_candidates', default=5, type=int)

    args = parser.parse_args()

    print('starting eval', args)
    main(args)

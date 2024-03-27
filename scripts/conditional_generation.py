import time
import argparse
import torch
import os
import json

import matplotlib.pyplot as plt
import numpy as np

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
from cdvae.common.data_utils import get_scaler_from_data_list
from visualization.visualize_materials import create_materials, plot_material_single, plot_xrd_single
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

USE_ALL_SPACEGROUPS = "aggregated stats (all spacegroups)"
COUNT = "number of crystals"

EPS = 1e-10

# Thanks ChatGPT!

# If you want to change the colors of the lines and shades, simply modify in the ax.fill_between() and ax.plot() functions
# A list of possible colors can be found at: https://matplotlib.org/stable/gallery/color/named_colors.html
def plot_overlaid_graphs(actual, prediction_nn, prediction_simulated, savepath):
    fig, ax = plt.subplots()

    thetas = [pos * 180 / len(actual) for pos in range(len(actual))]

    # Plot and fill the area under the first curve
    ax.fill_between(thetas, actual, color="royalblue", alpha=0.2)
    ax.plot(thetas, actual, color="blue", alpha=0.6, label="Actual")  # Curve line

    # Plot and fill the area under the second curve
    ax.fill_between(thetas, prediction_nn, color="mistyrose", alpha=0.2)
    ax.plot(thetas, prediction_nn, color="red", alpha=0.6, linestyle='dotted', linewidth=2, label="Prediction (NN)")  # Dotted curve line with increased linewidth

    # Plot and fill the area under the second curve
    ax.fill_between(thetas, prediction_simulated, color="lightgreen", alpha=0.2)
    ax.plot(thetas, prediction_simulated, color="green", alpha=0.6, linestyle='dashed', linewidth=2, label="Prediction (Simulated)")  # Dotted curve line with increased linewidth

    # Customizing the plot
    ax.set_title("XRD Patterns")
    ax.set_xlabel("Theta (degrees)")
    ax.set_ylabel("Scaled Intensity")
    ax.set_xlim(0, 180)  # Set x-axis limits
    ax.set_ylim(0, 1)  # Set y-axis limits
    ax.set_xticks(np.arange(0, 181, 10))
    ax.set_xticklabels(ax.get_xticks(), rotation=70)  # Rotate x-axis labels by 70 degrees
    ax.set_yticks(np.arange(0, 1.1, 0.1))  # Set horizontal gridlines every 0.1 from 0 to 1
    ax.grid(True)  # Show gridlines
    ax.legend()

    # Display the plot
    #plt.show()
    plt.savefig(savepath)
    plt.tight_layout()
    plt.close()

    return

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

def optimize_latent_code(args, model, batch, target_noisy_xrd):
    m = MultivariateNormal(torch.zeros(model.hparams.hidden_dim).cuda(), 
                           torch.eye(model.hparams.hidden_dim).cuda())
    
    z = torch.randn(args.num_starting_points, model.hparams.hidden_dim,
                    device=model.device)
    
    z.requires_grad = True
    opt = Adam([z], lr=args.lr)
    total_gradient_steps = args.num_gradient_steps * (1+2+4) - 1
    scheduler = CosineAnnealingWarmRestarts(opt, args.num_gradient_steps, T_mult=2, eta_min=args.min_lr)
    model.freeze()
    with tqdm(total=total_gradient_steps, desc="Property opt", unit="steps") as pbar:
        for i in range(total_gradient_steps):
            opt.zero_grad()
            xrd_loss = F.l1_loss(model.fc_property(z), target_noisy_xrd.broadcast_to(z.shape[0], 512)) if args.l1_loss \
                else F.mse_loss(model.fc_property(z), target_noisy_xrd.broadcast_to(z.shape[0], 512))
            prob = m.log_prob(z).mean()
            # predict the number of atoms, lattice, composition
            (pred_num_atoms, pred_lengths_and_angles, pred_lengths, pred_angles,
            pred_composition_per_atom) = model.decode_stats(
                z, batch.num_atoms, batch.lengths, batch.angles, teacher_forcing=False)
            num_atom_loss = F.cross_entropy(pred_num_atoms, 
                                            batch.num_atoms.repeat(args.num_starting_points))
            # TODO: they do some weird stuff with composition loss: double check (I think it was inconsequential, but idk)
            composition_loss = F.cross_entropy(pred_composition_per_atom, 
                                                (batch.atom_types - 1).repeat(args.num_starting_points))
            num_atom_accuracy = calculate_accuracy(pred_num_atoms, batch.num_atoms.repeat(args.num_starting_points))
            composition_accuracy = calculate_accuracy(pred_composition_per_atom, (batch.atom_types - 1).repeat(args.num_starting_points))
            pbar.set_postfix_str(f"XRD loss: {xrd_loss.item():.3e}; Gaussian log PDF: {prob.item():.3e}; " + 
                                    f"Num atom loss: {num_atom_loss.item():.3e}; Composition loss: {composition_loss.item():.3e}", 
                                    refresh=True)
            # Update the progress bar by one step
            pbar.update(1)
            # calculate total loss: minimize XRD loss, maximize latent code probability (min neg prob)
            total_loss = xrd_loss - args.l2_penalty * prob \
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
    return z

def process_candidates(args, xrd_args, j,
        curr_gen_crystals_list, all_opt_coords, all_opt_atom_types, 
        opt_generated_xrds, 
        min_loss_indices, opt_material_folder, opt_xrd_folder, pred_opt_xrd_folder, opt_cif_folder, metrics_folder, subdir,
        all_bestPred_crystals,
        target_noisy_xrd, final_pred_xrds, curr_gt_crystal, gt_atom_types,
        gt_material_filepath, gt_xrd_filepath,
        all_xrd_l1_errors, all_xrd_l2_errors, all_composition_errors, has_correct_num_atoms):

    opt_material_folder_cand = f'{opt_material_folder}/{subdir}'
    opt_xrd_folder_cand = f'{opt_xrd_folder}/{subdir}'
    pred_opt_xrd_folder_cand = f'{pred_opt_xrd_folder}/{subdir}'
    opt_cif_folder_cand = f'{opt_cif_folder}/{subdir}'
    for the_folder in [opt_material_folder_cand, opt_xrd_folder_cand, opt_cif_folder_cand, pred_opt_xrd_folder_cand]:
        os.makedirs(the_folder, exist_ok=True) 

    candidate_xrd_l1_errors = list()
    candidate_xrd_l2_errors = list()
    candidate_match_status = list()
    candidate_composition_errors = list()
    candidate_has_correct_num_atoms = list()

    print(f'crystal {j} has {len(min_loss_indices)} candidates')
    best_rms_dist = 1e6
    # By default, log best (lowest loss) crystal for metrics
    best_crystal = Crystal(curr_gen_crystals_list[min_loss_indices[0]])
    for i, min_loss_idx in enumerate(min_loss_indices): # for each candidate
        filename = f'candidate_{i}.png'
        # construct the corresponding crystal
        opt_coords = all_opt_coords[min_loss_idx]
        opt_atom_types = all_opt_atom_types[min_loss_idx]
        opt_xrd = opt_generated_xrds[min_loss_idx, :].cpu().numpy()
        curr_pred_crystal = Crystal(curr_gen_crystals_list[min_loss_idx])

        # save the optimal crystal and its xrd
        pred_material_filepath = plot_material_single(opt_coords, opt_atom_types, opt_material_folder_cand, idx=j, filename=filename)
        pred_xrd_filepath = plot_xrd_single(xrd_args, opt_xrd, opt_xrd_folder_cand, idx=j, filename=filename)
        torch.save(opt_generated_xrds[min_loss_idx, :], os.path.join(opt_xrd_folder_cand, f'candidate_{i}.pt'))
        pred_opt_xrd_filepath = plot_xrd_single(xrd_args, final_pred_xrds[min_loss_idx].detach().cpu().numpy(), pred_opt_xrd_folder_cand, idx=j, filename=filename)
        torch.save(final_pred_xrds[min_loss_idx].detach(), os.path.join(pred_opt_xrd_folder_cand, f'candidate_{i}.pt'))
        curr_pred_crystal.structure.to(filename=f'{opt_cif_folder_cand}/material{j}_candidate{i}.cif', fmt='cif')

        # Log image
        log_img = collate_images(gt_material=gt_material_filepath, gt_xrd=gt_xrd_filepath,
                                pred_material=pred_material_filepath, pred_xrd=pred_xrd_filepath, width=600)
        wandb.log({"prediction": wandb.Image(log_img)})  

        # metrics
        assert target_noisy_xrd.squeeze().shape == opt_generated_xrds[min_loss_idx].squeeze().shape
        xrd_l1_error = F.l1_loss(target_noisy_xrd.squeeze(), opt_generated_xrds[min_loss_idx].squeeze()).item()
        xrd_l2_error = F.mse_loss(target_noisy_xrd.squeeze(), opt_generated_xrds[min_loss_idx].squeeze()).item()
        candidate_xrd_l1_errors.append(xrd_l1_error)
        candidate_xrd_l2_errors.append(xrd_l2_error)
        print(f'xrd l1 error: {xrd_l1_error}')
        print(f'xrd l2 error: {xrd_l2_error}')

        composition_error = compare_composition(gt_atom_types, opt_atom_types)
        candidate_composition_errors.append(composition_error)
        print(f'composition error: {composition_error}')

        is_num_atoms_correct = compare_num_atoms(gt_atom_types=gt_atom_types, 
                                                pred_atom_types=opt_atom_types)
        candidate_has_correct_num_atoms.append(int(is_num_atoms_correct))
        print(f'num atoms: {len(gt_atom_types)} (gt) vs {len(opt_atom_types)} (pred)')

        # Check if this matches
        curr_match_stats = check_structure_match(
            gt_structures=[curr_gt_crystal], 
            pred_structures=[curr_pred_crystal])
        candidate_match_status.append(curr_match_stats)

        # Pick crystal with lowest RMS dist as our candidate
        if curr_match_stats[MATCH_RATE] > 0.5 and curr_match_stats[RMS_DIST] < best_rms_dist:
            assert int(curr_match_stats[MATCH_RATE]) == 1
            best_rms_dist = curr_match_stats[RMS_DIST]
            best_crystal = curr_pred_crystal

        plot_overlaid_graphs(actual=target_noisy_xrd.squeeze().detach().cpu().numpy(), 
            prediction_nn=final_pred_xrds[min_loss_idx].detach().cpu().numpy(),
            prediction_simulated=opt_xrd,
            savepath=f'{opt_xrd_folder_cand}/candidate_{i}_overlaidXRD.png')
    
    # Log the crystal with lowest RMS dist
    all_bestPred_crystals.append(best_crystal)

    curr_material_metrics = {
        AVG_XRD_MSE: np.mean(candidate_xrd_l2_errors),
        AVG_XRD_L1: np.mean(candidate_xrd_l1_errors),
        BEST_XRD_MSE: np.min(candidate_xrd_l2_errors),
        BEST_XRD_L1: np.min(candidate_xrd_l1_errors),
        MATCH_RATE: candidate_match_status
    }

    with open(f'{metrics_folder}/material{j}.json', 'w') as fout:
        json.dump(curr_material_metrics, fout, indent=4)
    print(json.dumps(curr_material_metrics, indent=4))

    all_xrd_l1_errors.append(candidate_xrd_l1_errors)
    all_xrd_l2_errors.append(candidate_xrd_l2_errors)
    all_composition_errors.append(candidate_composition_errors)
    has_correct_num_atoms.append(candidate_has_correct_num_atoms)

    wandb.finish() 
    return

def create_xrd_args(args):
    alt_args = SimpleNamespace()
    alt_args.wave_source = 'CuKa'
    alt_args.num_materials = args.num_starting_points
    alt_args.xrd_vector_dim = 4096
    alt_args.max_theta = 180
    alt_args.min_theta = 0

    return alt_args

def smooth_xrds(opt_generated_xrds, data_loader):
    smoothed_xrds = list()
    for i in range(opt_generated_xrds.shape[0]):
        smoothed_xrd = data_loader.dataset.augment_xrdStrip(torch.tensor(opt_generated_xrds[i,:]))
        smoothed_xrds.append(smoothed_xrd)
    opt_generated_xrds = torch.stack(smoothed_xrds, dim=0)
    return opt_generated_xrds

def plot_filter(filter, angs, filter_viz_folder):
    dtheta = (angs[-1] - angs[0])/filter.shape[0]
    # plot filter
    plt.figure()
    plt.plot(angs, filter)
    plt.xlabel('theta (rad)')
    plt.ylabel('filter value')
    plt.title('Sinc filter (spatial domain)')
    plt.savefig(f'{filter_viz_folder}/filter_spatial.png')

    # plot filter in frequency domain
    # inverse shift the signal and fourier transform to freq domain
    F = np.fft.fft(np.fft.ifftshift(filter))
    # shift the signal back in freq domain
    F_shifted = np.fft.fftshift(F)
    # calculate frequency bins
    freq = np.fft.fftfreq(filter.shape[0], dtheta)
    # shift the frequencies
    freq_shifted = np.fft.fftshift(freq)
    # scale and plot
    plt.figure()
    plt.plot(freq_shifted, dtheta * np.real(F_shifted))
    plt.xlabel('Frequency (1/rad)')
    plt.ylabel('Amplitude')
    plt.title('Sinc filter (frequency domain)')
    plt.savefig(f'{filter_viz_folder}/filter_freq.png')
    plt.close()

def optimization(args, model, ld_kwargs, data_loader):
    assert data_loader is not None
    
    # assert filtering matches the configs
    assert args.xrd_filter == data_loader.dataset.xrd_filter, "XRD filter in config does not match the one in the dataset"

    base_output_dir = f'materials_viz/{args.label}'
    os.makedirs(base_output_dir, exist_ok=True)
    with open(os.path.join(base_output_dir, 'parameters.json'), 'w') as fout:
        json.dump(vars(args), fout, indent=4)

    opt_material_folder = f'{base_output_dir}/opt_material'
    opt_xrd_folder = f'{base_output_dir}/opt_xrd'
    pred_opt_xrd_folder = f'{base_output_dir}/pred_opt_xrd'
    opt_cif_folder = f'{base_output_dir}/opt_cif'
    gt_material_folder = f'{base_output_dir}/base_truth_material'
    gt_xrd_folder = f'{base_output_dir}/base_truth_xrd'
    gt_cif_folder = f'{base_output_dir}/base_truth_cif'
    filter_viz_folder = f'{base_output_dir}/filter_viz'
    metrics_folder = f'{base_output_dir}/metrics'
    for the_folder in [opt_material_folder, opt_xrd_folder, opt_cif_folder, 
                       gt_material_folder, gt_xrd_folder, gt_cif_folder, metrics_folder, filter_viz_folder]:
        os.makedirs(the_folder, exist_ok=True)

    # visualize filter and transform
    if args.xrd_filter == 'sinc' or args.xrd_filter == 'both':
        angs, sinc_filter = data_loader.dataset.angs, data_loader.dataset.sinc_filt
        plot_filter(sinc_filter, angs, filter_viz_folder)

    all_gt_crystals = list()
    all_bestPred_crystals = list()

    all_composition_errors = list()
    all_xrd_l1_errors = list()
    all_xrd_l2_errors = list()
    has_correct_num_atoms = list()

    spacegroups = list()
    formula_strs = list()
    mpids = list()

    for j, batch in enumerate(data_loader):
        wandb.init(config=args, project='new conditional generation', name=f'crystal {j}', group=args.label)
        if j == args.num_tested_materials:
            break
        batch = batch.to(model.device)

        spacegroups.append(int(batch.spacegroup[0]))
        formula_strs.append(batch.pretty_formula[0])
        mpids.append(batch.mpid[0])
        
        # get xrd
        target_noisy_xrd = batch.y.reshape(1, 512)
        z = optimize_latent_code(args=args, model=model, batch=batch, target_noisy_xrd=target_noisy_xrd)

        # get predicted xrd for all optimized candidates
        final_pred_xrds = model.fc_property(z).reshape(-1, 512)

        # TODO: speed this one up
        init_num_atoms = batch.num_atoms.repeat(args.num_starting_points) if args.num_atom_lambda > EPS else None
        init_atom_types = batch.atom_types.repeat(args.num_starting_points) if args.composition_lambda > EPS else None
        print('know num atoms:', init_num_atoms is not None)
        print('know atom types:', init_atom_types is not None)
        
        crystals = model.langevin_dynamics(z, ld_kwargs, gt_num_atoms=init_num_atoms, gt_atom_types=init_atom_types)
        crystals = {k: crystals[k] for k in ['frac_coords', 'atom_types', 'num_atoms', 'lengths', 'angles']}

        xrd_args = create_xrd_args(args)
            
        # predictions
        frac_coords = crystals['frac_coords']
        num_atoms = crystals['num_atoms']
        atom_types = crystals['atom_types']
        lengths = crystals['lengths']
        angles = crystals['angles']

        all_opt_coords, all_opt_atom_types, opt_generated_xrds, curr_gen_crystals_list = create_materials(xrd_args, 
                frac_coords, num_atoms, atom_types, lengths, angles, create_xrd=True)

        # plot base truth
        frac_coords = batch.frac_coords
        num_atoms = batch.num_atoms
        atom_types = batch.atom_types
        lengths = batch.lengths
        angles = batch.angles

        assert num_atoms.shape[0] == 1
        assert frac_coords.shape[0] == atom_types.shape[0]

        the_coords, atom_types, bt_generated_xrds, singleton_gt_crystal_list = create_materials(xrd_args, 
                frac_coords, num_atoms, atom_types, lengths, angles, create_xrd=True)
        the_coords = np.array(the_coords)[0]
        atom_types = np.array(atom_types)[0]

        assert len(singleton_gt_crystal_list) == 1
        curr_gt_crystal = Crystal(singleton_gt_crystal_list[0])
        all_gt_crystals.append(curr_gt_crystal)
        # save cif
        curr_gt_crystal.structure.to(filename=f'{gt_cif_folder}/material{j}_{mpids[-1]}_{formula_strs[-1]}.cif', fmt='cif')

        gt_material_filepath = plot_material_single(the_coords, atom_types, gt_material_folder, idx=j)
        gt_xrd_filepath = plot_xrd_single(xrd_args, target_noisy_xrd.squeeze().cpu().numpy(), gt_xrd_folder, idx=j)
        torch.save(target_noisy_xrd.squeeze().cpu(), os.path.join(gt_xrd_folder, f'material{j}.pt'))
        # apply smoothing to the XRD patterns
        opt_generated_xrds = smooth_xrds(opt_generated_xrds=opt_generated_xrds, data_loader=data_loader).to(model.device)

        # compute loss on desired and generated xrds
        target = target_noisy_xrd.broadcast_to(bt_generated_xrds.shape[0], 512).to(model.device)
        loss = F.l1_loss(opt_generated_xrds, target, reduction='none').mean(dim=-1) if args.l1_loss \
            else F.mse_loss(opt_generated_xrds, target, reduction='none').mean(dim=-1)
        
        # find the (num_candidates) minimum loss elements
        min_loss_indices = torch.argsort(loss).squeeze(0)[:args.num_candidates].tolist()

        # create material subdir
        subdir = f'material_{j}_{mpids[-1]}_{formula_strs[-1]}'

        process_candidates(args=args, xrd_args=xrd_args, j=j,
                curr_gen_crystals_list=curr_gen_crystals_list, 
                all_opt_coords=all_opt_coords, all_opt_atom_types=all_opt_atom_types, 
                opt_generated_xrds=opt_generated_xrds, 
                min_loss_indices=min_loss_indices, 
                opt_material_folder=opt_material_folder, opt_xrd_folder=opt_xrd_folder, pred_opt_xrd_folder=pred_opt_xrd_folder,
                opt_cif_folder=opt_cif_folder, metrics_folder=metrics_folder, subdir=subdir,
                all_bestPred_crystals=all_bestPred_crystals,
                target_noisy_xrd=target_noisy_xrd, final_pred_xrds=final_pred_xrds, curr_gt_crystal=curr_gt_crystal, gt_atom_types=atom_types,
                gt_material_filepath=gt_material_filepath, gt_xrd_filepath=gt_xrd_filepath,
                all_xrd_l1_errors=all_xrd_l1_errors, all_xrd_l2_errors=all_xrd_l2_errors, 
                all_composition_errors=all_composition_errors, has_correct_num_atoms=has_correct_num_atoms)

    ret_val = dict()
    for curr_spacegroup in set([USE_ALL_SPACEGROUPS] + spacegroups):
        curr_results = calculate_metrics(all_gt_crystals=all_gt_crystals, all_bestPred_crystals=all_bestPred_crystals,
            all_xrd_l1_errors=all_xrd_l1_errors, all_xrd_l2_errors=all_xrd_l2_errors, 
            all_composition_errors=all_composition_errors, has_correct_num_atoms=has_correct_num_atoms,
            spacegroups=spacegroups, desired_spacegroup=curr_spacegroup)
        ret_val[curr_spacegroup] = curr_results

    with open(f'{metrics_folder}/aggregate_metrics.json', 'w') as fout:
        json.dump(ret_val, fout, indent=4)
    
    print(json.dumps(ret_val, indent=4))

    return ret_val

def calculate_metrics(all_gt_crystals, all_bestPred_crystals,
            all_xrd_l1_errors, all_xrd_l2_errors, all_composition_errors, has_correct_num_atoms,
            spacegroups, desired_spacegroup):
    # turn into numpy arrays
    spacegroups = np.array(spacegroups)
    all_gt_crystals = np.array(all_gt_crystals)
    all_bestPred_crystals = np.array(all_bestPred_crystals)
    all_xrd_l1_errors = np.array(all_xrd_l1_errors)
    all_xrd_l2_errors = np.array(all_xrd_l2_errors)
    all_composition_errors = np.array(all_composition_errors)
    has_correct_num_atoms = np.array(has_correct_num_atoms)

    num_materials_in_spacegroup = len(spacegroups)
    if desired_spacegroup != USE_ALL_SPACEGROUPS:
        index_mask = spacegroups == desired_spacegroup
        assert len(index_mask) == len(spacegroups)
        assert np.sum(index_mask) > 0 and np.sum(index_mask) < len(spacegroups)

        all_gt_crystals = all_gt_crystals[index_mask]
        all_bestPred_crystals = all_bestPred_crystals[index_mask]
        all_xrd_l1_errors = all_xrd_l1_errors[index_mask]
        all_xrd_l2_errors = all_xrd_l2_errors[index_mask]
        all_composition_errors = all_composition_errors[index_mask]
        has_correct_num_atoms = has_correct_num_atoms[index_mask]

        num_materials_in_spacegroup = np.sum(index_mask)

    # average xrd errors
    avg_xrd_mse = np.mean(np.array(all_xrd_l2_errors))
    avg_xrd_l1 = np.mean(np.array(all_xrd_l1_errors))

    # best of candidate xrd errors
    best_xrd_mse = np.mean([np.min(list) for list in all_xrd_l2_errors])
    best_xrd_l1 = np.mean([np.min(list) for list in all_xrd_l1_errors])

    ret_val = {
        COUNT: int(num_materials_in_spacegroup),
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
    ret_val[NUM_ATOM_ACCURACY] = np.sum(has_correct_num_atoms) \
        / (has_correct_num_atoms.shape[0] * has_correct_num_atoms.shape[1])
    
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
    if args.start_from == 'data':
        loader = test_loader
    else:
        loader = None
    optimization(args=args, model=model, ld_kwargs=ld_kwargs, data_loader=loader) 
    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--data_dir', default='data', type=str)
    parser.add_argument('--n_step_each', default=100, type=int)
    parser.add_argument('--step_lr', default=1e-4, type=float)
    parser.add_argument('--save_traj', default=False, type=bool)
    parser.add_argument('--min_sigma', default=0, type=float)
    parser.add_argument('--disable_bar', default=False, type=bool)
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
    parser.add_argument('--xrd_filter', default='both')
    args = parser.parse_args()

    print('starting eval', args)
    main(args)

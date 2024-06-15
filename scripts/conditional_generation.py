import time
import argparse
import torch
import os
import json

import matplotlib
matplotlib.use('agg')
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
from cdvae.common.data_utils import get_scaler_from_data_list, build_crystal, build_crystal_graph
from visualization.visualize_materials import create_materials, plot_material_single, plot_xrd_single
from compute_metrics import Crystal, RecEval, GenEval
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts

from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.io.cif import CifWriter, CifParser

import wandb

from PIL import Image

AVG_COMPOSITION_ERROR = 'composition error rate'
AVG_XRD_MSE = 'Scaled XRD mean squared error'
AVG_XRD_L1 = 'Scaled XRD mean absolute error'
BEST_XRD_MSE = 'Average best scaled XRD mean squared error among candidates'
BEST_XRD_L1 = 'Average best scaled XRD mean absolute error among candidates'
AVG_PDF_CORRELATION = "Average Pearson's correlation coefficient between PDFs"
BEST_PDF_CORRELATION = "Mean best Pearson's correlation coefficient between PDFs"
STD_PDF_CORRELATION = "Std of Pearson's correlation coefficieint between PDFs"
STD_BEST_PDF_CORRELATION = "Std of best Pearson's correlation coefficient between PDFs"
PDF_CORRELATIONS = "All PDF correlations"
AVG_R_FACTOR = 'Average r factor'
BEST_R_FACTOR = 'Best r factor'
STD_R_FACTOR = 'Std of r factors'
STD_BEST_R_FACTOR = 'Std of best r factors'
R_FACTORS = "All r factors"

MATCH_RATE = 'match_rate'
RMS_DIST = 'rms_dist'
COMPOSITION_VALIDITY = 'comp_valid'
STRUCTURE_VALIDITY = 'struct_valid'
VALIDITY = 'valid'
NUM_ATOM_ACCURACY = '% materials w/ # atoms pred correctly'
PDF_CORRELATION = "pearson's r between PDFs"

USE_ALL_SPACEGROUPS = "aggregated stats (all spacegroups)"
COUNT = "number of crystals"

EPS = 1e-10

# Thanks ChatGPT!

# If you want to change the colors of the lines and shades, simply modify in the ax.fill_between() and ax.plot() functions
# A list of possible colors can be found at: https://matplotlib.org/stable/gallery/color/named_colors.html
def plot_overlaid_graphs(actual, prediction_nn, prediction_simulated, Qs, savepath):
    fig, ax = plt.subplots()

    # Plot and fill the area under the first curve
    ax.fill_between(Qs, actual, color="royalblue", alpha=0.2)
    ax.plot(Qs, actual, color="blue", alpha=0.6, label="Actual")  # Curve line

    # Plot and fill the area under the second curve
    ax.fill_between(Qs, prediction_nn, color="mistyrose", alpha=0.2)
    ax.plot(Qs, prediction_nn, color="red", alpha=0.6, linestyle='dotted', linewidth=2, label="Prediction (NN)")  # Dotted curve line with increased linewidth

    # Plot and fill the area under the second curve
    ax.fill_between(Qs, prediction_simulated, color="lightgreen", alpha=0.2)
    ax.plot(Qs, prediction_simulated, color="green", alpha=0.6, linestyle='dashed', linewidth=2, label="Prediction (Simulated)")  # Dotted curve line with increased linewidth

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

def plot_smoothed_vs_sinc(smoothed, sincPattern, noiselessPattern, Qs, savepath):
    fig, ax = plt.subplots()

    if not isinstance(smoothed, np.ndarray):
        smoothed = torch.clone(smoothed).squeeze().detach().cpu().numpy()
    if not isinstance(sincPattern, np.ndarray):
        sincPattern = torch.clone(sincPattern).squeeze().detach().cpu().numpy()
    if not isinstance(noiselessPattern, np.ndarray):
        noiselessPattern = torch.clone(noiselessPattern).squeeze().detach().cpu().numpy()

    assert Qs.shape == smoothed.shape

    # Plot and fill the area under the first curve
    #ax.fill_between(thetas, smoothed, color="hotpink", alpha=0.1)
    ax.plot(Qs, smoothed, color="deeppink", alpha=0.4, linestyle='dashed', label="Smoothed")

    # Plot and fill the area under the second curve
    #ax.fill_between(thetas, sincPattern, color="purple", alpha=0.2)
    ax.plot(Qs, sincPattern, color="indigo", alpha=0.6, label="Sinc (Raw Nanomaterial)")

    # Plot and fill the area under the second curve
    ax.plot(Qs, noiselessPattern, color="gray", alpha=0.8, label="Noiseless (Ideal Material)")      

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

def point_pdf_query(Qs, signal, r):
    ret_val = 0
    assert np.isclose(np.mean(signal), 1)
    delta_Q = (Qs[-1] - Qs[0]) / (Qs.shape[0] - 1)
    assert np.isclose(delta_Q, Qs[1] - Qs[0])
    for i in range(len(signal)):
        q = Qs[i]
        s_q = signal[i]
        ret_val += 2 / np.pi * q * (s_q - 1) * np.sin(q * r) * delta_Q
    return ret_val

def overall_pdf(Qs, signal, r_min=0, r_max=25, num_samples=1000):
    assert Qs.shape == signal.shape
    signal = signal / np.mean(signal)
    rs = np.linspace(r_min, r_max, num_samples)
    the_pdf = list()
    for r in rs:
        the_pdf.append(point_pdf_query(Qs=Qs, signal=signal, r=r))
    return np.array(rs), np.array(the_pdf)

def calc_r_factor(gt_xrd, pred_xrd, Qs):
    delta_Q = (Qs[-1] - Qs[0]) / (Qs.shape[0] - 1)
    assert np.isclose(delta_Q, Qs[1] - Qs[0])
    assert np.isclose(delta_Q, Qs[-1] - Qs[-2])
    if isinstance(gt_xrd, torch.Tensor):
        gt_xrd = gt_xrd.squeeze().detach().cpu().numpy()
    if isinstance(pred_xrd, torch.Tensor):
        pred_xrd = pred_xrd.squeeze().detach().cpu().numpy()
    assert np.isclose(np.max(gt_xrd), 1)
    assert np.isclose(np.max(pred_xrd), 1)
    assert np.min(gt_xrd) >= 0
    assert np.min(pred_xrd) >= 0
    numerator = np.sum(delta_Q * (gt_xrd - pred_xrd)**2)
    denominator = np.sum(delta_Q * gt_xrd**2)
    return numerator / denominator

def calc_and_plot_pdf_correlation(args, gt_xrd, pred_xrd, Qs, save_dir):
    # plot XRD
    if isinstance(gt_xrd, torch.Tensor):
        gt_xrd = gt_xrd.squeeze().detach().cpu().numpy()
    if isinstance(pred_xrd, torch.Tensor):
        pred_xrd = pred_xrd.squeeze().detach().cpu().numpy()
    assert gt_xrd.shape == pred_xrd.shape
    plt.plot(Qs, gt_xrd, alpha=0.8, label='GT XRD (noiseless)')
    plt.plot(Qs, pred_xrd, alpha=0.8, label='Pred XRD (noiseless)')
    plt.xlabel(r'$Q (\mathring A^{-1})$')
    plt.ylabel("Scaled Intensity")
    plt.title('XRD patterns')
    plt.grid()
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'xrd_comparison.png'))
    plt.savefig(os.path.join(save_dir, 'xrd_comparison.pdf'))
    plt.close()
    # create PDF
    gt_rs, gt_pdf = overall_pdf(Qs=Qs, signal=gt_xrd, r_min=args.r_min, r_max=args.r_max)
    pred_rs, pred_pdf = overall_pdf(Qs=Qs, signal=pred_xrd, r_min=args.r_min, r_max=args.r_max)
    assert np.array_equal(gt_rs, pred_rs)
    # plot PDF
    plt.plot(gt_rs, gt_pdf, alpha=0.8, label='GT PDF')
    plt.plot(pred_rs, pred_pdf, alpha=0.8, label='Pred PDF')
    plt.xlabel(r'$r (\mathring A)$')
    plt.ylabel("G(r)")
    # save PDF torch
    torch.save(torch.from_numpy(gt_pdf), os.path.join(save_dir, 'gt_pdf.pt'))
    torch.save(torch.from_numpy(pred_pdf), os.path.join(save_dir, 'pred_pdf.pt'))
    # calculate correlation coefficient
    correlation_matrix = np.corrcoef(gt_pdf, pred_pdf)
    pearson_r = correlation_matrix[0, 1]
    assert np.isclose(correlation_matrix[0, 1], correlation_matrix[1, 0])
    assert np.isclose(correlation_matrix[0, 0], 1) and np.isclose(correlation_matrix[1, 1], 1)
    # save PDF image
    plt.title(f"Pair Distribution Function Comparison\n(Pearson's r = {pearson_r:.3f})")
    plt.grid()
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'pdf_comparison.png'))
    plt.savefig(os.path.join(save_dir, 'pdf_comparison.pdf'))
    plt.close()
    # return
    return pearson_r

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

def optimize_latent_code(args, model, batch, target_noisy_xrd, z_init=None):
    m = MultivariateNormal(torch.zeros(model.hparams.hidden_dim).cuda(), 
                           torch.eye(model.hparams.hidden_dim).cuda())
    
    if z_init is None:
        assert args.start_from_init is None
        print('random z')
        z = torch.randn(args.num_starting_points, model.hparams.hidden_dim,
                    device=model.device)
    else:
        print(f'init z from pre-existing: {args.start_from_init}')
        z = z_init.detach()
        assert z.shape == (args.num_starting_points, model.hparams.hidden_dim)
    
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
        min_loss_indices, 
        curr_material_folder, 
        all_bestPred_crystals,
        gt_noiseless_xrd,
        target_noisy_xrd, final_pred_xrds, 
        opt_sinc_only_xrds, noiseless_generated_xrds,
        curr_gt_crystal, gt_atom_types,
        gt_material_filepath, gt_xrd_filepath,
        all_xrd_l1_errors, all_xrd_l2_errors, all_composition_errors, has_correct_num_atoms,
        all_pdf_correlations, all_r_factors, Qs):

    candidate_xrd_l1_errors = list()
    candidate_xrd_l2_errors = list()
    candidate_match_status = list()
    candidate_composition_errors = list()
    candidate_has_correct_num_atoms = list()
    candidate_pdf_correlations = list()
    candidate_r_factors = list()

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

        curr_candidate_folder = os.path.join(curr_material_folder, 'pred', f'candidate{i}') 
        # save the optimal crystal and its xrd
        opt_material_folder_cand = os.path.join(curr_candidate_folder, 'visUnitCell')
        os.makedirs(opt_material_folder_cand, exist_ok=True)
        pred_material_filepath = plot_material_single(opt_coords, opt_atom_types, opt_material_folder_cand, idx=j, filename=filename)
        opt_xrd_folder_cand = os.path.join(curr_candidate_folder, 'xrd_opt_gen')
        os.makedirs(opt_xrd_folder_cand, exist_ok=True)
        pred_xrd_filepath = plot_xrd_single(xrd_args, opt_xrd, opt_xrd_folder_cand, idx=j, filename=filename, x_axis=Qs, 
                                            x_label=r'Q $({A^{\circ}}^{-1})$')
        torch.save(opt_generated_xrds[min_loss_idx, :], os.path.join(opt_xrd_folder_cand, f'candidate_{i}.pt'))
        pred_opt_xrd_folder_cand = os.path.join(curr_candidate_folder, 'xrd_ml_pred')
        os.makedirs(pred_opt_xrd_folder_cand, exist_ok=True)
        pred_opt_xrd_filepath = plot_xrd_single(xrd_args, final_pred_xrds[min_loss_idx].detach().cpu().numpy(), 
                                                pred_opt_xrd_folder_cand, idx=j, 
                                                filename=filename, x_axis=Qs,
                                                x_label=r'Q $({A^{\circ}}^{-1})$')
        torch.save(final_pred_xrds[min_loss_idx].detach(), os.path.join(pred_opt_xrd_folder_cand, f'candidate_{i}.pt'))
        opt_cif_folder_cand = os.path.join(curr_candidate_folder, 'cif')
        os.makedirs(opt_cif_folder_cand, exist_ok=True)
        curr_pred_crystal.structure.to(filename=f'{opt_cif_folder_cand}/noSpacegroup_material{j}_candidate{i}.cif', fmt='cif')
        pred_cif_writer = CifWriter(curr_pred_crystal.structure, symprec=0.01)
        pred_cif_writer.write_file(filename=f'{opt_cif_folder_cand}/material{j}_candidate{i}.cif')

        # Log image
        log_img = collate_images(gt_material=gt_material_filepath, gt_xrd=gt_xrd_filepath,
                                pred_material=pred_material_filepath, pred_xrd=pred_xrd_filepath, width=600)
        wandb.log({"prediction": wandb.Image(log_img)})  

        # metrics
        assert target_noisy_xrd.squeeze().shape == opt_generated_xrds[min_loss_idx].squeeze().shape
        the_curr_opt_generated_xrd = opt_generated_xrds[min_loss_idx].to(target_noisy_xrd.device).squeeze()
        xrd_l1_error = F.l1_loss(target_noisy_xrd.squeeze(), the_curr_opt_generated_xrd).item()
        xrd_l2_error = F.mse_loss(target_noisy_xrd.squeeze(), the_curr_opt_generated_xrd).item()
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
            Qs=Qs,
            savepath=f'{opt_xrd_folder_cand}/overlaidXRD.png')
     
        # plot smoothed vs sinc: opt
        plot_smoothed_vs_sinc(smoothed=the_curr_opt_generated_xrd, 
                                sincPattern=opt_sinc_only_xrds[min_loss_idx], 
                                noiselessPattern=noiseless_generated_xrds[min_loss_idx],
                                Qs=Qs,
                                savepath=os.path.join(opt_xrd_folder_cand, f'sincVsSmoothed.png'))
    
        # compare and pair distribution functions
        opt_pdf_folder_cand = os.path.join(curr_candidate_folder, 'pdf')
        os.makedirs(opt_pdf_folder_cand, exist_ok=True)
        pdf_correlation = calc_and_plot_pdf_correlation(args=args,
                                                        gt_xrd=gt_noiseless_xrd, 
                                                        pred_xrd=noiseless_generated_xrds[min_loss_idx],
                                                        Qs=Qs,
                                                        save_dir=opt_pdf_folder_cand)
        candidate_pdf_correlations.append(pdf_correlation)
        print(f"pdf correlation: {pdf_correlation}")
        r_factor = calc_r_factor(gt_xrd=gt_noiseless_xrd, pred_xrd=noiseless_generated_xrds[min_loss_idx], Qs=Qs)
        candidate_r_factors.append(r_factor)
        print(f"r factor: {r_factor}")
    # Log the crystal with lowest RMS dist
    all_bestPred_crystals.append(best_crystal)

    curr_material_metrics = {
        AVG_XRD_MSE: np.mean(candidate_xrd_l2_errors),
        AVG_XRD_L1: np.mean(candidate_xrd_l1_errors),
        BEST_XRD_MSE: np.min(candidate_xrd_l2_errors),
        BEST_XRD_L1: np.min(candidate_xrd_l1_errors),
        MATCH_RATE: candidate_match_status,
        AVG_PDF_CORRELATION: np.mean(candidate_pdf_correlations),
        BEST_PDF_CORRELATION: np.max(candidate_pdf_correlations),
        AVG_R_FACTOR: np.mean(candidate_r_factors),
        BEST_R_FACTOR: np.max(candidate_r_factors),
        PDF_CORRELATIONS: candidate_pdf_correlations,
        R_FACTORS: candidate_r_factors
    }

    metrics_folder = os.path.join(curr_material_folder, 'metrics')
    os.makedirs(metrics_folder, exist_ok=True)
    with open(f'{metrics_folder}/material{j}.json', 'w') as fout:
        json.dump(curr_material_metrics, fout, indent=4)
    print(json.dumps(curr_material_metrics, indent=4))

    all_xrd_l1_errors.append(candidate_xrd_l1_errors)
    all_xrd_l2_errors.append(candidate_xrd_l2_errors)
    all_composition_errors.append(candidate_composition_errors)
    has_correct_num_atoms.append(candidate_has_correct_num_atoms)
    all_pdf_correlations.append(candidate_pdf_correlations)
    all_r_factors.append(candidate_r_factors)

    wandb.finish() 
    return

def write_histogram(values, save_folder, title, xlabel, ylabel, standard_range=True):
    plt.grid()
    if standard_range:
        plt.xticks(np.linspace(0, 1, 11))
        plt.xlim(0, 1)
        bins = np.linspace(0, 1, 21)
    else:
        plt.xticks(np.linspace(int(np.min(values)), int(np.max(values)), 11))
        plt.xlim(int(np.min(values)), int(np.max(values)))
        bins = np.linspace(int(np.min(values)), int(np.max(values)), 21)
    plt.hist(values, density=True, cumulative=True, bins=bins)
    plt.yticks(np.linspace(0, 1, 11))
    plt.ylim(0, 1)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(os.path.join(save_folder, f'{title}.png'))
    plt.savefig(os.path.join(save_folder, f'{title}.pdf'))
    plt.close()

    return

def write_pdf_histogram(pdf_rs, save_folder, title):
    xlabel = "Pearson's Correlation (r) between Predicted and GT PDFs"
    ylabel = "Cumulative Density\n(% of Materials at or below r)"
    write_histogram(values=pdf_rs, save_folder=save_folder, title=title, xlabel=xlabel, ylabel=ylabel)
    return

def write_r_factor_histogram(r_factors, save_folder, title):
    xlabel = "R-Factor (Residuals Function) between\nPredicted and GT XRDs (Noiseless)"
    ylabel = "Cumulative Density\n(% of Materials at or below R)"
    write_histogram(values=r_factors, save_folder=save_folder, title=title, xlabel=xlabel, ylabel=ylabel, standard_range=False)
    return

def create_xrd_args(args):
    alt_args = SimpleNamespace()
    alt_args.wave_source = args.wave_source
    alt_args.num_materials = args.num_starting_points
    alt_args.xrd_vector_dim = 4096
    alt_args.max_theta = 180
    alt_args.min_theta = 0

    return alt_args

def smooth_xrds(opt_generated_xrds, data_loader):
    smoothed_xrds = list()
    sinc_xrds = list()
    for i in range(opt_generated_xrds.shape[0]):
        smoothed_xrd, sincOnly, _, _ = data_loader.dataset.augment_xrdStrip(torch.tensor(opt_generated_xrds[i,:]), return_both=True)
        smoothed_xrds.append(smoothed_xrd)
        sinc_xrds.append(sincOnly)
    opt_generated_xrds = torch.stack(smoothed_xrds, dim=0)
    opt_sinc_xrds = torch.stack(sinc_xrds, dim=0)
    return opt_generated_xrds, opt_sinc_xrds

def plot_filter(filter, Qs, filter_viz_folder, nanomaterial_size):
    resolution = Qs.shape[0]
    Q_min = Qs[0]
    Q_max = Qs[-1]
    _, ax = plt.subplots()
    # sim_filter = nanomaterial_size * np.sinc((np.pi * nanomaterial_size * Qs)/(2 * np.pi))
    # sim_filter = sim_filter / np.max(sim_filter)
    filter = filter# / np.max(filter)
    # plot filter
    # ax.plot(Qs, sim_filter, alpha=0.5, label='simulated filter')
    ax.plot(Qs, filter, alpha=0.5) #, label='true filter')
    ax.set_xlabel(r'Q $({A^{\circ}}^{-1})$')
    ax.set_ylabel('Filter value')
    ax.grid(True)  # Show gridlines
    # ax.legend()
    plt.tight_layout()

    plt.savefig(f'{filter_viz_folder}/filter_Q.png')
    plt.savefig(f'{filter_viz_folder}/filter_Q.pdf')
    plt.close()

    # plot filter in spatial domain
    # inverse shift the signal and fourier transform to freq domain
    F = np.fft.ifft(np.fft.fftshift(filter))
    # shift the signal back in freq domain
    F_shifted = np.fft.ifftshift(F)
    # calculate frequency bins
    d = -(resolution - 1) / (2 * resolution * Q_min)

    spatial_bins = d * np.arange(resolution)
    spatial_bins_shifted = spatial_bins - d * resolution / 2
    # scale and plot
    _, ax = plt.subplots()
    ax.plot(spatial_bins_shifted, np.real(F_shifted))
    ax.set_xlim(-100, 100)
    ax.set_xlabel(r'Spatial $(A^{\circ})$')
    ax.set_ylabel('Amplitude')
 
    # ax.set_xticks(np.arange(-5, 5, 1))
    # ax.set_xticklabels(ax.get_xticks(), rotation=70)  # Rotate x-axis labels by 70 degrees
    # ax.set_yticks(np.arange(-0.01, 0.05, 0.01))  # Set horizontal gridlines every 0.1 from 0 to 1
    ax.grid(True)  # Show gridlines
    plt.tight_layout()

    plt.savefig(f'{filter_viz_folder}/filter_spatial.png')
    plt.savefig(f'{filter_viz_folder}/filter_spatial.pdf')
    plt.close()

def create_z_from_init(args, batch, model, cif_path):
    batch = batch.clone() # do not do the overwriting
    assert os.path.exists(cif_path), f'{cif_path} does not exist'

    with open(cif_path, 'r') as fin:
        cif_str = ''.join(fin.readlines())
        crystal = build_crystal(cif_str)
        frac_coords, atom_types, lengths, angles, edge_indices, to_jimages, num_atoms = \
            build_crystal_graph(crystal)

    batch.frac_coords = torch.Tensor(frac_coords).to(device=batch.frac_coords.device)
    batch.atom_types = torch.LongTensor(atom_types).to(device=batch.atom_types.device)
    batch.lengths = torch.Tensor(lengths).view(1, -1).to(device=batch.lengths.device)
    batch.angles = torch.Tensor(angles).view(1, -1).to(device=batch.angles.device)
    batch.edge_index = torch.LongTensor(edge_indices.T).contiguous().to(device=batch.edge_index.device)
    batch.to_jimages = torch.LongTensor(to_jimages).to(device=batch.to_jimages.device)
    batch.num_atoms = torch.tensor([num_atoms]).to(device=batch.num_atoms.device)
    batch.num_bonds = torch.tensor([edge_indices.shape[0]]).to(device=batch.num_bonds.device)

    mu, log_var, z = model.encode(batch)
    assert mu.shape == (1, model.hparams.hidden_dim), f"actually, mu's shape is: {mu.shape}"
    assert log_var.shape == (1, model.hparams.hidden_dim), f"actually, log_var's shape is: {log_var.shape}"
    assert z.shape == (1, model.hparams.hidden_dim), f"actually, z's shape is: {z.shape}"
    mu = mu.repeat(args.num_starting_points, 1)
    log_var = log_var.repeat(args.num_starting_points, 1)
    z = model.reparameterize(mu, log_var)
    assert z.shape == (args.num_starting_points, model.hparams.hidden_dim)

    return z

    # data = Data(
    #     frac_coords=torch.Tensor(frac_coords),
    #     atom_types=torch.LongTensor(atom_types),
    #     lengths=torch.Tensor(lengths).view(1, -1),
    #     angles=torch.Tensor(angles).view(1, -1),
    #     edge_index=torch.LongTensor(
    #         edge_indices.T).contiguous(),  # shape (2, num_edges)
    #     to_jimages=torch.LongTensor(to_jimages),
    #     num_atoms=num_atoms,
    #     num_bonds=edge_indices.shape[0],
    #     spacegroup=data_dict['spacegroup.number'],
    #     pretty_formula=data_dict['pretty_formula'],
    #     mpid=data_dict['mp_id'],
    #     num_nodes=num_atoms,  # special attribute used for batching in pytorch geometric
    #     y=prop,
    #     raw_sinc=raw_sinc,
    #     raw_sinc_presubsample=raw_sinc_presubsample,
    #     xrd_presubsample=xrd_presubsample,
    #     raw_xrd=torch.tensor(data_dict['rawXRD'])
    # )

def optimization(args, model, ld_kwargs, data_loader):
    assert data_loader is not None
    
    sample_factor = data_loader.dataset.n_presubsample // data_loader.dataset.n_postsubsample
    downsampled_Qs = data_loader.dataset.Qs[::sample_factor]

    # assert filtering matches the configs
    assert args.xrd_filter == data_loader.dataset.xrd_filter, "XRD filter in config does not match the one in the dataset"

    base_output_dir = f'{args.output_dir}/{args.label}'
    os.makedirs(base_output_dir, exist_ok=True)
    with open(os.path.join(base_output_dir, 'parameters.json'), 'w') as fout:
        json.dump(vars(args), fout, indent=4)

    filter_viz_folder = f'{base_output_dir}/filter_viz'
    os.makedirs(filter_viz_folder, exist_ok=True)

    # visualize filter and transform
    if args.xrd_filter == 'sinc' or args.xrd_filter == 'both':
        Qs_shifted, sinc_filter = data_loader.dataset.Qs_shifted, data_loader.dataset.sinc_filt
        plot_filter(filter=sinc_filter, Qs=Qs_shifted, 
                    filter_viz_folder=filter_viz_folder, 
                    nanomaterial_size=data_loader.dataset.nanomaterial_size)

    all_gt_crystals = list()
    all_bestPred_crystals = list()

    all_composition_errors = list()
    all_xrd_l1_errors = list()
    all_xrd_l2_errors = list()
    has_correct_num_atoms = list()
    all_pdf_correlations = list()
    all_r_factors = list()

    spacegroups = list()
    formula_strs = list()
    mpids = list()

    for j, batch in enumerate(data_loader):
        if j < args.first_idx:
            continue
        wandb.init(mode="disabled")
        # wandb.init(config=args, project='new conditional generation', name=f'crystal {j}', group=args.label)
        if j == args.num_tested_materials:
            break
        batch = batch.to(model.device)

        spacegroups.append(int(batch.spacegroup[0]))
        formula_strs.append(batch.pretty_formula[0])
        mpids.append(batch.mpid[0])

        curr_material_folder = f'{base_output_dir}/material{j}_{mpids[-1]}_{formula_strs[-1]}'
        os.makedirs(curr_material_folder, exist_ok=True)
        
        # get xrd
        assert data_loader.dataset.n_postsubsample == 512
        target_noisy_xrd = batch.y.reshape(1, 512)
        target_sincOnly = batch.raw_sinc.reshape(1, 512)

        raw_sinc = batch.raw_sinc.reshape(1, 512)
        gt_noiseless_xrd = batch.raw_xrd.reshape(1, 512)

        if args.start_from_init:
            init_material_folder = f'{args.start_from_init}/material{j}_{mpids[-1]}_{formula_strs[-1]}'
            cif_path = os.path.join(init_material_folder, 'pred', 'candidate0', 'cif', f'noSpacegroup_material{j}_candidate0.cif')
            z_init = create_z_from_init(args, batch, model, cif_path)
        else:
            z_init = None
        z = optimize_latent_code(args=args, model=model, batch=batch, 
                                 target_noisy_xrd=target_noisy_xrd, z_init=z_init)

        # get predicted xrd for all optimized candidates
        final_pred_xrds = model.fc_property(z).reshape(-1, 512)

        # TODO: speed this one up
        init_num_atoms = batch.num_atoms.repeat(args.num_starting_points) if args.num_atom_lambda > EPS else None
        init_atom_types = batch.atom_types.repeat(args.num_starting_points) if args.composition_lambda > EPS else None
        print('know num atoms:', init_num_atoms is not None)
        print('know atom types:', init_atom_types is not None)
        
        crystals = model.langevin_dynamics(z, ld_kwargs, gt_num_atoms=init_num_atoms, gt_atom_types=init_atom_types)
        if not args.save_traj:
            crystals = {k: crystals[k] for k in ['frac_coords', 'atom_types', 'num_atoms', 'lengths', 'angles']}
        else:
            crystals = {k: crystals[k] for k in ['frac_coords', 'atom_types', 'num_atoms', 'lengths', 'angles',
                                                 'all_frac_coords', 'all_atom_types']}
        xrd_args = create_xrd_args(args)
            
        # predictions
        frac_coords = crystals['frac_coords']
        num_atoms = crystals['num_atoms']
        atom_types = crystals['atom_types']
        lengths = crystals['lengths']
        angles = crystals['angles']

        all_opt_coords, all_opt_atom_types, opt_generated_xrds, curr_gen_crystals_list = create_materials(xrd_args, 
                frac_coords, num_atoms, atom_types, lengths, angles, create_xrd=True, symprec=0.01)

        # plot base truth
        frac_coords = batch.frac_coords
        num_atoms = batch.num_atoms
        atom_types = batch.atom_types
        lengths = batch.lengths
        angles = batch.angles

        assert num_atoms.shape[0] == 1
        assert frac_coords.shape[0] == atom_types.shape[0]

        the_coords, atom_types, bt_generated_xrds, singleton_gt_crystal_list = create_materials(xrd_args, 
                frac_coords, num_atoms, atom_types, lengths, angles, create_xrd=True, symprec=0.01)
        the_coords = np.array(the_coords)[0]
        atom_types = np.array(atom_types)[0]

        assert len(singleton_gt_crystal_list) == 1
        curr_gt_crystal = Crystal(singleton_gt_crystal_list[0])
        all_gt_crystals.append(curr_gt_crystal)
        # save cif
        gt_cif_folder = os.path.join(curr_material_folder, 'gt', 'cif')
        os.makedirs(gt_cif_folder, exist_ok=True)
        curr_gt_crystal.structure.to(filename=f'{gt_cif_folder}/noSpacegroup_material{j}_{mpids[-1]}_{formula_strs[-1]}.cif', fmt='cif')
        # TODO: this will sometimes change the # of atoms in the outputted unit cell in the cif file
        gt_cif_writer = CifWriter(curr_gt_crystal.structure, symprec=0.01)
        gt_cif_writer.write_file(filename=f'{gt_cif_folder}/material{j}_{mpids[-1]}_{formula_strs[-1]}.cif')
        # plot image
        gt_vis_folder = os.path.join(curr_material_folder, 'gt', 'visUnitCell')
        os.makedirs(gt_vis_folder, exist_ok=True)
        gt_material_filepath = plot_material_single(the_coords, atom_types, gt_vis_folder, idx=j)
        # plot xrd
        gt_xrd_folder = os.path.join(curr_material_folder, 'gt', 'xrd')
        os.makedirs(gt_xrd_folder, exist_ok=True)
        gt_xrd_filepath = plot_xrd_single(xrd_args, target_noisy_xrd.squeeze().cpu().numpy(), gt_xrd_folder, 
                                          idx=j, x_axis=downsampled_Qs,
                                          x_label=r'Q $({A^{\circ}}^{-1})$')
        # save the noisy xrd
        torch.save(target_noisy_xrd.squeeze().cpu(), os.path.join(gt_xrd_folder, f'material{j}.pt'))
        # save sinc only xrd
        torch.save(target_sincOnly.squeeze().cpu(), os.path.join(gt_xrd_folder, f'sincOnly{j}.pt'))
        # apply smoothing to the XRD patterns
        noiseless_generated_xrds = np.array([data_loader.dataset.sample(an_xrd) for an_xrd in opt_generated_xrds.tolist()])
        opt_generated_xrds, opt_sinc_only_xrds = smooth_xrds(opt_generated_xrds=opt_generated_xrds, data_loader=data_loader)
        opt_generated_xrds = opt_generated_xrds.to(model.device)
        opt_sinc_only_xrds = opt_sinc_only_xrds.to(model.device)
        assert noiseless_generated_xrds.shape == opt_generated_xrds.shape
        assert noiseless_generated_xrds.shape == opt_sinc_only_xrds.shape

        # plot smoothed vs sinc: gt
        plot_smoothed_vs_sinc(smoothed=target_noisy_xrd, sincPattern=raw_sinc, noiselessPattern=gt_noiseless_xrd,
                              Qs=downsampled_Qs, savepath=os.path.join(gt_xrd_folder, f'sincVsSmoothed{j}.png'))

        # compute loss on desired and generated xrds
        target = target_noisy_xrd.broadcast_to(bt_generated_xrds.shape[0], 512).to(model.device)
        loss = F.l1_loss(opt_generated_xrds.to(model.device), target.to(model.device), reduction='none').mean(dim=-1) if args.l1_loss \
            else F.mse_loss(opt_generated_xrds.to(model.device), target.to(model.device), reduction='none').mean(dim=-1)
        
        # find the (num_candidates) minimum loss elements
        min_loss_indices = torch.argsort(loss).squeeze(0)[:args.num_candidates].tolist()

        if args.save_traj:
            n_steps = args.n_step_each * len(model.sigmas)
            for item in crystals:
                print(item)
            assert crystals['all_frac_coords'].shape[0] == n_steps, f"{crystals['all_frac_coords'].shape[0]} != {n_steps}"
            assert crystals['all_atom_types'].shape[0] == n_steps, f"{crystals['all_atom_types'].shape[0]} != {n_steps}"
            print(f'{n_steps} total steps: save traj')

            traj_folder = os.path.join(curr_material_folder, 'pred', 'diffusion_vis')
            os.makedirs(traj_folder, exist_ok=True)

            for step in range(0, n_steps, args.n_step_each):
                curr_frac_coords = crystals['all_frac_coords'][step]
                curr_atom_types = crystals['all_atom_types'][step]

                assert curr_frac_coords.shape == crystals['frac_coords'].shape, f"{curr_frac_coords.shape} != {crystals['frac_coords'].shape}"
                assert curr_atom_types.shape == crystals['atom_types'].shape, f"{curr_atom_types.shape} != {crystals['atom_types'].shape}"

                curr_step_coords, curr_step_atom_types, curr_step_xrds, curr_step_singleton_crystal_list = \
                    create_materials(xrd_args, curr_frac_coords, crystals['num_atoms'], curr_atom_types, crystals['lengths'], crystals['angles'], 
                                     create_xrd=True, symprec=0.01)
                assert len(curr_step_singleton_crystal_list) == args.num_starting_points
                curr_step_crystal = Crystal(curr_step_singleton_crystal_list[min_loss_indices[0]]) # just save 1

                # TODO: save these
                curr_step_crystal.structure.to(filename=f'{traj_folder}/step{step}_material{j}_candidate{0}_{mpids[-1]}_{formula_strs[-1]}.cif', fmt='cif')
    
        process_candidates(args=args, xrd_args=xrd_args, j=j,
                curr_gen_crystals_list=curr_gen_crystals_list, 
                all_opt_coords=all_opt_coords, all_opt_atom_types=all_opt_atom_types, 
                opt_generated_xrds=opt_generated_xrds, 
                min_loss_indices=min_loss_indices, 
                curr_material_folder=curr_material_folder,
                all_bestPred_crystals=all_bestPred_crystals,
                gt_noiseless_xrd=gt_noiseless_xrd,
                target_noisy_xrd=target_noisy_xrd, final_pred_xrds=final_pred_xrds, 
                opt_sinc_only_xrds=opt_sinc_only_xrds, noiseless_generated_xrds=noiseless_generated_xrds,
                curr_gt_crystal=curr_gt_crystal, gt_atom_types=atom_types,
                gt_material_filepath=gt_material_filepath, gt_xrd_filepath=gt_xrd_filepath,
                all_xrd_l1_errors=all_xrd_l1_errors, all_xrd_l2_errors=all_xrd_l2_errors, 
                all_composition_errors=all_composition_errors, has_correct_num_atoms=has_correct_num_atoms,
                all_pdf_correlations=all_pdf_correlations, all_r_factors=all_r_factors,
                Qs=downsampled_Qs)

    ret_val = dict()
    for curr_spacegroup in set([USE_ALL_SPACEGROUPS] + spacegroups):
        curr_results = calculate_metrics(all_gt_crystals=all_gt_crystals, all_bestPred_crystals=all_bestPred_crystals,
            all_xrd_l1_errors=all_xrd_l1_errors, all_xrd_l2_errors=all_xrd_l2_errors, 
            all_composition_errors=all_composition_errors, has_correct_num_atoms=has_correct_num_atoms,
            all_pdf_correlations=all_pdf_correlations, all_r_factors=all_r_factors,
            spacegroups=spacegroups, desired_spacegroup=curr_spacegroup)
        ret_val[curr_spacegroup] = curr_results

    metrics_folder = os.path.join(base_output_dir, 'metrics')
    os.makedirs(metrics_folder, exist_ok=True)
    with open(f'{metrics_folder}/aggregate_metrics.json', 'w') as fout:
        json.dump(ret_val, fout, indent=4)
    
    write_pdf_histogram(pdf_rs=np.array(all_pdf_correlations).flatten(), save_folder=metrics_folder, title='All Predicted PDFs vs. Ground Truth')
    write_pdf_histogram(pdf_rs=np.max(np.array(all_pdf_correlations), axis=1), save_folder=metrics_folder, title='Best PDFs (per Material) vs. Ground Truth')

    write_r_factor_histogram(r_factors=np.array(all_r_factors).flatten(), save_folder=metrics_folder, title='All Predicted R-Factors vs. Ground Truth')
    write_r_factor_histogram(r_factors=np.max(np.array(all_r_factors), axis=1), save_folder=metrics_folder, title='Best R-Factors (per Material) vs. Ground Truth')

    print(json.dumps(ret_val, indent=4))

    return ret_val

def calculate_metrics(all_gt_crystals, all_bestPred_crystals,
            all_xrd_l1_errors, all_xrd_l2_errors, all_composition_errors, has_correct_num_atoms,
            all_pdf_correlations, all_r_factors, spacegroups, desired_spacegroup):
    # turn into numpy arrays
    spacegroups = np.array(spacegroups)
    all_gt_crystals = np.array(all_gt_crystals)
    all_bestPred_crystals = np.array(all_bestPred_crystals)
    all_xrd_l1_errors = np.array(all_xrd_l1_errors)
    all_xrd_l2_errors = np.array(all_xrd_l2_errors)
    all_composition_errors = np.array(all_composition_errors)
    has_correct_num_atoms = np.array(has_correct_num_atoms)
    all_pdf_correlations = np.array(all_pdf_correlations)
    all_r_factors = np.array(all_r_factors)

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
        all_pdf_correlations = all_pdf_correlations[index_mask]
        all_r_factors = all_r_factors[index_mask]

        num_materials_in_spacegroup = np.sum(index_mask)

    # average xrd errors
    avg_xrd_mse = np.mean(all_xrd_l2_errors)
    avg_xrd_l1 = np.mean(all_xrd_l1_errors)
    avg_pdf_correlation = np.mean(all_pdf_correlations)
    std_pdf_correlation = np.std(all_pdf_correlations)
    avg_r_factor = np.mean(all_r_factors)
    std_r_factor = np.std(all_r_factors)

    # best of candidate xrd errors
    best_xrd_mse = np.mean([np.min(list) for list in all_xrd_l2_errors])
    best_xrd_l1 = np.mean([np.min(list) for list in all_xrd_l1_errors])
    best_pdf_correlation = np.mean(np.max(all_pdf_correlations, axis=1))
    std_best_pdf_correlation = np.std(np.max(all_pdf_correlations, axis=1))
    best_r_factor = np.mean(np.min(all_r_factors, axis=1)) 
    std_best_r_factor = np.std(np.min(all_r_factors, axis=1))

    ret_val = {
        COUNT: int(num_materials_in_spacegroup),
        AVG_COMPOSITION_ERROR: np.mean(all_composition_errors),
        AVG_XRD_MSE: avg_xrd_mse,
        AVG_XRD_L1: avg_xrd_l1,
        BEST_XRD_MSE: best_xrd_mse,
        BEST_XRD_L1: best_xrd_l1,
        AVG_PDF_CORRELATION: avg_pdf_correlation,
        BEST_PDF_CORRELATION: best_pdf_correlation,
        STD_PDF_CORRELATION: std_pdf_correlation,
        STD_BEST_PDF_CORRELATION: std_best_pdf_correlation,
        AVG_R_FACTOR: avg_r_factor,
        STD_R_FACTOR: std_r_factor,
        BEST_R_FACTOR: best_r_factor,
        STD_BEST_R_FACTOR: std_best_r_factor
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
    structure_evaluator = RecEval(pred_crys=pred_structures, gt_crys=gt_structures, 
                                  stol=0.3, angle_tol=5, ltol=0.2)
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
    parser.add_argument('--save_traj', action='store_true')
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
    parser.add_argument('--output_dir', default='materials_viz', type=str)
    parser.add_argument('--first_idx', default=0, type=int)
    parser.add_argument('--r_min', default=0, type=float)
    parser.add_argument('--r_max', default=30, type=float)
    parser.add_argument('--wave_source', default='CuKa', type=str)
    parser.add_argument('--start_from_init', default=None, type=str)
    args = parser.parse_args()

    print('starting eval', args)
    main(args)

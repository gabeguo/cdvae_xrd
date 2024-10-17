from calculate_xrd_patterns_post_hoc import generate_and_save_xrd_for_cif_file, \
    create_sinc_filter
from calculate_r_factor_post_hoc import calc_r_factor
import os
import argparse
from tqdm import tqdm
import torch
import json
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as ticker

WAVE_SOURCE = 'CuKa'
XRD_VECTOR_DIM = 8192
MIN_2_THETA = 0
MAX_2_THETA = 180
NANOMATERIAL_SIZE = 10
PLOT_XRD = True
BROADENING = 'gaussian'
GAUSSIAN_SIGMA_FRAC = 5e-3

def main(args):
    os.makedirs(args.output_dir, exist_ok=True)
    process_unrefined_dir(args)
    process_refined_dir(args)
    sinc_r_values, refined_r_values = calc_all_r_factors(args)
    plot_r_values(args, sinc_r_values, refined_r_values)
    save_r_values(args, sinc_r_values, refined_r_values)
    return

def process_unrefined_dir(args):
    assert os.path.exists(args.unrefined_input_directory)
    for material_folder in tqdm(os.listdir(args.unrefined_input_directory)):
        if not ('material' in material_folder and 'mp-' in material_folder):
            continue
        # ground truth CIF
        gt_cif_filepath = os.path.join(args.unrefined_input_directory, material_folder,
                                    f"{material_folder}_gt.cif")
        assert os.path.exists(gt_cif_filepath)

        the_number = int(material_folder.split('_')[0][len('material'):])
        gt_output_filepath_without_ext = os.path.join(args.output_dir, f"{the_number}_gt")
        generate_and_save_xrd_for_cif_file(
            cif_filepath=gt_cif_filepath,
            output_filepath_without_ext=gt_output_filepath_without_ext,
            wave_source=WAVE_SOURCE, xrd_vector_dim=XRD_VECTOR_DIM, 
            min_2_theta=MIN_2_THETA, max_2_theta=MAX_2_THETA,
            nanomaterial_size=NANOMATERIAL_SIZE, 
            plot_xrd=PLOT_XRD,
            broadening=BROADENING, gaussian_sigma_frac=GAUSSIAN_SIGMA_FRAC
        )

        # sinc CIF
        pred_cif_filepath = None
        for candidate in os.listdir(os.path.join(args.unrefined_input_directory, material_folder)):
            if material_folder in candidate and '.cif' in candidate and 'candidate' in candidate:
                pred_cif_filepath = os.path.join(args.unrefined_input_directory, material_folder, candidate)
                assert os.path.exists(pred_cif_filepath)
                candidate_number = get_candidate_number(candidate)
                pred_output_filepath_without_ext = os.path.join(args.output_dir, f"{the_number}_{candidate_number}_pred_sinc{args.sinc_level}_")
                generate_and_save_xrd_for_cif_file(
                    cif_filepath=pred_cif_filepath,
                    output_filepath_without_ext=pred_output_filepath_without_ext,
                    wave_source=WAVE_SOURCE, xrd_vector_dim=XRD_VECTOR_DIM, 
                    min_2_theta=MIN_2_THETA, max_2_theta=MAX_2_THETA,
                    nanomaterial_size=NANOMATERIAL_SIZE, 
                    plot_xrd=PLOT_XRD,
                    broadening=BROADENING, gaussian_sigma_frac=GAUSSIAN_SIGMA_FRAC
                )
        assert pred_cif_filepath is not None, material_folder

    return

def get_candidate_number(cif_filename):
    candidate_number_start_idx = cif_filename.index('candidate') + len('candidate')
    candidate_number_end_idx = cif_filename.index('.cif')
    candidate_number = int(cif_filename[candidate_number_start_idx:candidate_number_end_idx])
    return candidate_number

def process_refined_dir(args):
    assert os.path.exists(args.refined_input_directory)
    for cif_filename in tqdm(os.listdir(args.refined_input_directory)):
        if not('candidate' in cif_filename and '.cif' in cif_filename):
            continue
        item_number = int(cif_filename.split('_')[0][len('material'):])
        candidate_number = get_candidate_number(cif_filename)
        cif_filepath = os.path.join(args.refined_input_directory, cif_filename)
        output_filepath_without_ext = os.path.join(args.output_dir, f"{item_number}_{candidate_number}_refined")
        generate_and_save_xrd_for_cif_file(
            cif_filepath=cif_filepath,
            output_filepath_without_ext=output_filepath_without_ext,
            wave_source=WAVE_SOURCE, xrd_vector_dim=XRD_VECTOR_DIM, 
            min_2_theta=MIN_2_THETA, max_2_theta=MAX_2_THETA,
            nanomaterial_size=NANOMATERIAL_SIZE, 
            plot_xrd=PLOT_XRD,
            broadening=BROADENING, gaussian_sigma_frac=GAUSSIAN_SIGMA_FRAC
        )
    return

def calc_all_r_factors(args):
    #print('calculating r factors')
    # find Qs
    Qs, _ = create_sinc_filter(
        wave_source=WAVE_SOURCE, 
        min_2_theta=MIN_2_THETA, 
        max_2_theta=MAX_2_THETA, 
        xrd_vector_dim=XRD_VECTOR_DIM, 
        nanomaterial_size=NANOMATERIAL_SIZE
    )

    # compare PXRDs
    gt_xrd_filenames = [x for x in os.listdir(args.output_dir) if '_gt.pt' in x]
    pred_xrd_filenames = [x for x in os.listdir(args.output_dir) if f'_pred_sinc{args.sinc_level}_.pt' in x]
    refined_xrd_filenames = [x for x in os.listdir(args.output_dir) if '_refined.pt' in x]
    assert len(pred_xrd_filenames) == len(refined_xrd_filenames)
    assert 10 * len(gt_xrd_filenames) == len(pred_xrd_filenames), f"{len(gt_xrd_filenames)} {len(pred_xrd_filenames)}"
    assert len(gt_xrd_filenames) == 20, gt_xrd_filenames

    sinc_r_values = dict()
    refined_r_values = dict()

    for prediction_level, r_value_dict, curr_possible_pred_filenames in \
            zip([f"pred_sinc{args.sinc_level}_", "refined"], 
                [sinc_r_values, refined_r_values],
                [pred_xrd_filenames, refined_xrd_filenames]):
        for curr_gt_filename in gt_xrd_filenames:
            for candidate_idx in range(0, 10):
                curr_pred_filename = curr_gt_filename.replace('gt', f"{candidate_idx}_{prediction_level}")
                assert curr_pred_filename in curr_possible_pred_filenames, f"{curr_pred_filename} not in {curr_possible_pred_filenames}"

                gt_xrd = torch.load(os.path.join(args.output_dir, curr_gt_filename))
                pred_xrd = torch.load(os.path.join(args.output_dir, curr_pred_filename))

                r = calc_r_factor(gt_xrd=gt_xrd, pred_xrd=pred_xrd, Qs=Qs)
                #print(f"\t{curr_gt_filename}: r = {r:.3f}")

                material_num = int(curr_gt_filename.split('_')[0])
                if material_num not in r_value_dict:
                    r_value_dict[material_num] = list()
                r_value_dict[material_num].append(r)

        # write results
        with open(os.path.join(args.output_dir, f'Rw_values_{prediction_level}.json'), 'w') as fout:
            json.dump({x:r_value_dict[x] for x in sorted(r_value_dict)}, fout, indent=4)
    
    with open(os.path.join(args.output_dir, 'params.json'), 'w') as fout:
        json.dump(vars(args), fout)

    return sinc_r_values, refined_r_values

def get_best_refinement(sinc_r_values, refined_r_values):
    """
    Takes the best candidate, according to r-value of refined version
    """
    selected_sinc_r_values = list()
    best_refined_r_values = list()
    assert sinc_r_values.keys() == refined_r_values.keys()
    for material_num in refined_r_values:
        best_idx = np.argmin(refined_r_values[material_num])
        
        selected_sinc_r_values.append(sinc_r_values[material_num][best_idx])
        best_refined_r_values.append(refined_r_values[material_num][best_idx])
    
    return selected_sinc_r_values, best_refined_r_values

def plot_r_values(args, sinc_r_values, refined_r_values):
    plt.rcParams["figure.figsize"] = (6, 6)
    plt.rc('axes', labelsize=18)
    plt.rc('xtick', labelsize=14)
    plt.rc('ytick', labelsize=14)
    # plot
    sinc_r_values, refined_r_values = get_best_refinement(sinc_r_values, refined_r_values)
    plt.plot(sinc_r_values, refined_r_values, 'o')
    plt.xlabel('$R_{wp}^{2}$: raw AI generation')
    if args.sinc_level == 100:
        plt.ylabel("     ")
    else:
        plt.ylabel('$R_{wp}^{2}$: after XRD refinement')
    generic_x_vals = np.linspace(0, args.thresh, 20)
    plt.fill_between(generic_x_vals, np.full_like(generic_x_vals, 0.0), np.full_like(generic_x_vals, 0.05), color='darkgreen', alpha=0.2)
    plt.fill_between(generic_x_vals, np.full_like(generic_x_vals, 0.05), np.full_like(generic_x_vals, 0.1), color='palegreen', alpha=0.2)
    plt.fill_between(generic_x_vals, np.full_like(generic_x_vals, 0.1), np.full_like(generic_x_vals, 0.2), color='yellow', alpha=0.2)
    plt.fill_between(generic_x_vals, np.full_like(generic_x_vals, 0.2), np.full_like(generic_x_vals, 0.4), color='orange', alpha=0.2)
    plt.fill_between(generic_x_vals, np.full_like(generic_x_vals, 0.4), np.full_like(generic_x_vals, args.thresh), color='red', alpha=0.2)
    plt.plot(generic_x_vals, generic_x_vals,
             'gray', alpha=0.6, marker='', linestyle='--', label='identity line')
    plt.xlim(0, args.thresh)
    plt.ylim(0, args.thresh)
    # Thanks https://stackoverflow.com/a/58675407
    major_tick_spacing = 0.2
    plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(major_tick_spacing))
    plt.gca().yaxis.set_major_locator(ticker.MultipleLocator(major_tick_spacing))
    minor_tick_spacing = 0.05
    plt.gca().xaxis.set_minor_locator(ticker.MultipleLocator(minor_tick_spacing))
    plt.gca().yaxis.set_minor_locator(ticker.MultipleLocator(minor_tick_spacing))
    plt.grid(which='minor', color='#CCCCCC')
    plt.grid(which='major', color='#777777')
    
    # save figure
    plt.tight_layout()
    plot_filepath = os.path.join(args.output_dir, 'regression_plot.pdf')
    plt.savefig(plot_filepath)
    plt.savefig(plot_filepath.replace('.pdf', '.png'), dpi=300)
    plt.close()

    return

def save_r_values(args, sinc_r_values, refined_r_values):
    sinc_r_values = np.array([x for x in sinc_r_values.values()])
    refined_r_values = np.array([x for x in refined_r_values.values()])
    assert sinc_r_values.shape == (20, 10), f"{sinc_r_values.shape}"
    assert refined_r_values.shape == (20, 10), f"{refined_r_values.shape}"
    with open(os.path.join(args.output_dir, 'r_value_comparison.json'), 'w') as fout:
        results = {
            'pre-refinement ALL r-value (mean)': np.mean(sinc_r_values),
            'pre-refinement ALL r-value (std)': np.std(sinc_r_values),
            'post-refinement ALL r-value (mean)': np.mean(refined_r_values),
            'post-refinement ALL r-value (std)': np.std(refined_r_values),
            'pre-refinement BEST r-value (mean)': np.mean([min(x) for x in sinc_r_values]),
            'pre-refinement BEST r-value (std)': np.std([min(x) for x in sinc_r_values]),
            'post-refinement BEST r-value (mean)': np.mean([min(x) for x in refined_r_values]),
            'post-refinement BEST r-value (std)': np.std([min(x) for x in refined_r_values])
        }
        json.dump(results, fout, indent=4)
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--unrefined_input_directory', type=str,
                        default='/home/gabeguo/cdvae_xrd/Data_to_Max_05_07_24')
    parser.add_argument('--refined_input_directory', type=str,
                        default='/home/gabeguo/refined_candidates_05_16_24_rVal/fitted/pred_10/sinc10_and_gaussian_refineUnitCell')
    parser.add_argument('--output_dir', type=str,
                        default='/home/gabeguo/refined_candidates_05_16_24_rVal/xrd_comparison/sinc10')
    parser.add_argument('--sinc_level', type=int, 
                        default=10)
    parser.add_argument('--thresh', type=float, default=1.4)
    args = parser.parse_args()

    main(args)
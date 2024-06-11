from calculate_xrd_patterns_post_hoc import generate_and_save_xrd_for_cif_file, \
    create_sinc_filter
from calculate_r_factor_post_hoc import calc_r_factor
import os
import argparse
from tqdm import tqdm
import torch
import json

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
    process_gt_dir(args)
    process_pred_dir(args)
    calc_all_r_factors(args)
    return

def process_gt_dir(args):
    assert os.path.exists(args.gt_input_directory)
    for material_folder in tqdm(os.listdir(args.gt_input_directory)):
        if not ('material' in material_folder and 'mp-' in material_folder):
            continue
        cif_filepath = os.path.join(args.gt_input_directory, material_folder,
                                    f"gt_{material_folder}.cif")
        assert os.path.exists(cif_filepath)

        the_number = int(material_folder.split('_')[0][len('material'):])
        output_filepath_without_ext = os.path.join(args.output_dir, f"{the_number}_gt")
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

def process_pred_dir(args):
    assert os.path.exists(args.pred_input_directory)
    for cif_filename in tqdm(os.listdir(args.pred_input_directory)):
        if '_fit.cif' not in cif_filename:
            continue
        the_number = int(cif_filename.split('_')[0])
        cif_filepath = os.path.join(args.pred_input_directory, cif_filename)
        output_filepath_without_ext = os.path.join(args.output_dir, f"{the_number}_pred")
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
    pred_xrd_filenames = [x for x in os.listdir(args.output_dir) if '_pred.pt' in x]
    assert len(gt_xrd_filenames) == len(pred_xrd_filenames)
    assert len(gt_xrd_filenames) == 20, gt_xrd_filenames
    material_to_r = dict()
    for curr_gt_filename in gt_xrd_filenames:
        curr_pred_filename = curr_gt_filename.replace('gt', 'pred')
        assert curr_pred_filename in pred_xrd_filenames, pred_xrd_filenames

        gt_xrd = torch.load(os.path.join(args.output_dir, curr_gt_filename))
        pred_xrd = torch.load(os.path.join(args.output_dir, curr_pred_filename))

        r = calc_r_factor(gt_xrd=gt_xrd, pred_xrd=pred_xrd, Qs=Qs)
        print(f"{curr_gt_filename}: r = {r:.3f}")

        material_num = int(curr_gt_filename.split('_')[0])
        material_to_r[material_num] = r

    # write results
    with open(os.path.join(args.output_dir, 'Rw_values.txt'), 'w') as fout:
        for material_num in sorted(material_to_r):
            fout.write(f"{material_num} {100 * material_to_r[material_num]}\n")
    with open(os.path.join(args.output_dir, 'params.json'), 'w') as fout:
        json.dump(vars(args), fout)
    
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--gt_input_directory', type=str,
                        default='/home/gabeguo/cdvae_xrd/Data_to_Max_05_07_24')
    parser.add_argument('--pred_input_directory', type=str,
                        default='/home/gabeguo/refined_candidates_05_16_24_rVal/fitted/pred_10/sinc10_and_gaussian_refineUnitCell')
    parser.add_argument('--output_dir', type=str,
                        default='/home/gabeguo/refined_candidates_05_16_24_rVal/xrd_comparison/sinc10')
    args = parser.parse_args()

    main(args)
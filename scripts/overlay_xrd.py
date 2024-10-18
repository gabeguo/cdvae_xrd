import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
from tqdm import tqdm
import json

def plot_xrds(theta, gt, ai_pred, refined, filepath, pre_r, post_r):
    plt.plot(theta, gt, label='Ground Truth', linestyle='-', color='#55aa55')
    plt.plot(theta, ai_pred, label=r'AI Raw Pred:   $r^2$ = {:.3f}'.format(pre_r), linestyle='dotted', color='#cc666699')
    plt.plot(theta, refined, label=r'AI + Rietveld: $r^2$ = {:.3f}'.format(post_r), linestyle='dashed', color='#5555aa')

    plt.xlabel(r"2$\theta$")
    plt.legend()
    plt.tight_layout()

    plt.savefig(filepath)
    plt.close()

    return

def load_xrd(filepath):
    thetas = list()
    intensities = list()
    with open(filepath, 'r') as fin:
        for line in fin.readlines():
            curr_theta, curr_i = line.split()
            thetas.append(float(curr_theta.strip()))
            intensities.append(float(curr_i.strip()))
    return thetas, intensities

def get_best_candidate_lookup(args):
    lookup = dict()
    with open(args.r_val_lookup, 'r') as fin:
        all_lines = fin.readlines()
        for the_line in all_lines:
            if the_line.strip() == "":
                continue
            the_name, the_r_val = the_line.split()
            if 'gt' in the_name:
                continue
            curr_material_num = int(the_name.split('_')[0][len("material"):])
            candidate_idx = int(the_name.split('_')[-1][len("candidate"):])
            the_r_val = float(the_r_val)
            if curr_material_num in lookup:
                if the_r_val < lookup[curr_material_num][1]:
                    lookup[curr_material_num] = (candidate_idx, the_r_val)
            else:
                lookup[curr_material_num] = (candidate_idx, the_r_val)
    return lookup

def main(args):
    # best_candidate_by_material = get_best_candidate_lookup(args)
    sinc_level = args.input_dir.split('/')[-1]
    with open(os.path.join(args.input_dir, f"Rw_values_pred_{sinc_level}_.json"), 'r') as fin:
        pre_refinement_rs = json.load(fin)
        pre_refinement_rs = {int(x):pre_refinement_rs[x] for x in pre_refinement_rs}
    with open(os.path.join(args.input_dir, f"Rw_values_refined.json"), 'r') as fin:
        post_refinement_rs = json.load(fin)
        post_refinement_rs = {int(x):post_refinement_rs[x] for x in post_refinement_rs}
    for filename in tqdm(os.listdir(args.input_dir)):
        if '_gt.xy' in filename:
            theta, gt_xrd = load_xrd(os.path.join(args.input_dir, filename))
            material_num = int(filename.split('_')[0])
            candidate_num = np.argmin(post_refinement_rs[material_num]) #best_candidate_by_material[material_num][0]
            # raw AI pred
            pred_filename = filename.replace('gt', f'{candidate_num}_pred_{sinc_level}_')
            pred_filepath = os.path.join(args.input_dir, pred_filename)
            assert os.path.exists(pred_filepath), f"{pred_filepath}"
            _, pred_xrd = load_xrd(pred_filepath)
            curr_pre_r = pre_refinement_rs[material_num][candidate_num]
            # Rietveld refined
            refined_filename = filename.replace('gt', f'{candidate_num}_refined')
            refined_filepath = os.path.join(args.input_dir, refined_filename)
            assert os.path.exists(refined_filepath)
            _, refined_xrd = load_xrd(refined_filepath)
            curr_post_r = post_refinement_rs[material_num][candidate_num]

            img_filepath = os.path.join(args.output_dir, f"{material_num}_{candidate_num}_xrdOverlaid.pdf")
            plot_xrds(theta=theta, gt=gt_xrd, ai_pred=pred_xrd, refined=refined_xrd, filepath=img_filepath, pre_r=curr_pre_r, post_r=curr_post_r)
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str,
                        default='/home/gabeguo/cdvae_xrd/paper_results_PRELIM/_MAX_REFINED/xrd_comparison/sinc100')
    parser.add_argument('--output_dir', type=str,
                        default='/home/gabeguo/cdvae_xrd/paper_results_PRELIM/_MAX_REFINED/xrd_comparison/plots/sinc100')
    parser.add_argument('--r_val_lookup', type=str,
                        default='/home/gabeguo/cdvae_xrd/paper_results_PRELIM/_MAX_REFINED/make_sincSq100_/Rw_values.txt')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    main(args)
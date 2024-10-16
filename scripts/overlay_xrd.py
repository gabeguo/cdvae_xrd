import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
from tqdm import tqdm

def plot_xrds(theta, gt, ai_pred, refined, filepath):
    plt.plot(theta, gt, label='Ground Truth', linestyle='-', color='#99ee99bb')
    plt.plot(theta, ai_pred, label='AI Raw Prediction', linestyle='dotted', color='#ee9999bb')
    plt.plot(theta, refined, label='AI + Rietveld', linestyle='dashed', color='#9999eebb')

    plt.xlabel(r'2$\\theta$')

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

def main(args):
    sinc_level = args.input_dir.split('/')[-1]
    for filename in tqdm(os.listdir(args.input_dir)):
        if '_gt.xy' in filename:
            theta, gt_xrd = load_xrd(os.path.join(args.input_dir, filename))
            material_num = int(filename.split('_')[0])
            for candidate_num in range(0, 10):
                # raw AI pred
                pred_filename = filename.replace('gt', f'{candidate_num}_pred_{sinc_level}_')
                pred_filepath = os.path.join(args.input_dir, pred_filename)
                assert os.path.exists(pred_filepath), f"{pred_filepath}"
                _, pred_xrd = load_xrd(pred_filepath)
                # Rietveld refined
                refined_filename = filename.replace('gt', f'{candidate_num}_refined')
                refined_filepath = os.path.join(args.input_dir, refined_filename)
                assert os.path.exists(refined_filepath)
                _, refined_xrd = load_xrd(refined_filepath)

                img_filepath = os.path.join(args.output_dir, f"{material_num}_{candidate_num}_xrdOverlaid.pdf")
                plot_xrds(theta=theta, gt=gt_xrd, ai_pred=pred_xrd, refined=refined_xrd, filepath=img_filepath)
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str,
                        default='/home/gabeguo/cdvae_xrd/paper_results_PRELIM/_MAX_REFINED/xrd_comparison/sinc100')
    parser.add_argument('--output_dir', type=str,
                        default='/home/gabeguo/cdvae_xrd/paper_results_PRELIM/_MAX_REFINED/xrd_comparison/plots/sinc100')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    main(args)
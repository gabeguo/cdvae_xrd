import numpy as np
import matplotlib.pyplot as plt
import os
import json
import torch
from tqdm import tqdm
from pymatgen.analysis.diffraction.xrd import WAVELENGTHS

def calc_r_factor(gt_xrd, pred_xrd, Qs):
    delta_Q = (Qs[-1] - Qs[0]) / (Qs.shape[0] - 1)
    assert np.isclose(delta_Q, Qs[1] - Qs[0])
    assert np.isclose(delta_Q, Qs[-1] - Qs[-2])
    if isinstance(gt_xrd, torch.Tensor):
        gt_xrd = gt_xrd.squeeze().detach().cpu().numpy()
    if isinstance(pred_xrd, torch.Tensor):
        pred_xrd = pred_xrd.squeeze().detach().cpu().numpy()
    assert np.isclose(np.max(gt_xrd), 1, atol=1e-2)
    assert np.isclose(np.max(pred_xrd), 1, atol=1e-2)
    assert np.min(gt_xrd) >= 0
    assert np.min(pred_xrd) >= 0
    numerator = np.sum(delta_Q * (gt_xrd - pred_xrd)**2)
    denominator = np.sum(delta_Q * gt_xrd**2)
    return float(numerator / denominator)

def plot_overlaid_graphs(actual, prediction_simulated, Qs, savepath):
    fig, ax = plt.subplots()
    if isinstance(actual, torch.Tensor):
        actual = actual.squeeze().detach().cpu().numpy()
    if isinstance(prediction_simulated, torch.Tensor):
        prediction_simulated = prediction_simulated.squeeze().detach().cpu().numpy()
    # Plot and fill the area under the first curve
    ax.fill_between(Qs, actual, color="royalblue", alpha=0.2)
    ax.plot(Qs, actual, color="blue", alpha=0.6, label="Actual")  # Curve line

    # Plot and fill the area under the second curve
    ax.fill_between(Qs, prediction_simulated, color="lightgreen", alpha=0.2)
    ax.plot(Qs, prediction_simulated, color="green", alpha=0.6, linestyle='dashed', linewidth=2, label="Prediction (Simulated)")  # Dotted curve line with increased linewidth

    # Customizing the plot
    ax.set_title("XRD Patterns")
    ax.set_xlabel(r'$Q (\mathring A^{-1})$')
    ax.set_ylabel("Scaled Intensity")
    ax.set_ylim(0, 1)  # Set y-axis limits
    ax.set_yticks(np.arange(0, 1.1, 0.1))  # Set horizontal gridlines every 0.1 from 0 to 1
    ax.grid(True)  # Show gridlines
    ax.legend()

    # Display the plot
    plt.tight_layout()
    plt.savefig(savepath)
    plt.savefig(savepath.replace('.png', '.pdf'))
    plt.close()

def compute_r_values(
    path, 
    num_candidates, 
    wavesource, 
    min_2_theta,
    max_2_theta,
    N_postsubsample
):
    wavelength = WAVELENGTHS[wavesource]
    ## compute Q values
    Q_min = 4 * np.pi * np.sin(np.radians(min_2_theta/2)) / wavelength
    Q_max = 4 * np.pi * np.sin(np.radians(max_2_theta/2)) / wavelength

    Qs = np.linspace(Q_min, Q_max, N_postsubsample)

    # load predictions
    materials = os.listdir(path)
    materials = [material for material in materials if 'material' in material]

    # sort by material number
    materials = sorted(materials, key=lambda x: int(x.split('_')[0].split('material')[1]))

    r_values = {}

    xrd_viz_base_path = os.path.join(path, 'xrd_gt_vs_pred_sim')
    os.makedirs(xrd_viz_base_path, exist_ok=True)

    all_r_values_flat_list = list()
    best_r_values_by_material = list()

    for i, material_dir in tqdm(enumerate(materials)):
        # create a xrd viz directory for each material
        xrd_viz_material_path = os.path.join(xrd_viz_base_path, material_dir)
        os.makedirs(xrd_viz_material_path, exist_ok=True)

        r_values_current_material = {}

        best_r_value_curr_material = 1e6

        gt_xrd = torch.load(os.path.join(path, material_dir, 
            f'gt/xrd/hiRes_gaussian_0.005_{material_dir}.pt'))
        for j in range(num_candidates):
            # has Gaussian ONLY smoothing
            pred_xrd_path = os.path.join(path, material_dir, 
                        f'pred/candidate{j}/xrd_opt_gen/gaussian_0.005_{material_dir}.pt')
            if not os.path.exists(pred_xrd_path):
                raise ValueError(f'{pred_xrd_path} does not exist')
            pred_xrd = torch.load(pred_xrd_path)

            r_value_current_material_current_cand = calc_r_factor(gt_xrd, pred_xrd, Qs)
            r_values_current_material[f'candidate{j}'] = r_value_current_material_current_cand
            
            all_r_values_flat_list.append(r_value_current_material_current_cand)
            best_r_value_curr_material = min(best_r_value_curr_material, r_value_current_material_current_cand)
            
            # overlay the two XRDs
            xrd_viz_material_candidate_path = os.path.join(xrd_viz_material_path, f'candidate{j}.png')
            plot_overlaid_graphs(gt_xrd, pred_xrd, Qs, xrd_viz_material_candidate_path)
        r_values[material_dir] = r_values_current_material
        
        best_r_values_by_material.append(best_r_value_curr_material)

    # log aggregate stats
    r_values['_avg_r_value'] = float(np.mean(all_r_values_flat_list))
    r_values['_std_r_value'] = float(np.std(all_r_values_flat_list))
    r_values['_avg_best_r_value'] = float(np.mean(best_r_values_by_material))
    r_values['_std_best_r_value'] = float(np.std(best_r_values_by_material))
    # print(r_values)

    save_file = os.path.join(path, 'r_values.json')
    json.dump(r_values, open(save_file, 'w'), indent=4)

    return

if __name__ == "__main__":
    num_candidates = 5
    wavesource = 'CuKa'
    min_2_theta = 0
    max_2_theta = 180
    N_postsubsample = 8192

    for path in [
        '/home/gabeguo/cdvae_xrd/paper_results_PRELIM/_random_baseline_EXPERIMENTAL_',
        '/home/gabeguo/cdvae_xrd/paper_results_PRELIM/experimental_sinc10Filter_',
        '/home/gabeguo/cdvae_xrd/paper_results_PRELIM/experimental_baseline_noOpt',
        '/home/gabeguo/cdvae_xrd/paper_results_PRELIM/_random_baseline_sinc10_',
        '/home/gabeguo/cdvae_xrd/paper_results_PRELIM/_sinc10_replicate',
        '/home/gabeguo/cdvae_xrd/paper_results_PRELIM/_sinc10_baseline_noOpt',
        '/home/gabeguo/cdvae_xrd/paper_results_PRELIM/_random_baseline_sinc100_',
        '/home/gabeguo/cdvae_xrd/paper_results_PRELIM/_sinc100_',
        '/home/gabeguo/cdvae_xrd/paper_results_PRELIM/_sinc100_baseline_noOpt'
    ]:
        compute_r_values(
            path=path, 
            num_candidates=num_candidates, 
            wavesource=wavesource, 
            min_2_theta=min_2_theta,
            max_2_theta=max_2_theta,
            N_postsubsample=N_postsubsample
        )
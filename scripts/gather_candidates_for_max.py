import os
import shutil
import matplotlib.pyplot as plt

def plot_pxrd(xy_2theta_filepath, save_path):
    _2thetas = list()
    intensities = list()
    with open(xy_2theta_filepath, 'r') as fin:
        for line in fin.readlines():
            (curr_2theta, curr_intensity) = (float(x) for x in line.split())
            _2thetas.append(curr_2theta)
            intensities.append(curr_intensity)
    assert len(_2thetas) == len(intensities)
    plt.plot(_2thetas, intensities)
    plt.savefig(save_path)
    plt.close()

    return

def gather_from_dir(results_dir, output_dir, desired_indices, peak_profile):
    os.makedirs(output_dir, exist_ok=True)
    desired_names = [f'material{the_idx}_mp-' for the_idx in desired_indices]
    for material_dir in os.listdir(results_dir):
        # check if this is the material we want
        material_num = material_dir.split('_')[0]
        if not any([x in material_dir for x in desired_names]): # not one of our materials
            continue
        # create folder for curr material
        curr_material_output_dir = os.path.join(output_dir, material_dir)
        os.makedirs(curr_material_output_dir, exist_ok=True)
        # get GT
        gt_cif_filepath = os.path.join(results_dir, material_dir, 'gt', 'cif',
                                       f'noSpacegroup_{material_dir}.cif')
        assert os.path.exists(gt_cif_filepath), gt_cif_filepath
        shutil.copy(gt_cif_filepath, os.path.join(curr_material_output_dir,
                        f'{material_dir}_gt.cif'))
        # get PXRD
        gt_sinc2_filepath = os.path.join(results_dir, material_dir, 'gt', 'xrd',
                                        f'hiRes_{peak_profile}_{material_dir}.xy')
        assert os.path.exists(gt_sinc2_filepath), gt_sinc2_filepath
        shutil.copy(gt_sinc2_filepath, os.path.join(curr_material_output_dir,
                        f'{material_dir}_xrd_2theta.xy'))
        plot_pxrd(gt_sinc2_filepath, os.path.join(curr_material_output_dir,
                        f'{material_dir}_xrd_2theta.pdf'))
        # go through 10 samples
        for i in range(10):
            desired_cif_filepath = os.path.join(results_dir, material_dir,
                                    'pred', f'candidate{i}', 'cif',
                                    f'noSpacegroup_{material_num}_candidate{i}.cif')
            assert os.path.exists(desired_cif_filepath), desired_cif_filepath
            shutil.copy(desired_cif_filepath, os.path.join(curr_material_output_dir, 
                        f'{material_dir}_candidate{i}.cif'))
    return

if __name__ == "__main__":
    desired_indices = [i for i in range(9, 200, 10)]
    assert len(desired_indices) == 20
    gather_from_dir(results_dir='/home/gabeguo/cdvae_xrd/paper_results_PRELIM/_sinc10_', 
                    output_dir='/home/gabeguo/cdvae_xrd/candidates_to_max_09_20_24/_sincSq10_', 
                    desired_indices=desired_indices, peak_profile="both_10_0.001")
    gather_from_dir(results_dir='/home/gabeguo/cdvae_xrd/paper_results_PRELIM/_sinc100_', 
                    output_dir='/home/gabeguo/cdvae_xrd/candidates_to_max_09_20_24/_sincSq100_', 
                    desired_indices=desired_indices, peak_profile="both_100_0.001")
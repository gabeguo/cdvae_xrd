import os
import shutil

def gather_from_dir(results_dir, output_dir, desired_indices):
    os.makedirs(output_dir, exist_ok=True)
    desired_names = [f'material{the_idx}_mp-' for the_idx in desired_indices]
    for material_dir in os.listdir(results_dir):
        material_num = material_dir.split('_')[0]
        if not any([x in material_dir for x in desired_names]): # not one of our materials
            continue
        for i in range(5):
            desired_cif_filepath = os.path.join(results_dir, material_dir,
                                    'pred', f'candidate{i}', 'cif',
                                    f'noSpacegroup_{material_num}_candidate{i}.cif')
            assert os.path.exists(desired_cif_filepath), desired_cif_filepath
            shutil.copy(desired_cif_filepath, os.path.join(output_dir, 
                        f'{material_dir}_candidate{i}.cif'))
    return

if __name__ == "__main__":
    desired_indices = [i for i in range(0, 200, 10)] + [199]
    gather_from_dir(results_dir='/home/gabeguo/cdvae_xrd/paper_results_PRELIM/_sinc10_replicate', 
                    output_dir='/home/gabeguo/cdvae_xrd/candidates_to_max_05_16_24/_sinc10_', 
                    desired_indices=desired_indices)
    gather_from_dir(results_dir='/home/gabeguo/cdvae_xrd/paper_results_PRELIM/_sinc100_', 
                    output_dir='/home/gabeguo/cdvae_xrd/candidates_to_max_05_16_24/_sinc100_', 
                    desired_indices=desired_indices)
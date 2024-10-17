import os
import shutil

def gather_candidates(candidate_folder, indices, output_folder):
    # create output folder
    os.makedirs(output_folder, exist_ok=True)
    # go through all compositions
    for composition_folder in os.listdir(candidate_folder):
        if any([f"material{x}_mp-" in composition_folder for x in indices]):
            material_num = composition_folder.split("_")[0]
            composition = composition_folder.split("_")[-1]
            # go through all predictions
            for candidate_num in range(0, 10):
                material_path = os.path.join(candidate_folder, composition_folder, "pred", f"candidate{candidate_num}", "cif", f"{material_num}_candidate{candidate_num}.cif")
                assert os.path.exists(material_path)
                # copy it
                assert os.path.isdir(output_folder)
                shutil.copy(src=material_path, dst=os.path.join(output_folder, f"{composition}_{candidate_num}.cif"))
    return

def main():
    gather_candidates(candidate_folder="/home/gabeguo/cdvae_xrd/paper_results_PRELIM/_jamie_data_", indices=[x for x in range(0, 5)], output_folder="Mn3Ge_candidates")
    gather_candidates(candidate_folder="/home/gabeguo/cdvae_xrd/paper_results_PRELIM/_jamie_data_", indices=[x for x in range(5, 10)], output_folder="Mn3GeX_candidates")

if __name__ == "__main__":
    main()

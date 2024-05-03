from pymatgen.core.structure import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.io.cif import CifParser
import argparse
import os
import json
import numpy as np
from tqdm import tqdm

MATCH_RATE = "match_rate"
PEARSON_R = "Mean best Pearson's correlation coefficient between PDFs"
COUNT = 'count'

def is_match(the_dict):
    all_match_statuses = the_dict[MATCH_RATE]
    for curr_pred in all_match_statuses:
        if curr_pred[MATCH_RATE] > 0.5:
            return 1.0
    return 0.0

def get_correlation(the_dict):
    return the_dict[PEARSON_R]

def get_crystal_system(cif_filepath):
    structure = Structure.from_file(cif_filepath)
    sga = SpacegroupAnalyzer(structure=structure, symprec=0.01, angle_tolerance=2.5)
    return sga.get_crystal_system()

def update_curr_system_dict(aggregate_results, the_system, the_correlation, 
                            match_status):
    if the_system not in aggregate_results:
        aggregate_results[the_system] = {PEARSON_R: list(),
                                         MATCH_RATE: list()}
    aggregate_results[the_system][PEARSON_R].append(float(the_correlation))
    aggregate_results[the_system][MATCH_RATE].append(float(match_status))

    return

def postprocess(aggregate_results):
    for crystal_system in aggregate_results:
        system_results = aggregate_results[crystal_system]
        system_results[COUNT] = len(system_results[MATCH_RATE])
        assert system_results[COUNT] == len(system_results[PEARSON_R])
        system_results[PEARSON_R] = (round(np.mean(system_results[PEARSON_R]), 4),
                                     round(np.std(system_results[PEARSON_R]), 4))
        system_results[MATCH_RATE] = np.mean(system_results[MATCH_RATE])
    return

def main(args):
    aggregate_results = dict()

    the_result_folders = [folder for folder in os.listdir(args.results_dir) \
                          if 'material' in folder]
    
    for material_folder in tqdm(the_result_folders):
        cif_filepath = os.path.join(args.results_dir, material_folder, 'gt', 'cif', f'{material_folder}.cif')
        assert os.path.exists(cif_filepath), f"{cif_filepath} not valid"
        metrics_file = os.path.join(args.results_dir, material_folder, 'metrics', f"{material_folder.split('_')[0]}.json")
        assert os.path.exists(metrics_file), f"{metrics_file} not valid"
        with open(metrics_file, 'r') as fin:
            curr_metrics_dict = json.load(fin)

        the_system = get_crystal_system(cif_filepath)
        the_correlation = get_correlation(curr_metrics_dict)
        match_status = is_match(curr_metrics_dict)
        update_curr_system_dict(aggregate_results, 
                                the_system=the_system, the_correlation=the_correlation, 
                                match_status=match_status)
    
    postprocess(aggregate_results)
    print(json.dumps(aggregate_results, indent=4))

    os.makedirs(args.output_dir, exist_ok=True)
    experiment_name = args.results_dir.split('/')[-1]
    output_filepath = os.path.join(args.output_dir, f"{experiment_name}.json")
    with open(output_filepath, 'w') as fout:
        json.dump(aggregate_results, fout, indent=4)

    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir', type=str)
    parser.add_argument('--output_dir', type=str, default='results_by_crystal_system')
    args = parser.parse_args()

    print('results for:', args.results_dir)
    main(args)

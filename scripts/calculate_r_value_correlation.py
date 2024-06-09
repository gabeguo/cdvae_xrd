import argparse
import numpy as np
import scipy
import matplotlib.pyplot as plt
import os
import json

def get_refined_candidate_to_rw(input_filepath):
    ret_val = dict()
    with open(input_filepath, 'r') as fin:
        for curr_line in fin.readlines():
            stripped_line = curr_line.strip()
            the_tokens = stripped_line.split()
            if len(the_tokens) == 0:
                continue
            assert len(the_tokens) == 2, f"{the_tokens}"
            curr_candidate_name = the_tokens[0]
            curr_rw = float(the_tokens[1]) / 100
            ret_val[curr_candidate_name] = curr_rw
    return ret_val

# take only best Rw
def filter_candidates(refined_candidates):
    ret_val = dict()
    # group them by material
    grouped_by_material = dict()
    for curr_candidate_name in refined_candidates:
        curr_material_name = curr_candidate_name[:-len('candidate0')]
        if curr_material_name not in grouped_by_material:
            grouped_by_material[curr_material_name] = dict()
        grouped_by_material[curr_material_name][curr_candidate_name] = refined_candidates[curr_candidate_name]
    # take the best candidate only, after refinement
    for the_material in grouped_by_material:
        #print(the_material)
        assert len(grouped_by_material[the_material]) <= 5, grouped_by_material[the_material]
        material_candidates = grouped_by_material[the_material]
        best_candidate = ''
        best_r = 1e6
        for curr_candidate in material_candidates:
            if material_candidates[curr_candidate] < best_r:
                best_r = material_candidates[curr_candidate]
                best_candidate = curr_candidate
        ret_val[best_candidate] = best_r
    #print(ret_val)

    return ret_val

def get_unrefined_candidate_to_rw(input_filepath):
    ret_val = dict()
    with open(input_filepath, 'r') as fin:
        the_results = json.load(fin)
        for the_material in the_results:
            curr_material_results = the_results[the_material]
            if not isinstance(curr_material_results, dict):
                continue
            for the_candidate in curr_material_results:
                full_name = f"{the_material}_{the_candidate}"
                the_rval = curr_material_results[the_candidate]
                ret_val[full_name] = the_rval
    return ret_val

def main(args):
    refined_candidate_to_rw = get_refined_candidate_to_rw(args.refined_txt)
    # refined_candidate_to_rw = filter_candidates(refined_candidate_to_rw)
    unrefined_candidate_to_rw = get_unrefined_candidate_to_rw(args.unrefined_json)
    # unrefined_candidate_to_rw = filter_candidates(unrefined_candidate_to_rw)

    print(f'num refined candidates: {len(refined_candidate_to_rw)}')
    print(f'num unrefined candidates: {len(unrefined_candidate_to_rw)}')
    assert len(refined_candidate_to_rw) <= len(unrefined_candidate_to_rw)
    # assert len(refined_candidate_to_rw) % 5 == 0
    # assert len(unrefined_candidate_to_rw) % 5 == 0

    # have them in the same order, so we can calculate correlation
    unrefined_rws_ordered = list() 
    refined_rws_ordered = list()

    for candidate_name in refined_candidate_to_rw:
        curr_unrefined_rw = unrefined_candidate_to_rw[candidate_name]
        curr_refined_rw = refined_candidate_to_rw[candidate_name]

        unrefined_rws_ordered.append(curr_unrefined_rw)
        refined_rws_ordered.append(curr_refined_rw)
    
    # calculate linear regression
    regression_result = scipy.stats.linregress(x=unrefined_rws_ordered, y=refined_rws_ordered)
    print(f'correlation between refined and unrefined: {regression_result.rvalue}')
    print(f'Refined = {regression_result.slope:.3f} * Raw + {regression_result.intercept:.3f}')

    # plot
    plt.plot(unrefined_rws_ordered, refined_rws_ordered, 'o')
    plt.xlabel('R_w: raw AI generation')
    plt.ylabel('R_w: after PDF refinement')
    plt.plot(unrefined_rws_ordered, regression_result.intercept + regression_result.slope * np.array(unrefined_rws_ordered), 
             'r', label='fitted line')
    plt.plot(np.linspace(0, max(unrefined_rws_ordered)), np.linspace(0, max(unrefined_rws_ordered)),
             'g', marker='', linestyle='--', label='identity line')
    plt.grid()
    plt.xlim(0, 1.6)
    plt.ylim(0, 1.6)
    #plt.xlim(min(unrefined_rws_ordered), max(unrefined_rws_ordered))
    #plt.ylim(min(unrefined_rws_ordered), max(unrefined_rws_ordered))
    plt.legend()
    
    # save regression results
    os.makedirs(args.save_dir, exist_ok=True)
    regression_filepath = os.path.join(args.save_dir, 'regression_results.json')
    with open(regression_filepath, 'w') as fout:
        ret_val = {
            'pearson correlation':regression_result.rvalue,
            'slope':regression_result.slope,
            'intercept':regression_result.intercept
        }
        ret_val.update(vars(args))
        json.dump(ret_val, fout, indent=4)
        print(json.dumps(ret_val, indent=4))
    
    # save figure
    plot_filepath = os.path.join(args.save_dir, 'regression_plot.pdf')
    plt.savefig(plot_filepath)
    plt.close()

    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--unrefined_json', type=str, 
                        help='json file with r values for each of the AI-generated candidates')
    parser.add_argument('--refined_txt', type=str,
                        help='txt file with r values for each of the refined candidates')
    parser.add_argument('--save_dir', type=str,
                        help='directory to save results to')
    args = parser.parse_args()

    main(args)
import argparse
import numpy as np
import scipy
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
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
            curr_candidate_name = int(the_tokens[0])
            curr_rw = float(the_tokens[1]) / 100
            ret_val[curr_candidate_name] = curr_rw
    print('r values refined:', ret_val)
    return ret_val

def get_unrefined_candidate_to_rw(input_filepath, desired_candidates):
    ret_val = dict()
    with open(input_filepath, 'r') as fin:
        the_results = json.load(fin)
        for the_material in the_results:
            if not any([f'material{curr_material_num}_' \
                        in the_material for curr_material_num in desired_candidates]):
                continue
            curr_material_num = int(the_material.split('_')[0][len('material'):])
            # print('curr material num', curr_material_num)
            curr_material_results = the_results[the_material]
            if not isinstance(curr_material_results, dict):
                continue
            found = False
            for the_candidate in curr_material_results:
                if the_candidate != f'candidate{desired_candidates[curr_material_num]}':
                    continue
                found = True
                the_rval = curr_material_results[the_candidate]
                ret_val[curr_material_num] = the_rval
            if not found:
                raise ValueError('could not find candidate for:', the_material)
    assert len(ret_val) == 20, ret_val
    print('r values unrefined:', ret_val)
    return ret_val

def get_desired_candidates(curated_candidates_folder, sinc_level):
    materialNum_to_candidateNum = dict()
    for curr_material in os.listdir(curated_candidates_folder):
        if 'material' not in curr_material:
            continue
        material_num = int(curr_material.split('_')[0][len('material'):])
        # print('material', material_num)
        curr_material_folder = os.path.join(curated_candidates_folder, curr_material)
        found = False
        for the_filename in os.listdir(curr_material_folder):
            if f'pred_sinc{sinc_level}' in the_filename:
                candidate_num = int(the_filename.split('_')[3][len('candidate'):])
                # print('\tcandidate', candidate_num)
                materialNum_to_candidateNum[material_num] = candidate_num
                found = True
                break
        if not found:
            raise ValueError('could not find desired candidate')
    assert len(materialNum_to_candidateNum) == 20, materialNum_to_candidateNum
    return materialNum_to_candidateNum

def main(args):
    refined_candidate_to_rw = get_refined_candidate_to_rw(args.refined_txt)
    desired_candidates = get_desired_candidates(args.curated_candidates_folder, args.sinc)
    unrefined_candidate_to_rw = get_unrefined_candidate_to_rw(args.unrefined_json, desired_candidates)

    print(f'num refined candidates: {len(refined_candidate_to_rw)}')
    print(f'num unrefined candidates: {len(unrefined_candidate_to_rw)}')
    assert len(refined_candidate_to_rw) <= len(unrefined_candidate_to_rw)
    # assert len(refined_candidate_to_rw) % 5 == 0
    # assert len(unrefined_candidate_to_rw) % 5 == 0

    # have them in the same order, so we can calculate correlation
    unrefined_rws_ordered = list() 
    refined_rws_ordered = list()

    num_outliers = 0
    for candidate_name in refined_candidate_to_rw:
        curr_unrefined_rw = unrefined_candidate_to_rw[candidate_name]
        curr_refined_rw = refined_candidate_to_rw[candidate_name]

        if curr_refined_rw > args.thresh_y or curr_unrefined_rw > args.thresh_x:
            num_outliers += 1
            continue

        unrefined_rws_ordered.append(curr_unrefined_rw)
        refined_rws_ordered.append(curr_refined_rw)
    
    print(f"{num_outliers} outliers")

    # calculate linear regression
    regression_result = scipy.stats.linregress(x=unrefined_rws_ordered, y=refined_rws_ordered)
    print(f'correlation between refined and unrefined: {regression_result.rvalue}')
    print(f'Refined = {regression_result.slope:.3f} * Raw + {regression_result.intercept:.3f}')

    # plot
    plt.plot(unrefined_rws_ordered, refined_rws_ordered, 'o')
    plt.xlabel('$R_w$: raw AI generation')
    plt.ylabel('$R_w$: after XRD refinement')
    generic_x_vals = np.linspace(0, max(args.thresh_x, args.thresh_y), 20)
    plt.xlim(0, args.thresh_x)
    plt.ylim(0, args.thresh_y)
    # plt.plot(generic_x_vals, regression_result.intercept + regression_result.slope * generic_x_vals, 
    #          'blue', 
    #          label=f"fitted line: y = {regression_result.slope:.3f} * x + {regression_result.intercept:.3f}" + 
    #             f" (corr = {regression_result.rvalue:.3f})")
    plt.fill_between(generic_x_vals, np.full_like(generic_x_vals, 0.0), np.full_like(generic_x_vals, 0.05), color='darkgreen', alpha=0.2)
    plt.fill_between(generic_x_vals, np.full_like(generic_x_vals, 0.05), np.full_like(generic_x_vals, 0.1), color='palegreen', alpha=0.2)
    plt.fill_between(generic_x_vals, np.full_like(generic_x_vals, 0.1), np.full_like(generic_x_vals, 0.2), color='yellow', alpha=0.2)
    plt.fill_between(generic_x_vals, np.full_like(generic_x_vals, 0.2), np.full_like(generic_x_vals, 0.4), color='orange', alpha=0.2)
    plt.fill_between(generic_x_vals, np.full_like(generic_x_vals, 0.4), np.full_like(generic_x_vals, args.thresh_y), color='red', alpha=0.2)
    plt.plot(generic_x_vals, generic_x_vals,
             'gray', alpha=0.6, marker='', linestyle='--', label='identity line')
    # plt.xlim(0, 1.4)
    # plt.ylim(0, 1.4)
    # plt.title(f'Before and After Refinement: sinc{args.sinc}')
    #plt.xlim(min(unrefined_rws_ordered), max(unrefined_rws_ordered))
    #plt.ylim(min(unrefined_rws_ordered), max(unrefined_rws_ordered))
    # plt.legend()
    # Thanks https://stackoverflow.com/a/58675407
    major_tick_spacing = 0.2
    plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(major_tick_spacing))
    plt.gca().yaxis.set_major_locator(ticker.MultipleLocator(major_tick_spacing))
    minor_tick_spacing = 0.05
    plt.gca().xaxis.set_minor_locator(ticker.MultipleLocator(minor_tick_spacing))
    plt.gca().yaxis.set_minor_locator(ticker.MultipleLocator(minor_tick_spacing))
    plt.grid(which='minor', color='#CCCCCC')
    plt.grid(which='major', color='#777777')

    # save regression results
    os.makedirs(args.save_dir, exist_ok=True)
    regression_filepath = os.path.join(args.save_dir, 'regression_results.json')
    with open(regression_filepath, 'w') as fout:
        ret_val = {
            'pearson correlation':regression_result.rvalue,
            'slope':regression_result.slope,
            'intercept':regression_result.intercept,
            'num outliers':num_outliers,
            'unrefined mean r_value': np.mean(unrefined_rws_ordered),
            'unrefined std r_value': np.std(unrefined_rws_ordered),
            'refined mean r_value': np.mean(refined_rws_ordered),
            'refined std r_value': np.std(refined_rws_ordered)
        }
        ret_val.update(vars(args))
        json.dump(ret_val, fout, indent=4)
        print(json.dumps(ret_val, indent=4))
    
    # save figure
    plt.tight_layout()
    plot_filepath = os.path.join(args.save_dir, 'regression_plot.pdf')
    plt.savefig(plot_filepath)
    plt.savefig(plot_filepath.replace('.pdf', '.png'), dpi=300)
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
    parser.add_argument('--curated_candidates_folder', type=str,
                        help='folder directory of the data we sent to Max')
    parser.add_argument('--sinc', type=int,
                        help='desired sinc level')
    parser.add_argument('--thresh_x', type=float, default=1.4,
                        help='threshold for outlier R_w values before refinement')
    parser.add_argument('--thresh_y', type=float, default=1.0,
                        help='threshold for outlier R_w values after refinement')
    args = parser.parse_args()

    main(args)
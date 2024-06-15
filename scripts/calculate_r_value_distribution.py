import argparse
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import matplotlib.ticker as ticker

def main(args):
    # calculate values
    with open(args.r_values_json, 'r') as fin:
        r_vals_by_material = json.load(fin)
        top_r_values = list()
        for material in r_vals_by_material:
            if isinstance(r_vals_by_material[material], dict):
                # lower r value is better
                if args.all_candidates:
                    top_r_values.extend(r_vals_by_material[material].values())
                else:
                    top_r_values.append(min(r_vals_by_material[material].values()))
    plt.ecdf(top_r_values)
    plt.xlabel('Prediction Error (R-Value)')
    plt.ylabel("Percentage of Materials At or Below Prediction Error")

    # Thanks https://stackoverflow.com/a/58675407
    major_tick_spacing = 0.2
    plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(major_tick_spacing))
    plt.gca().yaxis.set_major_locator(ticker.MultipleLocator(major_tick_spacing))
    minor_tick_spacing = 0.05
    plt.gca().xaxis.set_minor_locator(ticker.MultipleLocator(minor_tick_spacing))
    plt.gca().yaxis.set_minor_locator(ticker.MultipleLocator(minor_tick_spacing))
    plt.yticks(ticks=plt.yticks()[0], 
               labels=[f"{int(y * 100)}%" for y in plt.yticks()[0]])

    # set axis limits
    plt.xlim(0, max(top_r_values))

    # have error thresholds
    generic_y_vals = np.linspace(0, 1, 11)
    plt.fill_betweenx(generic_y_vals, np.full_like(generic_y_vals, 0.0), np.full_like(generic_y_vals, 0.05), color='darkgreen', alpha=0.2)
    plt.fill_betweenx(generic_y_vals, np.full_like(generic_y_vals, 0.05), np.full_like(generic_y_vals, 0.1), color='palegreen', alpha=0.2)
    plt.fill_betweenx(generic_y_vals, np.full_like(generic_y_vals, 0.1), np.full_like(generic_y_vals, 0.2), color='yellow', alpha=0.2)
    plt.fill_betweenx(generic_y_vals, np.full_like(generic_y_vals, 0.2), np.full_like(generic_y_vals, 0.4), color='orange', alpha=0.2)
    plt.fill_betweenx(generic_y_vals, np.full_like(generic_y_vals, 0.4), np.full_like(generic_y_vals, max(top_r_values)), color='red', alpha=0.2)

    # plot
    os.makedirs(args.output_dir, exist_ok=True)
    output_filepath = os.path.join(args.output_dir, 'r_value_cdf.pdf')
    plt.grid(which='minor', color='#CCCCCC')
    plt.grid(which='major', color='#777777')
    plt.tight_layout()
    plt.savefig(output_filepath)

    # sanity check statistics
    print('average r:', np.mean(top_r_values))
    print('std r:', np.std(top_r_values))
                
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--r_values_json', type=str)
    parser.add_argument('--all_candidates', action='store_true')
    parser.add_argument('--output_dir', type=str, default='')

    args = parser.parse_args()

    main(args)
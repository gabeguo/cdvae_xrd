import argparse
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import matplotlib.ticker as ticker

def main(args):
    all_r_values = [args.r_values_random_json,
                    args.r_values_latentSearch_json, 
                    args.r_values_pxrdnet_json]
    labels = ['Random', 'Latent Search', 'PXRDNet']
    line_colors = ['red', 'blue', 'green']
    # line_styles = ['solid', 'dashed', 'dotted']
    hatches = ['.', 'O', '*']
    max_x_val = -1
    # calculate values
    for r_values_json, label, line_color, hatch in \
            zip(all_r_values, labels, line_colors, hatches):
        with open(r_values_json, 'r') as fin:
            r_vals_by_material = json.load(fin)
            top_r_values = list()
            for material in r_vals_by_material:
                if isinstance(r_vals_by_material[material], dict):
                    # lower r value is better
                    if args.all_candidates:
                        top_r_values.extend(r_vals_by_material[material].values())
                    else:
                        top_r_values.append(min(r_vals_by_material[material].values()))
        bins = np.linspace(0, 2, 21)
        plt.hist(top_r_values, bins=bins, histtype='step', 
                 color=line_color, label=label, linewidth=2, hatch=hatch, alpha=0.8)
        print('r range:', min(top_r_values), max(top_r_values))
        #plt.ecdf(top_r_values)
        plt.xlabel('Prediction Error ($R_{wp}^{2}$)')
        plt.ylabel("Number of Materials")
        if args.disable_y_label:
            plt.ylabel("   ")

        # Thanks https://stackoverflow.com/a/58675407
        plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(0.4))
        plt.gca().yaxis.set_major_locator(ticker.MultipleLocator(8))
        plt.gca().xaxis.set_minor_locator(ticker.MultipleLocator(0.1))
        plt.gca().yaxis.set_minor_locator(ticker.MultipleLocator(2))
        # plt.yticks(ticks=plt.yticks()[0], 
        #            labels=[f"{int(y * 100)}%" for y in plt.yticks()[0]])
        max_x_val = max(max_x_val, max(top_r_values))

        # sanity check statistics
        print('average r:', np.mean(top_r_values))
        print('std r:', np.std(top_r_values))
    
    max_x_val = int((max_x_val / 0.2) + 1) * 0.2
    print(max_x_val)
    plt.xlim(0, max_x_val)

    plt.ylim(0, 40)

    # plot
    os.makedirs(args.output_dir, exist_ok=True)
    output_filepath = os.path.join(args.output_dir, 'r_value_histogram.pdf')
    plt.grid(which='minor', color='#CCCCCC')
    plt.grid(which='major', color='#777777')
    plt.tight_layout()
    plt.savefig(output_filepath)
                
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--r_values_pxrdnet_json', type=str)
    parser.add_argument('--r_values_latentSearch_json', type=str)
    parser.add_argument('--r_values_random_json', type=str)
    parser.add_argument('--disable_y_label', action='store_true')
    parser.add_argument('--all_candidates', action='store_true')
    parser.add_argument('--output_dir', type=str, default='')

    args = parser.parse_args()

    main(args)
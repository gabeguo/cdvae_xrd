import matplotlib.pyplot as plt
import os
import numpy as np

def standardize_pattern(x):
    return (np.array(x) - np.min(x)) / (np.max(x) - np.min(x))

def convert_to_q(x, wavelength=0.7294437):
    x = np.array(x)
    peak_locations_theta_deg = x / 2
    peak_locations_theta_rad = np.radians(peak_locations_theta_deg)
    peak_locations_Q = 4 * np.pi * np.sin(peak_locations_theta_rad) / wavelength

    return peak_locations_Q

def get_2theta_intensity(data_path):
    _2thetas = list()
    _intensities_a = list()
    _intensities_b = list()
    with open(data_path, 'r') as fin:
        the_lines = [x.strip() for x in fin.readlines()]
        first_line_idx = the_lines.index('*/') + 1
        the_lines = the_lines[first_line_idx:]
        for curr_line in the_lines:
            curr_2theta, curr_intensity_a, curr_intensity_b = curr_line.split()
            _2thetas.append(float(curr_2theta))
            _intensities_a.append(float(curr_intensity_a))
            _intensities_b.append(float(curr_intensity_b))
    _intensities_a = standardize_pattern(_intensities_a)
    _intensities_b = standardize_pattern(_intensities_b)

    # plot 2-theta
    plt.xlabel(r'2\theta')
    plt.plot(_2thetas, _intensities_a, alpha=0.7, label='background removed')
    plt.plot(_2thetas, _intensities_b, alpha=0.7, label='raw')
    plt.legend()
    save_dir = os.path.dirname(data_path)
    img_filename = os.path.basename(data_path)[:-4] + ".png"
    plt.savefig(os.path.join(save_dir, img_filename))
    plt.close()

    q_values = convert_to_q(_2thetas)

    # plot Q
    plt.xlabel("Q")
    plt.plot(q_values, _intensities_a)
    plt.savefig(os.path.join(save_dir, f"q_{img_filename}"))
    plt.close()

    return q_values, _intensities_a

if __name__ == "__main__":
    get_2theta_intensity('/home/gabeguo/cdvae_xrd/real_data/BL21Robot_0321-2023-09-02-1241_scan2_Mn3Ge.xye')
    get_2theta_intensity('/home/gabeguo/cdvae_xrd/real_data/BL21Robot_0327-2023-09-02-1312_scan2_Mn3GeN.xye')
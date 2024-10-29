# Thanks ChatGPT!

import numpy as np
import matplotlib.pyplot as plt
import os

output_dir = 'experimental_pxrd_comparison'
os.makedirs(output_dir, exist_ok=True)
# Load the data from the .xy file
for filepath in [
        "/home/gabeguo/cdvae_xrd/process_real_xrds/all_fits_experimental/1/fits/br1322Isup2.rtv.combined.cif_AlPO4_2_fit.xy", 
        "/home/gabeguo/cdvae_xrd/process_real_xrds/all_fits_experimental/2/fits/br1340Isup2.rtv.combined.cif_CdBiClO2_1_fit.xy",
        "/home/gabeguo/cdvae_xrd/process_real_xrds/all_fits_experimental/6/fits/ks5409BTsup2.rtv.combined.cif_BaTiO3_4_fit.xy",
        "/home/gabeguo/cdvae_xrd/process_real_xrds/all_fits_experimental/7/fits/sh0123Xraysup5.rtv.combined.cif_Sr2YbNbO6_6_fit.xy",
        "/home/gabeguo/cdvae_xrd/process_real_xrds/all_fits_experimental/8/fits/sq1033Isup2.rtv.combined.cif_LaInO3_3_fit.xy"
]:
    data = np.loadtxt(filepath)

    # Assign each column to the corresponding variable
    two_theta = data[:, 0]
    observed_intensity = data[:, 1]
    predicted_intensity = data[:, 2]
    error = data[:, 3]

    # Plotting the data
    plt.figure(figsize=(10, 6))

    # Plot observed intensity with error bars
    plt.plot(two_theta, observed_intensity, color='blue', alpha=0.6, label='Experimentally Observed PXRD', linewidth=1.5)

    # Plot predicted intensity
    plt.plot(two_theta, predicted_intensity, color='red', alpha=0.6, label='Predicted Candidate (AI + Rietveld) PXRD', linewidth=1.5)

    # Adding labels and title
    plt.xlabel(r'2$\theta$ (degrees)')
    plt.ylabel('Intensity')
    # plt.title('Rietveld Refinement Results')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()

    filename = os.path.basename(filepath)
    for filetype in ["pdf", "png"]:
        output_filepath = os.path.join(output_dir, f'pxrd_fit_{filename}.{filetype}')
        # Display the plot
        plt.savefig(output_filepath, dpi=400)
    plt.close()

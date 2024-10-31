import numpy as np
import matplotlib.pyplot as plt

# Load the data from the .xy file
filename = '/home/gabeguo/cdvae_xrd/paper_results_PRELIM/_sinc10_/material29_mp-1205608_SmMn2SiC/gt/xrd/hiRes_both_10_0.001_material29_mp-1205608_SmMn2SiC.xy'
data = np.loadtxt(filename)

# Assign each column to the corresponding variable
two_theta = data[:, 0]
intensity = data[:, 1]

# Plotting the data
plt.figure(figsize=(10, 6))

# Plot intensity
plt.plot(two_theta, intensity, linewidth=4)
plt.tight_layout()
plt.grid(True)

# Display the plot
plt.savefig('sample_pxrd.png', dpi=400)

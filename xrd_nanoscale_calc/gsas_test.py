import os,sys
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0,'/home/gabeguo/anaconda3/envs/GSASII/GSAS-II/GSASII')
import GSASIIscriptable as G2sc
gpx = G2sc.G2Project(newgpx='dummy.gpx') # create a project
phase0 = gpx.add_phase("/home/gabeguo/cdvae_xrd/paper_results_PRELIM/_sinc10_/material10_mp-1104269_Er(Al2Cu)4/gt/cif/material10_mp-1104269_Er(Al2Cu)4.cif",
         phasename="dummy",fmthint='CIF') # add a phase to the project
# add a simulated histogram and link it to the previous phase(s)
hist1 = gpx.add_simulated_powder_histogram(
    histname="dummy simulation",
    iparams='INST_XRY.prm',
    Tmin=0.,
    Tmax=180.,
    Npoints=10000,
    phases=gpx.phases(),
    scale=None)
gpx.do_refinements()   # calculate pattern
gpx.save()

# # save results
# gpx.histogram(0).Export('PbSO4data','.csv','hist') # data
# gpx.histogram(0).Export('PbSO4refl','.csv','refl') # reflections

# Get the calculated pattern
assert len(gpx.histograms()) == 1
pattern = gpx.histograms()[0]['data'][1]

print(len(pattern))

two_theta = pattern[0]
intensity = pattern[1]

peak_locations_theta_deg = two_theta / 2
peak_locations_theta_rad = np.radians(peak_locations_theta_deg)
peak_locations_Q = 4 * np.pi * np.sin(peak_locations_theta_rad) / 1.5406

plt.plot(peak_locations_Q, intensity)

plt.savefig('dummy.png')
plt.close()
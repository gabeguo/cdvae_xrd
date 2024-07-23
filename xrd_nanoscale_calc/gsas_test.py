import os,sys
import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy.signal import convolve
from pymatgen.io.cif import CifParser
sys.path.insert(0,'/home/gabeguo/anaconda3/envs/GSASII/GSAS-II/GSASII')
import GSASIIscriptable as G2sc
import GSASIIElem as G2e

def mirror(arr):
    """
    Returns [-arr[-1], -arr[-2], ... , -arr[0], +arr[0], ... , +arr[-2], +arr[-1]]
    Retval is an odd function
    """
    ret_val = np.array([-x for x in reversed(arr)] + [+x for x in arr])
    assert len(ret_val) == 2 * len(arr)
    assert ret_val[0] == -ret_val[-1]
    return ret_val

def get_unit_cell_info(cif, atom_names, form_factors, 
                       ref_pos_Q, desired_Q):
    """
    Returns N, <f>^2, <f^2>
    """
    parser = CifParser(cif)
    structure = parser.get_structures()[0]

    N = len(structure.species)

    atomic_numbers = list()

    resampled_form_factors = {
        atom_names[i]: np.interp(desired_Q, xp=ref_pos_Q, fp=form_factors[:,i])
        for i in range(len(atom_names))
    }

    # for key in resampled_form_factors:
    #     print(resampled_form_factors[key][::20])
    #     plt.plot(desired_Q, resampled_form_factors[key], label=key)
    # plt.legend()
    # plt.savefig('dummy_ff.png')

    count_by_elem = dict()
    for site in structure.sites:
        element = site.species.elements
        assert len(element) == 1
        element = element[0]
        atomic_numbers.append(element.Z)
        elem_symbol = str(element)
        if elem_symbol not in count_by_elem:
            count_by_elem[elem_symbol] = 0
        count_by_elem[elem_symbol] += 1
    
    # Calculate the scattering
    f_sq_then_avg = np.zeros_like(desired_Q)
    f_avg_then_sq = np.zeros_like(desired_Q)
    for curr_elem in count_by_elem:
        curr_count = count_by_elem[curr_elem]
        f_sq_then_avg += curr_count / N * resampled_form_factors[curr_elem] ** 2

        for other_elem in count_by_elem:
            other_count = count_by_elem[other_elem]
            weight = (curr_count * other_count) / (N**2)
            f_avg_then_sq += weight * resampled_form_factors[curr_elem] * resampled_form_factors[other_elem]

    return N, f_avg_then_sq, f_sq_then_avg

def sinc_convolve(Q, intensity, reflections, cif, nano_size):

    atom_names = reflections['FF']['El']
    form_factors = reflections['FF']['FF']
    ref_pos_Q = twoTheta_to_Q(reflections['RefList'][:,5])

    N, avg_f_then_sq, sq_f_then_avg = get_unit_cell_info(cif=cif, 
        atom_names=atom_names, form_factors=form_factors, ref_pos_Q=ref_pos_Q, desired_Q=Q)
    
    # TODO: unsure if this is right?
    avg_f_then_sq = mirror(avg_f_then_sq)
    sq_f_then_avg = mirror(sq_f_then_avg)

    print(f"N = {N}")
    print(f"<f>^2 = {avg_f_then_sq}")
    print(f"<f^2> = {sq_f_then_avg}")

    print(f'len Q, I before: {len(Q)} {len(intensity)}')

    assert Q[-1] > Q[0]
    assert np.isclose(Q[1] - Q[0], Q[-1]-Q[-2], atol=1e-8)
    assert min(Q) >= 0
    assert abs(min(intensity)) <= 1e-2 * abs(max(intensity))
    Q = mirror(Q)
    intensity = mirror(intensity)

    print(f'len Q, I after: {len(Q)} {len(intensity)}')

    # TODO: check that we have right sinc (radians, scale)
    # scale by pi, as per: https://numpy.org/doc/stable/reference/generated/numpy.sinc.html
    sinc = np.sinc((Q / np.pi) * (nano_size / 2))
    
    # convolution = convolve(intensity, sinc, mode='same')
    # convolution = convolution[len(convolution) // 2:]
    # return convolution

    left_conv_term = (intensity / (N * avg_f_then_sq) - sq_f_then_avg / avg_f_then_sq)
    left_conv_term *= 1 / (2 * np.pi)
    right_conv_term = nano_size * sinc
    non_conv_term = sq_f_then_avg / avg_f_then_sq

    unscaled_term = convolve(left_conv_term, right_conv_term, mode='same') + non_conv_term
    result = N * avg_f_then_sq * unscaled_term
    
    return result[len(result)//2:]

def twoTheta_to_Q(twoTheta_deg):
    peak_locations_theta_deg = twoTheta_deg / 2
    peak_locations_theta_rad = np.radians(peak_locations_theta_deg)
    peak_locations_Q = 4 * np.pi * np.sin(peak_locations_theta_rad) / 1.5406
    return peak_locations_Q

def main():
    print(G2e.GetFormFactorCoeff('H'))

    cif_file = "/home/gabeguo/cdvae_xrd/paper_results_PRELIM/_sinc10_/material10_mp-1104269_Er(Al2Cu)4/gt/cif/material10_mp-1104269_Er(Al2Cu)4.cif"

    gpx = G2sc.G2Project(newgpx='dummy.gpx') # create a project
    phase_name = 'dummy'
    phase0 = gpx.add_phase(cif_file,
            phasename=phase_name,fmthint='CIF') # add a phase to the project
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

    reflections = hist1.reflections()[phase_name]
    peak_locations_Q = twoTheta_to_Q(two_theta)

    N_SAMPLES = 10000
    even_spaced_Q = np.linspace(peak_locations_Q[0], peak_locations_Q[-1], num=N_SAMPLES)
    intensity_gridded = np.interp(x=even_spaced_Q, xp=peak_locations_Q, fp=intensity)

    plt.plot(peak_locations_Q, intensity / max(intensity), label='original')
    plt.plot(even_spaced_Q, intensity_gridded / max(intensity_gridded), label='evenly spaced')

    nano_intensity = sinc_convolve(Q=even_spaced_Q, intensity=intensity_gridded, reflections=reflections,
                                   cif=cif_file, nano_size=100)
    
    nano_intensity[even_spaced_Q > 7.5] = np.min(nano_intensity[even_spaced_Q < 7.5])
    nano_intensity = (nano_intensity - np.min(nano_intensity))
    nano_intensity /= np.max(nano_intensity)

    plt.plot(even_spaced_Q, nano_intensity, label='nano')

    plt.legend()

    print('moo')

    plt.savefig('dummy.png')
    plt.close()

if __name__ == "__main__":
    main()
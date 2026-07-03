from astropy.io import ascii
import numpy as np
import sys
import os

def Upsilon_r(Ups_bulge, xi, Sigma_NSC, Sigma_B):
    numerator = Ups_bulge * (Sigma_B + xi * Sigma_NSC)
    denominator = Sigma_B + Sigma_NSC
    upsilon_r = numerator / denominator
    return upsilon_r


# Get the galaxy name and analysis root from environment variables
# These should be set by the calling script (runalls, runallt, etc.)
galaxy_name = os.environ.get('SCO_GALAXY_NAME', None)
analysis_root = os.environ.get('SCO_ANALYSIS_ROOT', None)

if not galaxy_name or not analysis_root:
    print("Error: SCO_GALAXY_NAME and SCO_ANALYSIS_ROOT environment variables must be set.")
    print("       These should be set by the calling script before running this program.")
    sys.exit(1)

sbpath = os.path.join(analysis_root, galaxy_name, 'sb_analysis', 'deprojection') + os.sep
comp = os.path.join(sbpath, 'deprojection.outputs', 'npdyn_components.out')

if not os.path.exists(comp):
    print(f"npdyn_components.out not found at {comp}")
    print("This file is required for variable M/L calculations.")
    print("Please ensure the surface brightness analysis has been completed.")
    sys.exit(1)

sb = ascii.read(comp, names=['r', 'r_pc', 'Sigma_NSC', 'Sigma_bulge'])

bindemor = ascii.read('./bindemo_r.out', data_start = 5, 
                     names = ['ir', 'irc', 'bin_lower_edge', 
                              'bin_center', 'bin_upper_edge', 
                              'bin_lower_arcsec', 'bin_upper_arcsec'])
bindemov = ascii.read('./bindemo_v.out', data_start = 4, 
                     names = ['iv', 'ivc', 'bin_lower_edge', 
                              'bin_center', 'bin_upper_edge'])

r = sb['r'] # in arcsec

Sigma_nsc = sb['Sigma_NSC']
Sigma_bulge = sb['Sigma_bulge']

try:
    Ups_bulge_value = float(sys.argv[1])
except ValueError:
    print("Error: Ups_bulge_value must be a numeric value.")
    sys.exit(1)
try:
    xi = float(sys.argv[2])
except ValueError:
    print("Error: xi must be a numeric value.")
    sys.exit(1)

Upsilon = Upsilon_r(Ups_bulge_value, xi, Sigma_nsc, Sigma_bulge)

# Interpolate data
r_bin_mid = (bindemor['bin_lower_arcsec'] + bindemor['bin_upper_arcsec']) / 2

sorted_indices = np.argsort(r.value)
r_sorted = r.value[sorted_indices]
Upsilon_sorted = Upsilon[sorted_indices]

Upsilon_interp = np.interp(r_bin_mid, r_sorted, Upsilon_sorted)

# Write output
with open('ratML.dat', 'w') as f:
    for i, r_idx in enumerate(bindemor['ir']):
        for j, theta_idx in enumerate(bindemov['iv']):
            f.write(f'{r_idx:12d} {theta_idx:12d} {Upsilon_interp[i]:15.4f}\n')

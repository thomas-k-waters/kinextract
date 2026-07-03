import sys
import numpy as np
from scipy.ndimage import gaussian_filter1d

if len(sys.argv) < 3:
    print("Usage: python fix_losvd_errors.py infile.txt outfile.txt")
    sys.exit(1)

input_file = sys.argv[1]
output_file = sys.argv[2]

# Load data, assuming: vel, losvd, lower_err, upper_err
data = np.loadtxt(input_file)
vel, losvd, lower_err, upper_err = data.T

# Parameters
LOSVD_clip = 0.0007   # Below this, LOSVD is considered "zero"
min_error = 0.005     # Minimum error value to prevent pinching
smoothing_sigma = 3   # Smoothing parameter for Gaussian filter

# 1. Correct lower_err so it satisfies (losvd - lower_err_corr) >= min_error
lower_err_corr = np.where((losvd - lower_err) < min_error, losvd - min_error, lower_err)
lower_err_corr = np.clip(lower_err_corr, 0, losvd)  # Make sure lower_err isn't negative or above losvd

# Set to zero where LOSVD is below the clip threshold
lower_err_corr[losvd < LOSVD_clip] = 0

# 2. Correct upper_err so it satisfies (upper_err_corr - losvd) >= min_error
upper_err_corr = upper_err.copy()
upper_err_corr = np.maximum(upper_err_corr, losvd + min_error)  # Enforce minimum error

# 3. Smooth the entire LOSVD for both lower_err and upper_err
lower_err_corr = gaussian_filter1d(lower_err_corr, sigma=smoothing_sigma)
upper_err_corr = gaussian_filter1d(upper_err_corr, sigma=smoothing_sigma)

# 4. Zero the lower error bars outward of their minimum in each wing
center_idx = np.argmax(losvd)

lower_err_final = lower_err_corr.copy()

# Right wing (high velocities)
right_errs = lower_err_final[center_idx:]
min_idx_right = np.argmin(right_errs)
right_min_global = center_idx + min_idx_right
lower_err_final[right_min_global:] = 0

# Left wing (low velocities)
left_errs = lower_err_final[:center_idx+1][::-1]
min_idx_left = np.argmin(left_errs)
left_min_global = center_idx - min_idx_left
lower_err_final[:left_min_global+1] = 0

# Write output
np.savetxt(output_file, np.column_stack([vel, losvd, lower_err_final, upper_err_corr]),
           fmt='%.6f %.8f %.8f %.8f')

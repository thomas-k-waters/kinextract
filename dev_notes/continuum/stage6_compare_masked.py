"""VALIDATION_CAMPAIGN.md Stage 6: compare the original (unmasked)
100-bin real-data fit against the sky-residual-masked re-fit, to see
whether masking the 4 recurring bad-pixel windows found in Stage 4
measurably changes the previously-documented real-data discrepancies
(sigma radial trend, V offset, h4 offset) -- and in which direction.
"""
import numpy as np

orig = np.load("real_data_all_bins.npz", allow_pickle=True)["results"].item()
masked = np.load("real_data_all_bins_masked.npz", allow_pickle=True)["results"].item()

common = sorted(set(orig) & set(masked))
print(f"{len(common)} bins in common\n")


def summarize(name, orig_vals, masked_vals, legacy_vals, legerr_vals):
    d_orig = np.array(orig_vals) - np.array(legacy_vals)
    d_masked = np.array(masked_vals) - np.array(legacy_vals)
    print(f"{name:8s}  ORIG   mean(diff)={np.mean(d_orig):+7.2f}  RMS={np.sqrt(np.mean(d_orig**2)):6.2f}  "
          f"within legacy err: {100*np.mean(np.abs(d_orig) < legerr_vals):5.1f}%")
    print(f"{name:8s}  MASKED mean(diff)={np.mean(d_masked):+7.2f}  RMS={np.sqrt(np.mean(d_masked**2)):6.2f}  "
          f"within legacy err: {100*np.mean(np.abs(d_masked) < legerr_vals):5.1f}%")
    print()


V_orig = [orig[k]["vherm"] for k in common]
V_masked = [masked[k]["vherm"] for k in common]
V_leg = [orig[k]["V_legacy"] for k in common]
Verr_leg = np.array([orig[k]["Verr_legacy"] for k in common])
summarize("V", V_orig, V_masked, V_leg, Verr_leg)

sig_orig = [orig[k]["sherm"] for k in common]
sig_masked = [masked[k]["sherm"] for k in common]
sig_leg = [orig[k]["sig_legacy"] for k in common]
sigerr_leg = np.array([orig[k]["sigerr_legacy"] for k in common])
summarize("sigma", sig_orig, sig_masked, sig_leg, sigerr_leg)

h4_orig = [orig[k]["h4"] for k in common]
h4_masked = [masked[k]["h4"] for k in common]
h4_leg = [orig[k]["h4_legacy"] for k in common]
h4err_leg = np.array([orig[k]["h4err_legacy"] for k in common])
summarize("h4", h4_orig, h4_masked, h4_leg, h4err_leg)

# radial trend check specifically for sigma (the headline finding this was meant to test)
radii = np.array([orig[k]["radius"] for k in common])
sig_diff_orig = np.array(sig_orig) - np.array(sig_leg)
sig_diff_masked = np.array(sig_masked) - np.array(sig_leg)
absr = np.abs(radii)
inner, outer = absr < 2, absr > 8
print("sigma radial trend check:")
print(f"  ORIG   inner (|r|<2\"): {sig_diff_orig[inner].mean():+.2f}   outer (|r|>8\"): {sig_diff_orig[outer].mean():+.2f}")
print(f"  MASKED inner (|r|<2\"): {sig_diff_masked[inner].mean():+.2f}   outer (|r|>8\"): {sig_diff_masked[outer].mean():+.2f}")

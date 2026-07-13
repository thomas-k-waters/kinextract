"""Check whether the GH-moment fit (fit_losvd_gauss_hermite) adds V bias
beyond what's already in the raw histogram's own first moment (_losvd_moments)."""
import sys, time
sys.path.insert(0, "dev_notes/stress_test")
from harness_emiles import fit_one_emiles

EMILES_TLIST = "dev_notes/stress_test/emiles_templates/Tlist"
EMILES_DIR = "dev_notes/stress_test/emiles_templates"

if __name__ == "__main__":
    sigmas = [30, 70, 160, 250, 350]
    seeds = [42, 1]
    true_v = 80.0
    print(f"{'sigma':>6s} {'seed':>5s} {'gh_v':>8s} {'raw_v':>8s} {'gh_sigma':>9s} {'raw_sigma':>10s}")
    for sigma in sigmas:
        vrange = max(300.0, 3.5 * sigma)
        for seed in seeds:
            r = fit_one_emiles(true_v, sigma, seed, EMILES_TLIST, EMILES_DIR, vrange, sigma)
            print(f"{sigma:6.0f} {seed:5d} {r['gh_v']:8.2f} {r['raw_v']:8.2f} {r['gh_sigma']:9.2f} {r['raw_sigma']:10.2f}")

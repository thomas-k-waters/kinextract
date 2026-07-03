import numpy as np
import json
import os
import math

#def get_user_input(prompt, default=None):
#    if default:
#        prompt = f"{prompt} [{default}]: "
#    else:
#        prompt = f"{prompt}: "
#    user_input = input(prompt)
#    return user_input if user_input else default

def get_user_input(prompt, default=None):
    """Prompt with optional default; show and preserve falsy defaults like 0 or 0.0."""
    if default is not None:
        prompt = f"{prompt} [{default}]: "
    else:
        prompt = f"{prompt}: "
    user_input = input(prompt)
    return user_input if user_input else default

def custom_format(value):
    return f'{value:.2e}'.replace('+', '')

def grid_from_step(vmin, vmax, step, include_end=True):
    """Build a grid from vmin to vmax with fixed step.

    - Starts at vmin and increments by step until <= vmax (with small epsilon).
    - If include_end and vmax is not exactly hit, append vmax.
    - Protects against floating artifacts by rounding based on step magnitude.
    """
    if not np.isfinite(vmin) or not np.isfinite(vmax) or not np.isfinite(step) or step <= 0:
        return np.array([float(vmin)])
    if vmax < vmin:
        vmin, vmax = vmax, vmin
    span = vmax - vmin
    try:
        decimals = max(0, int(-math.floor(math.log10(abs(step)))))
        if decimals < 2:
            decimals = 2
    except Exception:
        decimals = 8
    eps = abs(step) * 1e-9
    n = int(math.floor((span + eps) / step)) + 1
    vals = vmin + step * np.arange(n, dtype=float)
    if include_end and (vmax - vals[-1]) > eps:
        vals = np.append(vals, vmax)
    vals = np.round(vals, decimals)
    if vals.size:
        vals[0] = float(vmin)
        vals[-1] = float(vmax) if include_end else float(vals[-1])
    return vals.astype(float), decimals

def main():
    defaults_file = '.modelgridnodmincdefaults.json'
    defaults = {}

    if os.path.exists(defaults_file):
        with open(defaults_file, 'r') as f:
            defaults = json.load(f)
    
    min_mbh = float(get_user_input("Enter the minimum value for M_BH", defaults.get('min_mbh', '1.0e04')))
    max_mbh = float(get_user_input("Enter the maximum value for M_BH", defaults.get('max_mbh', '5.0e06')))
    mbh_step = float(get_user_input("Enter the stepsize of M_BH values", defaults.get('step_mbh', '1.0e05')))
    include_zero_mass_bh = get_user_input("Do you want to include the 0 mass black hole model? (Y/N)")

    min_ml = float(get_user_input("Enter the minimum value for M/L", defaults.get('min_ml', 0.25)))
    max_ml = float(get_user_input("Enter the maximum value for M/L", defaults.get('max_ml', 1.5)))
    ml_step = float(get_user_input("Enter the stepsize of M/L values", defaults.get('step_ml', 0.25)))

    parameterize_upsilon_ratio = get_user_input("Do you want to parametrize the NSC-Bulge M/L ratio? (Y/N)")
    if parameterize_upsilon_ratio == 'Y' or parameterize_upsilon_ratio == 'y':
        min_ml_ratio = float(get_user_input("Enter the minimum value for the M/L ratio", defaults.get('min_ml_ratio', 0.5)))
        max_ml_ratio = float(get_user_input("Enter the maximum value for the M/L ratio", defaults.get('max_ml_ratio', 2.0)))
        ml_ratio_step = float(get_user_input("Enter the stepsize of the M/L ratio", defaults.get('step_ml_ratio', 0.25)))
    else:
        min_ml_ratio = max_ml_ratio = float(get_user_input("Enter the fixed value for the M/L ratio", defaults.get('min_ml_ratio', 1.0)))
        ml_ratio_step = None

    include_dm_halo = get_user_input("Do you want to include dark matter in the models? (Y/N)")
    if include_dm_halo == 'Y' or include_dm_halo == 'y':
        min_vel = float(get_user_input("Enter the minimum value for v_c (km/s)", defaults.get('min_vel', 0.0)))
        max_vel = float(get_user_input("Enter the maximum value for v_c (km/s)", defaults.get('max_vel', 0.0)))
        vel_step = float(get_user_input("Enter the stepsize of v_c values (km/s)", defaults.get('step_vel', 100.0)))

        min_rc = float(get_user_input("Enter the minimum value for r_c (pc)", defaults.get('min_rc', 1.0)))
        max_rc = float(get_user_input("Enter the maximum value for r_c (pc)", defaults.get('max_rc', 20.0)))
        rc_step = float(get_user_input("Enter the stepsize of r_c values (pc)", defaults.get('step_rc', 1.0)))
    else:
        min_vel = max_vel = 0.10
        vel_step = None

        min_rc = max_rc = 1.0
        rc_step = None

    apparent_q = float(get_user_input("Enter the apparent axis ratio (b/a)", defaults.get('apparent_q', 0.85)))

    i_min_deg = None
    try:
        if 0 < apparent_q <= 1:
            i_min_deg = float(np.degrees(np.arccos(apparent_q)))
        else:
            print("Warning: apparent axis ratio must be in (0,1].")
    except Exception as e:
        print(f"Warning: could not evaluate inclination due to: {e}")

    parameterize_inclination = get_user_input("Do you want to parametrize the inclination? (Y/N)")
    if parameterize_inclination == 'Y' or parameterize_inclination == 'y':
        print(f"For apparent axis ratio q={apparent_q:.2f}, inclinations must be in [acos(q), 90] deg = [{i_min_deg:.2f}, 90.00].")
        min_inc = float(get_user_input("Enter the minimum value for inclination (degrees)", defaults.get('min_inc', 60.0)))
        max_inc = float(get_user_input("Enter the maximum value for inclination (degrees)", defaults.get('max_inc', 90.0)))
        inc_step = float(get_user_input("Enter the stepsize of inclination values (deg)", defaults.get('step_inc', 1.0)))
    else:
        min_inc = max_inc = float(get_user_input("Enter the fixed value for inclination (degrees)", defaults.get('min_inc', 60.0)))
        inc_step = None

    # Effective inclination min bound (respect acos(q))
    min_inc_eff = min_inc
    if parameterize_inclination in ('Y','y') and i_min_deg is not None and 0 < apparent_q <= 1:
        if min_inc < i_min_deg:
            print(f"Note: raising min inclination from {min_inc:.2f}° to acos(q)={i_min_deg:.2f}°.")
        min_inc_eff = max(min_inc, i_min_deg)

    # For M_BH: step-based only
    mbhvalues, mbh_decimals = grid_from_step(min_mbh, max_mbh, mbh_step, include_end=True)
    if include_zero_mass_bh == 'Y' or include_zero_mass_bh == 'y':
        mbhvalues = np.insert(mbhvalues, 0, 0.00e00)  # Insert 0 mass BH at the start

    # Other parameters: step-based only
    mlvalues, ml_decimals = grid_from_step(min_ml, max_ml, ml_step, include_end=True)
    if ml_ratio_step is not None:
        ml_ratio_values, ml_ratio_decimals = grid_from_step(min_ml_ratio, max_ml_ratio, ml_ratio_step, include_end=True)
    else:
        ml_ratio_values, ml_ratio_decimals = np.array([min_ml_ratio], dtype=float), 4
    if inc_step is not None:
        incvalues, inc_decimals = grid_from_step(min_inc_eff, max_inc, inc_step, include_end=True)
    else:
        incvalues, inc_decimals = np.array([min_inc_eff], dtype=float), 2
    if vel_step is not None:
        velvalues, vel_decimals = grid_from_step(min_vel, max_vel, vel_step, include_end=True)
    else:
        velvalues, vel_decimals = np.array([min_vel], dtype=float), 2
    if rc_step is not None:
        rcvalues, rc_decimals = grid_from_step(min_rc, max_rc, rc_step, include_end=True)
    else:
        rcvalues, rc_decimals = np.array([min_rc], dtype=float), 2
    # Enforce the physical inclination lower bound: i >= acos(q_obs) and inform the user if any values are removed
    try:
        if i_min_deg is not None and 0 < apparent_q <= 1:
            original_count = len(incvalues)
            incvalues = incvalues[incvalues >= i_min_deg]
            filtered = original_count - len(incvalues)
            if filtered > 0:
                print(f"Note: {filtered} inclination value(s) below {i_min_deg:.2f}° were removed from the grid.")
        elif not (0 < apparent_q <= 1):
            print("Warning: apparent axis ratio must be in (0,1]; skipping inclination validity checks.")
    except Exception as e:
        print(f"Warning: could not evaluate valid inclination range due to: {e}")

    if len(incvalues) == 0:
        print("No valid inclination values remain after applying the acos(q) lower bound. Nothing to generate.")
        entries = np.empty((0, 5))
    else:
        entries = []
        step_entries = []
#        for i in range(len(mlvalues)):
#            for j in range(len(ml_ratio_values)):
#                for k in range(len(mbhvalues)):
#                    for l in range(len(velvalues)):
#                        for m in range(len(rcvalues)):
#                            for n in range(len(incvalues)):
#                                entries.append([mlvalues[i], ml_ratio_values[j], mbhvalues[k], velvalues[l], rcvalues[m], incvalues[n]])
#                                step_entries.append([ml_decimals, ml_ratio_decimals, mbh_decimals, vel_decimals, rc_decimals, inc_decimals])
        for i in range(len(mlvalues)):
            for j in range(len(ml_ratio_values)):
                for k in range(len(mbhvalues)):
                    for l in range(len(velvalues)):
                        for m in range(len(rcvalues)):
                            for n in range(len(incvalues)):
                                # Check if ml_decimals >= 3 and the value has a zero in the third decimal place
                                if ml_decimals >= 3 and (round(mlvalues[i], 3) == round(mlvalues[i], 2)):
                                    mlvalues[i] = round(mlvalues[i], 2)
                                    ml_decimals_used = 2
                                    entries.append([mlvalues[i], ml_ratio_values[j], mbhvalues[k], velvalues[l], rcvalues[m], incvalues[n]])
                                    step_entries.append([ml_decimals_used, ml_ratio_decimals, mbh_decimals, vel_decimals, rc_decimals, inc_decimals])
                                else:
                                    entries.append([mlvalues[i], ml_ratio_values[j], mbhvalues[k], velvalues[l], rcvalues[m], incvalues[n]])
                                    step_entries.append([ml_decimals, ml_ratio_decimals, mbh_decimals, vel_decimals, rc_decimals, inc_decimals])

        entries = np.array(entries)
        step_entries = np.array(step_entries)

    save = get_user_input("Do you want to save the modeling outputs? (Y/N)")

    # Helper to write entries with the simplified runalls/runallt signature
    def write_entries(path, entries, step_entries, cmd_name="runalls"):
        with open(path, 'w') as f:
            for i in range(len(entries)):
                f.write(
                    f'{cmd_name} {entries[i][0]:.{step_entries[i][0]}f} {entries[i][1]:.{step_entries[i][1]}f} {custom_format(entries[i][2])} {entries[i][3]:.{step_entries[i][3]}f} {entries[i][4]:.{step_entries[i][4]}f} {apparent_q:.2f} {entries[i][5]:.{step_entries[i][5]}f}\n'
                )
                    #{runalls/t} {M/L}               {M/L ratio}         {M_BH}                         {v_c}               {r_c}               {apparent_q}     {inclination}
    output_file = get_user_input("Enter the name of the output file", defaults.get('output_file', 'modeling_run'))

    # Print unique values for each parameter (after any inclination filtering)
    def _print_unique(label, arr, decimals, formatter):
        if arr.size == 0:
            print(f"{label}: []")
            return
        uniq = np.unique(np.round(arr.astype(float), decimals))
        s = ", ".join(formatter(x) for x in uniq)
        print(f"{label} ({len(uniq)}): [" + s + "]")

    # Print with consistent formatting
    print("")
    print("-"*10)
    print("The resultant parameter grid is:")
    _print_unique("M_BH", mbhvalues, mbh_decimals, lambda x: custom_format(float(x)))
    _print_unique("M/L", mlvalues, ml_decimals, lambda x: f"{x:.{ml_decimals}f}")
    _print_unique("M/L ratio (NSC/Bulge)", ml_ratio_values, ml_ratio_decimals, lambda x: f"{x:.{ml_ratio_decimals}f}")
    _print_unique("v_c (km/s)", velvalues, vel_decimals, lambda x: f"{x:.{vel_decimals}f}")
    _print_unique("r_c (pc)", rcvalues, rc_decimals, lambda x: f"{x:.{rc_decimals}f}")
    _print_unique("inclination (deg)", incvalues, inc_decimals, lambda x: f"{x:.{inc_decimals}f}")
    print("")

    if entries.size:
        if save == 'Y' or save == 'y':
            write_entries(output_file, entries, step_entries, cmd_name="runalls")
        else:
            write_entries(output_file, entries, step_entries, cmd_name="runallt")

    defaults = {
        'apparent_q': apparent_q,
        'min_mbh': custom_format(min_mbh),
        'max_mbh': custom_format(max_mbh),
        'step_mbh': custom_format(mbh_step) if mbh_step is not None else '',
        'min_ml': min_ml,
        'max_ml': max_ml,
        'step_ml': ml_step if ml_step is not None else '',
        'min_ml_ratio': min_ml_ratio,
        'max_ml_ratio': max_ml_ratio,
        'step_ml_ratio': ml_ratio_step if ml_ratio_step is not None else '',
        'min_inc': min_inc,
        'max_inc': max_inc,
        'step_inc': inc_step if inc_step is not None else '',
        'min_vel': min_vel,
        'max_vel': max_vel,
        'step_vel': vel_step if vel_step is not None else '',
        'min_rc': min_rc,
        'max_rc': max_rc,
        'step_rc': rc_step if rc_step is not None else '',
        'output_file': output_file
    }

    with open(defaults_file, 'w') as f:
        json.dump(defaults, f) # save last inputs as defaults

if __name__ == "__main__":
    main()


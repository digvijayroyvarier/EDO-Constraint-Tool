#!/usr/bin/env python
# coding: utf-8

# # Extended Dark Object Microlensing Constraint Calculator
# 
# Computes upper limits on the dark-matter fraction $f_{\mathrm{EDO}}$ in extended dark objects (EDOs) of a given mass profile (e.g. boson stars, ultra-compact minihalos, NFW or self-similar subhalos, uniform spheres, or point lenses) from gravitational microlensing survey data.
# 
# **Usage:** Run the code cell below and answer the prompts. Pressing Enter accepts the default shown, which corresponds to an OGLE-IV/LMC configuration.
# 
# ## Workflow
# 
# 0. **Statistical method:** (L) unbinned Poisson likelihood using the observed event timescales and a background model, with a frequentist ($\Delta\,2\ln\mathcal{L}=5.991$) or Bayesian (flat prior $f\in[0,10]$) 95% limit; (P) Poisson upper limit with zero background; (B) binned likelihood-ratio deviance against a binned foreground.
# 1. **EDO mass profile:** Built-in profile(s) or a custom $\rho(r)$, and one or more values of $R_{90}$ (radius enclosing 90% of the mass).
# 2. **Survey geometry:** Source fields $(D_S, l, b, N_\star)$ and exposure, entered as a table or read in bulk from a star-count file plus per-field efficiency files (optionally split into long- and short-baseline populations).
# 3. **Halos:** Milky-Way lens halo models S, B, C of Green (2017), or U which is a user-defined density (cored, cuspy, NFW, the contracted NFW of Cautun et al. 2020, or a custom expression) with a Maxwellian velocity distribution. Optionally, also the source's own halo.
# 4. **Observed data:** Event timescales $t_E$ (L), number of candidates and confidence level (P), or bin edges (B).
# 5. **Detection efficiency** $\varepsilon(t_E)$, per field.
# 6. **Background** $dN_{\mathrm{bkg}}/dt_E$ (L and B), as a density or a binned histogram.
# 
# The tool then tabulates the threshold impact parameter $u_T(\tau_M)$ for each profile, evaluates the event rate $d\Gamma/dt_E$ summed over fields, and computes the limit on a mass grid that is extended automatically until $f_{\mathrm{EDO}} \approx 1$ at both ends, for every combination of EDO type, Milky-Way model and $R_{90}$.
# 
# ## Input file formats
# 
# - **Tables** ($[t_E, \varepsilon]$, $[t_E, dN/dt_E]$, field tables): a Python list literal, or a path to a whitespace- or comma-delimited text file.
# - **Bulk star counts:** one row per field; the columns holding the field ID and star count are selected at the prompt.
# - **Per-field efficiency:** a folder or archive (.zip, .tar, .tar.gz, ...) with one file per field: rows `log10(tE_min) log10(tE_max) efficiency`, and a header comment line containing `field <ID>` (otherwise the file name is used as the ID).
# 
# ## Output
# 
# - A PNG of the limit versus mass (one panel per EDO type; colour encodes $R_{90}$, line style the Milky-Way model).
# - Optionally, one text file per curve with columns `M [M_sun]` and `f_EDO`.
# 
# ## References
# 
# Croon, McKeen & Raj (2020), arXiv:2002.08962 · Mróz et al. (2024), Nature, arXiv:2403.02386 · Mróz et al. (2024), ApJS 273, 4, arXiv:2403.02398 · Green (2017), arXiv:1705.10818 · Cautun et al. (2020), MNRAS · Evans (1993, 1994) · Alcock et al. (1995), ApJ 449, 28 · Niikura et al. (2019)
# 

# In[ ]:


# Extended Dark Object (EDO) microlensing constraint calculator.
# Interactive: run and answer the prompts ("Enter" accepts the default shown).

# ===========================
# Fixed physical constants
# ===========================

import ast
import glob
import os
import re
import tarfile
import time
import warnings
import zipfile

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.integrate import quad, dblquad
from scipy.interpolate import interp1d, CubicSpline
from scipy.optimize import brentq, minimize_scalar
from scipy.stats import chi2

warnings.filterwarnings("ignore", category=RuntimeWarning)

# Computer Modern (LaTeX-style) fonts for all figures, without requiring a TeX installation
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['cmr10', 'Computer Modern Roman', 'DejaVu Serif']
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['axes.unicode_minus'] = False        # the cm font has no unicode minus glyph
plt.rcParams['axes.formatter.use_mathtext'] = True

C_KMS = 2.99792e5                  # speed of light [km/s]
G_KPC = 4.30091e-6                 # Newton's constant [kpc Msun^-1 (km/s)^2]
RSUN_KPC = 2.25461e-11             # 1 solar radius, in kpc
MSUN_TO_G = 1.98847e33             # 1 solar mass, in grams (for the top M(g) axis on plots)
GEV_CM3_TO_MSUN_KPC3 = 2.6315e7    # GeV/cm^3  ->  Msun/kpc^3
KMS_TO_KPCYR = 1.0227e-9           # km/s      ->  kpc/yr
DAYS_PER_YEAR = 365.25

MU_THRESHOLD = 1.34        # standard microlensing detection-magnification threshold
DELTA_CHI2_95 = 5.991      # 95% CL threshold on the likelihood-ratio test statistic
SMALL_TAUM_EXACT_THRESHOLD = 1e-4   # below this, use the exact point-lens formula

def safe_log(x, floor=1e-300):
    """Natural log floored at a tiny positive value (never -inf)."""
    return np.log(max(float(x), floor))

print("Constants and safety helpers loaded.")


# ===================================================================================
# Interactive input helpers (invalid input re-prompts; blank input uses the default)
# ===================================================================================

def prompt_str(msg, default=None):
    suffix = f" [default: {default}]" if default is not None else ""
    while True:
        raw = input(msg + suffix + "\n> ").strip()
        if raw == "" and default is not None:
            return default
        if raw != "":
            return raw
        print("  A value is required here; please enter one.")

def prompt_float(msg, default=None, min_val=None, max_val=None):
    suffix = f" [default: {default}]" if default is not None else ""
    while True:
        raw = input(msg + suffix + "\n> ").strip()
        if raw == "" and default is not None:
            return float(default)
        try:
            val = float(raw)
            if min_val is not None and val < min_val:
                print(f"  Value must be >= {min_val}. Try again.")
                continue
            if max_val is not None and val > max_val:
                print(f"  Value must be <= {max_val}. Try again.")
                continue
            return val
        except ValueError:
            print("  Could not parse that as a number. Try again.")

def prompt_int(msg, default=None, min_val=None, max_val=None):
    """Prompt for an integer (an answer such as '1.0' is accepted)."""
    suffix = f" [default: {default}]" if default is not None else ""
    while True:
        raw = input(msg + suffix + "\n> ").strip()
        if raw == "" and default is not None:
            return int(default)
        try:
            val = int(round(float(raw)))
            if min_val is not None and val < min_val:
                print(f"  Value must be >= {min_val}. Try again.")
                continue
            if max_val is not None and val > max_val:
                print(f"  Value must be <= {max_val}. Try again.")
                continue
            return val
        except ValueError:
            print("  Could not parse that as an integer. Try again.")

def prompt_yes_no(msg, default=True):
    d = "Y/n" if default else "y/N"
    while True:
        raw = input(f"{msg} [{d}]\n> ").strip().lower()
        if raw == "":
            return default
        if raw in ("y", "yes"):
            return True
        if raw in ("n", "no"):
            return False
        print("  Please answer y or n.")

def _safe_literal_array(raw):
    """Parse a Python-literal nested list into a float array."""
    data = ast.literal_eval(raw)
    arr = np.array(data, dtype=float)
    return arr

def prompt_2d_array(msg, default_array=None, example_str=None, allow_none=False):
    """Prompt for an [x, y] table: a pasted Python literal, a two-column .csv/.txt file, or the default."""
    print(msg)
    if example_str:
        print("  Example format:")
        print(example_str)
    if default_array is not None:
        print("  Press Enter to use the default shown above.")
    if allow_none:
        print("  Type 'none' to skip this and assume zero.")
    while True:
        raw = input("> ").strip()
        if raw == "" and default_array is not None:
            return np.array(default_array, dtype=float)
        if allow_none and raw.lower() == "none":
            return None
        if raw == "":
            print("  A value is required here (no default available). Please enter data, a file path, or 'none' if permitted.")
            continue
        if os.path.isfile(raw):
            arr = None
            for delim in (None, ','):
                try:
                    cand = np.atleast_2d(np.loadtxt(raw, delimiter=delim))
                    if cand.shape[1] == 2:
                        arr = cand
                        break
                except Exception:
                    continue
            if arr is not None:
                return np.array(arr, dtype=float)
            print(f"  '{raw}' looked like a file but could not be read as a two-column "
                  f"(whitespace- or comma-delimited) table. Trying to parse it as a "
                  f"Python literal instead...")
        try:
            arr = _safe_literal_array(raw)
            if arr.ndim != 2 or arr.shape[1] != 2:
                print("  That parsed, but isn't a 2-column [x, y] array. Please re-enter.")
                continue
            return arr
        except Exception as e:
            print(f"  Could not parse that as a 2D array or a file path ({e}). Please try again.")

def prompt_field_table(msg, default_fields=None, example_str=None):
    """Prompt for the source fields, one [DS_kpc, l_deg, b_deg, Nstar] row per field."""
    print(msg)
    if example_str:
        print("  Example format:")
        print(example_str)
    while True:
        raw = input("> ").strip()
        if raw == "":
            if default_fields is not None:
                return [dict(DS=float(row[0]), l_deg=float(row[1]), b_deg=float(row[2]), Nstar=float(row[3]))
                        for row in default_fields]
            print("  A value is required here. Please enter data or a file path.")
            continue
        arr = None
        if os.path.isfile(raw):
            for delim in (None, ','):
                try:
                    cand = np.atleast_2d(np.loadtxt(raw, delimiter=delim))
                    if cand.shape[1] == 4:
                        arr = cand
                        break
                except Exception:
                    continue
            if arr is None:
                print(f"  '{raw}' looked like a file but could not be read as a four-column "
                      f"(whitespace- or comma-delimited) table. Trying to parse it as a "
                      f"Python literal instead...")
        if arr is None:
            try:
                arr = _safe_literal_array(raw)
            except Exception as e:
                print(f"  Could not parse that as a 2D array or a file path ({e}). Please try again.")
                continue
        if arr.ndim != 2 or arr.shape[1] != 4:
            print("  That parsed, but isn't a 4-column [DS, l_deg, b_deg, Nstar] array. Please re-enter.")
            continue
        if np.any(arr[:, 0] <= 0) or np.any(arr[:, 3] <= 0):
            print("  DS and Nstar must both be positive. Please re-enter.")
            continue
        return [dict(DS=row[0], l_deg=row[1], b_deg=row[2], Nstar=row[3]) for row in arr]

def prompt_list_of_floats(msg, default_list=None, allow_empty=True):
    print(msg)
    if default_list is not None:
        print(f"  Press Enter to use the default ({len(default_list)} values).")
    if allow_empty:
        print("  Type 'none' or leave input as an empty list [] for zero observed events.")
    while True:
        raw = input("> ").strip()
        if raw == "":
            if default_list is not None:
                return np.array(default_list, dtype=float)
            if allow_empty:
                return np.array([], dtype=float)
            print("  A value is required here; please enter a list of numbers.")
            continue
        if raw.lower() in ("none", "[]") and allow_empty:
            return np.array([], dtype=float)
        try:
            data = ast.literal_eval(raw)
            arr = np.array(data, dtype=float).ravel()
            return arr
        except Exception as e:
            print(f"  Could not parse that as a list of numbers ({e}). Please try again, "
                  f"e.g. [35.2, 12.8, 61.0]")


# =============================================================================
# Dark-object mass profile: choose a built-in profile, or enter your own rho(r)
# =============================================================================

# Precomputed [tau, m(tau)] tables (enclosed projected-mass fraction) and
# s_factor = R90/r_max for the built-in profiles.
_PRESET_UNIFORM_MPOINTS = [
    [3.16227766e-09,0],[1e-08,1.110223025e-16],[3.16227766e-08,1.443289932e-15],[1e-07,1.498801083e-14],
    [3.16227766e-07,1.499911306e-13],[1e-06,1.499911306e-12],[3.16227766e-06,1.500000124e-11],[1e-05,1.500000124e-10],
    [3.16227766e-05,1.499999902e-09],[0.0001,1.500000002e-08],[0.000316227766,1.499999962e-07],[0.001,1.499999625e-06],
    [0.00316227766,1.49999625e-05],[0.01,0.0001499962499],[0.0316227766,0.001499624937],[0.1,0.01496243726],
    [0.316227766,0.1461850318],[1,1]
]
_PRESET_UNIFORM_SFACTOR = 0.9654893846045981

_PRESET_BOSON_STAR_MPOINTS = [
    [1e-10,1.807749533190224e-19],[3.1622776601683795e-10,1.8077495331902245e-18],
    [1e-09,1.8077495331902244e-17],[3.1622776601683795e-09,1.8077495331902238e-16],
    [1e-08,1.807749533190222e-15],[3.162277660168379e-08,1.8077495331902053e-14],
    [1e-07,1.8077495331900377e-13],[3.162277660168379e-07,1.8077495331883604e-12],
    [1e-06,1.80774953317159e-11],[3.162277660168379e-06,1.8077495330038836e-10],
    [0.00001,1.8077495313268211e-09],[0.000031622776601683795,1.8077495145561926e-08],
    [0.0001,1.807749346849599e-07],[0.00031622776601683794,1.8077476697529882e-06],
    [0.001,0.00001807730895706264],[0.0031622776601683794,0.00018075628332062753],
    [0.01,0.0018058403362508884],[0.0,0.0],[0.05,0.044028669553117614],
    [0.1,0.16320496289739725],[0.15,0.3256698681882252],[0.2,0.49576963836783117],
    [0.25,0.6469425590505278],[0.3,0.7663702565925695],[0.35,0.8526349014884668],
    [0.4,0.9107334128773106],[0.45,0.9477466680990138],[0.5,0.9702953077854352],
    [0.55,0.983539881660972],[0.6,0.9910881615314362],[0.65,0.9952821256578751],
    [0.7,0.997561978637135],[0.75,0.9987772454038498],[0.8,0.9994128401199277],
    [0.85,0.9997381765746003],[0.9,0.9998994768285666],[0.95,0.999741534772131],
    [1.0,1.0]
]
_PRESET_BOSON_STAR_SFACTOR = 0.5392169551010401

_PRESET_UCMH32_MPOINTS = [
    [1e-10,2.62204e-15],[3.16227766e-10,1.47447e-14],[1e-09,8.29147e-14],[3.16227766e-09,4.66255e-13],
    [1e-08,2.62197e-12],[3.16227766e-08,1.47435e-11],[1e-07,8.29017e-11],[3.16227766e-07,4.66125e-10],
    [1e-06,2.62056e-09],[3.16227766e-06,1.47299e-08],[1e-05,8.27667e-08],[3.16227766e-05,4.64775e-07],
    [0.0001,2.60706e-06],[0.000316227766,1.45949e-05],[0.001,8.14167e-05],[0.00316227766,0.000451275],
    [0.01,0.00247206],[0.0316227766,0.0132448],[0.05,0.025565],[0.1,0.0679092],
    [0.15,0.11854],[0.2,0.174403],[0.25,0.233709],[0.3,0.295225],
    [0.35,0.358015],[0.4,0.421322],[0.45,0.484503],[0.5,0.546984],
    [0.55,0.608233],[0.6,0.667741],[0.65,0.724996],[0.7,0.779464],
    [0.75,0.830563],[0.8,0.87762],[0.85,0.919803],[0.9,0.955976],
    [0.95,0.984309],[1,1]
]
_PRESET_UCMH32_SFACTOR = 0.9321697517997111

_PRESET_R94_MPOINTS = [
    [0.0001,0.001289578478],[0.000127427499,0.001558695056],[0.000162377674,0.001881473277],[0.000206913808,0.002268566735],
    [0.00026366509,0.002732835997],[0.000335981829,0.003289667652],[0.00042813324,0.003957517197],[0.000545559478,0.004758462697],
    [0.000695192796,0.00571907673],[0.00088586679,0.006871182068],[0.001128837892,0.008252920392],[0.001438449888,0.009910029793],
    [0.001832980711,0.0118974123],[0.002335721469,0.01428076331],[0.002976351442,0.01713964022],[0.003792690191,0.02056674669],
    [0.004832930239,0.02467700548],[0.006158482111,0.02960552733],[0.007847599704,0.03551471202],[0.01,0.04265929024],
    [0.0316227766,0.1009322322],[0.1,0.2370566285],[0.15,0.3186126916],[0.2,0.3916836473],
    [0.25,0.4584158537],[0.3,0.5199769329],[0.316227766,0.5389730206],[0.35,0.5770848228],
    [0.4,0.6302129244],[0.45,0.6796854019],[0.5,0.7257266449],[0.55,0.7684886019],
    [0.6,0.8080660072],[0.65,0.8445039834],[0.7,0.87779978],[0.75,0.9078985426],
    [0.8,0.934680742],[0.85,0.9579341831],[0.9,0.9772892946],[0.95,0.9920337597],
    [1,1]
]
_PRESET_R94_SFACTOR = 0.8689404461562149   # pure r^-9/4 power law (also the self-similar subhalo)
_PRESET_SELFSIM_MPOINTS = _PRESET_R94_MPOINTS
_PRESET_SELFSIM_SFACTOR = _PRESET_R94_SFACTOR

_PRESET_NFW_MPOINTS = [
    [0.0001,6.618749504e-05],[0.000127427499,0.000102049521],[0.000162377674,0.000156901619],[0.000206913808,0.0002404864401],
    [0.00026366509,0.000367319717],[0.000335981829,0.0005588648213],[0.00042813324,0.0008465775177],[0.000545559478,0.001276079786],
    [0.000695192796,0.001912729946],[0.00088586679,0.002848802999],[0.001128837892,0.004212329468],[0.001438449888,0.006177299249],
    [0.001832980711,0.008974360714],[0.002335721469,0.01290032213],[0.002976351442,0.01832379802],[0.003792690191,0.02568353509],
    [0.004832930239,0.03547582033],[0.006158482111,0.04822852084],[0.007847599704,0.06446207439],[0.01,0.08464174576],
    [0.0316227766,0.2411702166],[0.1,0.4840720398],[0.15,0.5820191901],[0.2,0.6534584858],
    [0.25,0.7093661118],[0.3,0.7549933126],[0.316227766,0.7681165188],[0.35,0.7932586323],
    [0.4,0.8259492612],[0.45,0.8542359438],[0.5,0.8789238514],[0.55,0.9005865695],
    [0.6,0.919642636],[0.65,0.9364013478],[0.7,0.9510907682],[0.75,0.9638743583],
    [0.8,0.9748589971],[0.85,0.9840940171],[0.9,0.991555088],[0.95,0.9970842587],
    [1,1]
]
_PRESET_NFW_SFACTOR = 0.6898142503159757   # for rs = r_max/100


# Projected enclosed-mass fraction m(tau) of a spherical rho(r), and R90.
def _compute_m_of_tau(rho_func, r_max, tau_values, epsabs=1e-11, epsrel=1e-9, limit=150):
    def rho_hat(x):
        if x <= 0 or x > 1.0:
            return 0.0
        try:
            val = np.asarray(rho_func(x * r_max)).item()
        except Exception:
            return 0.0
        return val if np.isfinite(val) and val >= 0 else 0.0

    def inner_upper(sigma):
        return np.sqrt(max(0.0, 1.0 - sigma**2))

    def integrand(lam, sigma):
        return sigma * rho_hat(np.sqrt(sigma**2 + lam**2))

    D, _ = quad(lambda g: g**2 * rho_hat(g), 0.0, 1.0, epsabs=epsabs, epsrel=epsrel, limit=limit)
    if not np.isfinite(D) or D <= 0:
        raise ValueError("The total mass integral of rho(r) is zero, negative, or "
                          "non-finite -- check that rho(r) > 0 somewhere in (0, r_max).")

    tau_values = np.atleast_1d(np.asarray(tau_values, dtype=float))
    m_values = np.empty_like(tau_values)
    for i, tau in enumerate(tau_values):
        if tau <= 0:
            m_values[i] = 0.0
        elif tau >= 1.0:
            m_values[i] = 1.0
        else:
            try:
                N, _ = dblquad(integrand, 0.0, tau, 0.0, inner_upper, epsabs=epsabs, epsrel=epsrel)
                m_values[i] = np.clip(N / D, 0.0, 1.0)
            except Exception:
                m_values[i] = m_values[i-1] if i > 0 else 0.0
    return np.maximum.accumulate(m_values)

def _compute_R90_and_scale_factor(rho_func, r_max, epsabs=1e-11, epsrel=1e-9, limit=150):
    def rho_hat(x):
        if x <= 0 or x > 1.0:
            return 0.0
        val = np.asarray(rho_func(x * r_max)).item()
        return val if np.isfinite(val) and val >= 0 else 0.0
    def enclosed_mass_fraction(x):
        num, _ = quad(lambda g: g**2 * rho_hat(g), 0.0, x, epsabs=epsabs, epsrel=epsrel, limit=limit)
        den, _ = quad(lambda g: g**2 * rho_hat(g), 0.0, 1.0, epsabs=epsabs, epsrel=epsrel, limit=limit)
        return num / den
    s = brentq(lambda x: enclosed_mass_fraction(x) - 0.9, 1e-8, 1.0, xtol=1e-10)
    return s * r_max, s

def _build_default_tau_grid(n_log=20, n_lin=21, tau_min=1e-4):
    return np.concatenate([np.geomspace(tau_min, 0.05, n_log), np.linspace(0.06, 1.0, n_lin)])

def generate_profile_from_density(rho_func, r_max, verbose=True):
    """Compute the enclosed projected-mass table for an arbitrary rho(r)."""
    tau_grid = _build_default_tau_grid()
    if verbose:
        print("  Computing the projected mass profile from your rho(r) "
              "(this takes a little while)...")
    m_values = _compute_m_of_tau(rho_func, r_max, tau_grid)
    points = np.column_stack([tau_grid, m_values])
    _, s_factor = _compute_R90_and_scale_factor(rho_func, r_max)
    if verbose:
        print(f"  Done. s_factor = R90/r_max = {s_factor:.6f}")
    return points, s_factor


# Detect when a custom rho(r) matches a built-in profile, so its precomputed table can be reused.
_MATCH_TEST_R = np.array([1e-4, 1e-3, 1e-2, 0.05, 0.15, 0.35, 0.6, 0.85])

def _match_safe_sample(rho_func, r_max):
    vals = []
    for r in _MATCH_TEST_R:
        try:
            v = np.asarray(rho_func(r * r_max)).item()
        except Exception:
            return None
        if not (np.isfinite(v) and v > 0):
            return None
        vals.append(v)
    return np.array(vals)

def _match_powerlaw_residual(log_rho, log_r, n):
    pred = n * log_r
    offset = np.mean(log_rho - pred)
    return np.sqrt(np.mean((log_rho - (pred + offset))**2))

def _match_general_powerlaw_residual(log_rho, log_r):
    n_fit, offset = np.polyfit(log_r, log_rho, 1)
    return np.sqrt(np.mean((log_rho - (n_fit*log_r + offset))**2))

def _match_nfw_residual(log_rho, log_r):
    def cost(log_rs):
        r = np.exp(log_r); rs = np.exp(log_rs); u = r/rs
        pred = -np.log(u) - 2*np.log(1+u)
        offset = np.mean(log_rho - pred)
        return np.sqrt(np.mean((log_rho - (pred+offset))**2))
    res = minimize_scalar(cost, bounds=(-10, 10), method='bounded')
    return res.fun, np.exp(res.x)

def match_known_profile(rho_func, r_max, tol=0.03):
    """Return (name, info) if rho(r) matches a built-in profile, else (None, {})."""
    samples = _match_safe_sample(rho_func, r_max)
    if samples is None:
        return None, {}
    log_r = np.log(_MATCH_TEST_R)
    log_rho = np.log(samples)

    candidates = {"uniform_sphere": 0.0, "ucmh_r32": -1.5, "r94_powerlaw": -2.25}
    best_name, best_resid = None, tol
    for name, n in candidates.items():
        resid = _match_powerlaw_residual(log_rho, log_r, n)
        if resid < best_resid:
            best_resid, best_name = resid, name

    general_resid = _match_general_powerlaw_residual(log_rho, log_r)
    if general_resid > tol:
        nfw_resid, rs_fit = _match_nfw_residual(log_rho, log_r)
        rs_at_boundary = not (1e-4 < rs_fit < 1e4)
        if nfw_resid < best_resid and not rs_at_boundary:
            return "nfw", {"rs_over_rmax": rs_fit, "residual": nfw_resid}

    if best_name is not None:
        return best_name, {"residual": best_resid}
    return None, {}

# ==========================================================================================
# Statistical method: unbinned likelihood (L), Poisson upper limit (P), binned deviance (B)
# ==========================================================================================
print("#" * 78)
print("#  CHOOSE YOUR ANALYSIS METHOD")
print("#" * 78)
print("""
  L : UNBINNED POISSON LIKELIHOOD. Needs the actual list of your survey's
      observed event timescales, a digitized detection-efficiency curve, and a
      digitized expected-background curve (the timescale distribution of
      ordinary events that mimic a lens).
      Appropriate when your survey has many (tens to thousands of) candidate
      events and a characterized background shape in tE (e.g. OGLE-IV observations of LMC).

  P : Simple POISSON UPPER LIMIT, zero background assumed. Needs only the
      NUMBER of candidate events your survey observed (e.g. 0, 1, 2, ...) and
      a confidence level; no event-timescale list and no background model are
      used at all (a detection-efficiency curve is still needed, however).
      This is the method used for EROS-2 by Croon, McKeen & Raj (2020), and more
      generally whenever a survey reports only a handful of candidates with
      no characterized background shape e.g. also the Subaru/HSC search of
      M31 (Niikura et al. 2019, 1 candidate) and the Kepler PBH search
      (Griest, Cieplak & Lehner 2013, 0 candidates). NOT appropriate if your
      survey has a well-characterized, non-negligible background that varies
      with tE; use L instead in that case.

  B : BINNED DEVIANCE (the "kappa statistic"). Needs the same continuous
      background curve as method L (plus a detection-efficiency curve), but
      not a raw list of individual observed event timescales. It only needs a choice
      of bin edges, over which that curve is integrated to get each bin's
      foreground count. Use this when you have a foreground/background curve
      but no individual events to compare it against directly, only its
      aggregate shape e.g. Niikura et al. (2019)'s published tE
      distribution for OGLE-IV, which is what Croon, McKeen & Raj (2020)
      reuse (via Niikura's own choice of bins) for their OGLE-IV
      Milky-Way-Bulge limit.
""")
ANALYSIS_METHOD_CHOICE = prompt_str(
    "Enter L (likelihood), P (Poisson upper limit), or B (binned deviance).", default="L").strip().upper()
while ANALYSIS_METHOD_CHOICE not in ("L", "P", "B"):
    ANALYSIS_METHOD_CHOICE = prompt_str("  Please enter L, P, or B.", default="L").strip().upper()
ANALYSIS_METHOD = {"L": "likelihood", "P": "poisson_ul", "B": "binned"}[ANALYSIS_METHOD_CHOICE]
_METHOD_DESCRIPTIONS = {
    "likelihood": "unbinned Poisson likelihood",
    "poisson_ul": "Poisson upper limit (zero background assumed)",
    "binned": "binned deviance (kappa statistic)",
}
print(f"\n-> Analysis method: {_METHOD_DESCRIPTIONS[ANALYSIS_METHOD]}")


print("STEP 1 of 6: Dark-object mass profile")
print("=" * 70)
print("""
Choose one or more profiles (the built-in ones show what rho(r) each assumes).
Enter a single number, or a list to get a separate constraint analysis (its
own plot panel, computed independently) for each EDO type in one run, e.g.
[1, 2] for boson star AND UCMH-like. Every other input (R90, Milky-Way
model(s), survey, efficiency, background, analysis method, etc.) is shared
across all of them, rather than asking you to repeat the whole tool once per
type. Custom (5) cannot be combined with other types in a list and must be
entered alone, since each custom rho(r) needs its own separate prompts.

  Looking for a point lens? There is no separate option for it here. Just
  pick any of the extended profiles below, then enter R90 = 0 at the R90
  prompt that follows. That gives the exact point-lens result regardless of
  which profile you picked, since the profile's shape has no effect once
  R90 = 0 (tauM = R90/R_Einstein is then identically 0, and the lensing
  formula already has an exact point-lens branch for that limit).

  0. Uniform sphere         -- rho(r) = const
  1. Boson star             -- non-relativistic Schrodinger-Poisson ground state;
                                rho(r) has no simple closed form (solved numerically)
  2. UCMH-like              -- rho(r) ~ r^-3/2
  3. NFW subhalo            -- rho(r) = rho_s / [(r/rs)(1+r/rs)^2], with rs/r_max = 0.01
                                assumed (i.e. r_max = 100 rs, following Croon, McKeen & Raj 2020)
  4. Self-similar subhalo   -- rho(r) ~ r^-9/4
  5. Custom -- enter your own rho(r) (must be entered alone, not in a list)
""")

def _parse_profile_choice_list(raw):
    """Parse a profile selection such as 2 or [0, 2, 3] into a sorted list of ints in 0-5 (None if invalid)."""
    toks = [t for t in re.split(r"[\s,;\[\]()]+", raw.strip()) if t]
    if not toks:
        return None
    try:
        vals = [int(t) for t in toks]
    except ValueError:
        return None
    if any(v not in range(6) for v in vals):
        return None
    vals = sorted(set(vals))
    if len(vals) > 1 and 5 in vals:
        return None
    return vals

while True:
    _raw_profile = prompt_str("Enter a number 0-5, or a list such as [1, 2].", default="1")
    _profile_choices = _parse_profile_choice_list(_raw_profile)
    if _profile_choices is not None:
        break
    print("  Could not read that. Use numbers 0-5 only, e.g. 2 or [1, 2]. And note "
          "that 5 (custom) cannot be combined with other types in a list.")

_PRESET_LABELS = {
    0: "Uniform sphere", 1: "Boson star", 2: "UCMH-like (r^-3/2)",
    3: "NFW subhalo", 4: "Self-similar subhalo",
}

edo_profiles_raw = []

for _profile_choice in _profile_choices:
    if _profile_choice == 0:
        mass_profile_points_raw = np.array(_PRESET_UNIFORM_MPOINTS)
        s_factor = _PRESET_UNIFORM_SFACTOR
        print(f"\n-> Uniform sphere selected (precomputed table, s_factor={s_factor:.6f}).")

    elif _profile_choice == 1:
        mass_profile_points_raw = np.array(_PRESET_BOSON_STAR_MPOINTS)
        s_factor = _PRESET_BOSON_STAR_SFACTOR
        print(f"\n-> Boson star selected (precomputed table, s_factor={s_factor:.6f}).")

    elif _profile_choice == 2:
        mass_profile_points_raw = np.array(_PRESET_UCMH32_MPOINTS)
        s_factor = _PRESET_UCMH32_SFACTOR
        print(f"\n-> UCMH-like (r^-3/2) selected (precomputed table, s_factor={s_factor:.6f}).")

    elif _profile_choice == 3:
        mass_profile_points_raw = np.array(_PRESET_NFW_MPOINTS)
        s_factor = _PRESET_NFW_SFACTOR
        print(f"\n-> NFW subhalo selected (precomputed table, rs=r_max/100, s_factor={s_factor:.6f}).")

    elif _profile_choice == 4:
        mass_profile_points_raw = np.array(_PRESET_SELFSIM_MPOINTS)
        s_factor = _PRESET_SELFSIM_SFACTOR
        print(f"\n-> Self-similar subhalo selected (precomputed table, s_factor={s_factor:.6f}).")

    else:
        print("\nEnter rho(r) as a one-line NumPy expression in r (any normalization is fine;")
        print("only the SHAPE matters). Do not implement your own cutoff inside the expression;")
        print("that is handled separately below via r_max.")
        print("  Example: r**(-1.8)   or   1.0/((r/0.02)*(1+r/0.02)**2)")
        _rho_expr = prompt_str("rho(r) =")

        def _custom_rho(r, _expr=_rho_expr):
            try:
                val = eval(_expr, {"np": np, "__builtins__": {}}, {"r": r})
                val = float(val)
                return val if (np.isfinite(val) and val >= 0) else 0.0
            except Exception:
                return 0.0

        _r_max_local = prompt_float(
            "\nEnter r_max: the radius, in the SAME units your rho(r) expression uses for r,\n"
            "beyond which the object's density is treated as zero (its outer edge).",
            default=1.0, min_val=1e-300
        )

        print("\nChecking whether this matches one of the built-in profiles...")
        _match_name, _match_info = match_known_profile(_custom_rho, _r_max_local)

        _preset_lookup = {
            "uniform_sphere": (_PRESET_UNIFORM_MPOINTS, _PRESET_UNIFORM_SFACTOR, "uniform sphere"),
            "ucmh_r32": (_PRESET_UCMH32_MPOINTS, _PRESET_UCMH32_SFACTOR, "UCMH-like (r^-3/2)"),
            "r94_powerlaw": (_PRESET_SELFSIM_MPOINTS, _PRESET_SELFSIM_SFACTOR, "self-similar subhalo (r^-9/4)"),
        }

        if _match_name in _preset_lookup:
            pts, sf, label = _preset_lookup[_match_name]
            print(f"  Matched: {label} (residual={_match_info['residual']:.2e}). "
                  f"Using the precomputed, validated table.")
            mass_profile_points_raw = np.array(pts)
            s_factor = sf

        elif _match_name == "nfw":
            rs_ratio = _match_info["rs_over_rmax"]
            if 0.0067 < rs_ratio < 0.015:  # close to the preset's rs/r_max=0.01
                print(f"  Matched: NFW subhalo with rs/r_max={rs_ratio:.4f} (close to the "
                      f"precomputed table's rs/r_max=0.01, residual={_match_info['residual']:.2e}). "
                      f"Using the precomputed, validated table.")
                mass_profile_points_raw = np.array(_PRESET_NFW_MPOINTS)
                s_factor = _PRESET_NFW_SFACTOR
            else:
                print(f"  Recognized as NFW-shaped, but with rs/r_max={rs_ratio:.4f} (different "
                      f"from the precomputed table's rs/r_max=0.01), so computing your specific "
                      f"profile fresh rather than reusing a mismatched table.")
                mass_profile_points_raw, s_factor = generate_profile_from_density(_custom_rho, _r_max_local)
        else:
            print("  No match to a built-in profile. Computing your profile fresh.")
            mass_profile_points_raw, s_factor = generate_profile_from_density(_custom_rho, _r_max_local)

        print(f"\n-> Custom profile ready. s_factor = {s_factor:.6f}")

    _edo_label = _PRESET_LABELS.get(_profile_choice, "Custom profile")
    edo_profiles_raw.append((_edo_label, mass_profile_points_raw, s_factor))

if len(edo_profiles_raw) > 1:
    print(f"\n-> {len(edo_profiles_raw)} EDO types queued: {[lbl for lbl, _, _ in edo_profiles_raw]}")

while True:
    R90_list_solar_radii = prompt_list_of_floats(
        "\nEnter R90 (radius enclosing 90% of the object's total mass), in solar\n"
        "radii, as a single number, or a list to overlay several constraint curves,\n"
        "e.g. 30  or  [10, 30, 60]. Enter 0 (or include 0 in the list) for the exact\n"
        "point-lens result at that entry, regardless of the profile chosen above.",
        default_list=[30.0], allow_empty=False
    )
    if np.any(R90_list_solar_radii < 0):
        print("  R90 cannot be negative. Please re-enter.")
        continue
    if len(R90_list_solar_radii) == 0:
        print("  At least one R90 value is required. Please re-enter.")
        continue
    break

R90_kpc = float(R90_list_solar_radii[0]) * RSUN_KPC

print(f"\n-> R90 value(s), in solar radii: {list(R90_list_solar_radii)}")
print(f"-> scale factor = {s_factor}")


# =============================================================================
# Survey geometry: distance to source, exposure, star count
# =============================================================================
print("\nSTEP 2 of 6: Survey geometry")
print("=" * 70)

print("""
A survey can monitor more than one source field at once (e.g. EROS-2 combines
its LMC and SMC fields into a single limit). Every field shares the same
exposure, analysis-method inputs, and (unless told otherwise) background
(below); only its own geometry, star count, and detection efficiency differ.
Every field's rate is summed (not plotted separately) into one combined
constraint.
""")

FIELD_DATA_MODE = prompt_str(
    "How do you have this survey's field data? Enter 'combined' to paste/point to\n"
    "one [DS, l_deg, b_deg, Nstar] table (fine for a handful of fields),\n"
    "or 'per_field_bulk' if you have many fields (e.g. O(100)+, like OGLE-IV's\n"
    "individual LMC pointings) with their own star counts and efficiency curves,\n"
    "but all sharing one (DS, l, b).",
    default="combined"
).strip().lower()
while FIELD_DATA_MODE not in ("combined", "per_field_bulk"):
    FIELD_DATA_MODE = prompt_str("  Please enter 'combined' or 'per_field_bulk'.", default="combined").strip().lower()

_split_baselines = False  # True if star counts are split into long/short observing baselines
_FIELD_EFFICIENCIES_PREBUILT = False  # True if the bulk branch already read per-field efficiencies

if FIELD_DATA_MODE == "combined":
    FIELDS = prompt_field_table(
        "Enter the field table, e.g. the one below (EROS-2's LMC + SMC fields):",
        default_fields=[[50.0, 280.46, -32.89, 29271600.0]],
        example_str="  [[50.0, 280.46, -32.89, 5.49e6], [60.0, 302.81, -44.33, 0.86e6]]"
    )
else:
    print("""
--- Bulk per-field setup ---
This builds the field table from two sources instead of typed-in rows:
  1) ONE shared (DS, l_deg, b_deg) for every field (prompted just below).
  2) A bulk star-count file and a folder of per-field efficiency files,
     matched to each other by field ID (an identifier like "LMC501").
Only fields found in BOTH sources are kept; any field present in just one is
reported and excluded (its rate can't be computed without both pieces).
""")
    _DS_shared = prompt_float("Distance to the source, DS (kpc)", default=50.0, min_val=1e-3)
    _l_shared = prompt_float("Galactic longitude, l_deg (deg)", default=280.46)
    _b_shared = prompt_float("Galactic latitude, b_deg (deg)", default=-32.89)

    # Optional split into long- and short-baseline star populations (e.g. OGLE-III+IV
    # vs OGLE-IV only), each with its own efficiency files and exposure.
    _split_baselines = prompt_yes_no(
        "\nDoes this survey have two sub-populations of source stars with genuinely\n"
        "different observing baselines (e.g. OGLE-IV's own 'observed during both\n"
        "OGLE-III and OGLE-IV' vs 'observed during OGLE-IV only'), each needing its\n"
        "own efficiency dataset and exposure? Answer 'n' for the ordinary case (one\n"
        "star count, one efficiency dataset, one exposure, for every field).",
        default=False
    )

    if _split_baselines:
        print("""
--- Two-baseline setup ---
This asks for everything twice: once for the long-baseline population, once
for the short-baseline one. Then combines them: the short-
baseline population's star count is internally rescaled by
(short_exposure / long_exposure) so that a single global Exposure (set to the
LONG baseline) can be used for both populations without double-counting.
""")
        _nstar_val1_col = prompt_str(
            "Which column of the star-count file holds the LONG-baseline star count\n"
            "(e.g. OGLE-IV's 'observed during both OGLE-III and OGLE-IV')? Enter\n"
            "'first', 'last', or a column number (1-indexed).", default="last"
        ).strip().lower()
        _exposure_long = prompt_float(
            "Long-baseline exposure (days) -- e.g. OGLE-III+IV: 2001 June 24 to 2020 March 15 = 6839 days.",
            default=6839.0, min_val=1e-6
        )
        _nstar_val2_col = prompt_str(
            "Which column of the star-count file holds the SHORT-baseline star count\n"
            "(e.g. OGLE-IV's 'observed during OGLE-IV only')? Enter 'first', 'last',\n"
            "or a column number (1-indexed).", default="last"
        ).strip().lower()
        _exposure_short = prompt_float(
            "Short-baseline exposure (days) -- e.g. OGLE-IV only: 2010 June 29 to 2020 March 15 = 3546 days.",
            default=3546.0, min_val=1e-6
        )

    _nstar_path = prompt_str(
        "\nPath to the star-count file ('#' comment lines are skipped; delimiter \n"
        "(comma or whitespace) is auto-detected from the file's own first data line).\n"
        "e.g. the OGLE-IV field table format, 'LMC501 05:19:00 -71:46:10 ... 787.8',\n"
        "or a plain two-column file, 'LMC501, 787800'. (The next two questions ask\n"
        "which column of whichever format you give is the field ID and which is the\n"
        "star count. You don't need to restructure the file to match either example.)",
        default=""
    )
    _nstar_scale = prompt_float(
        "Multiplier to convert the star-count column into an ACTUAL star count\n"
        "(e.g. 1000 if the file gives star counts in thousands; 1 if it's already an absolute count).",
        default=1000.0, min_val=0.0
    )
    _nstar_id_col = prompt_str(
        "Which column holds the field ID? Enter 'first', 'last', or a column\n"
        "number (1-indexed).", default="first"
    ).strip().lower()
    if not _split_baselines:
        _nstar_val_col = prompt_str(
            "Which column holds the star count? Enter 'first', 'last', or a column\n"
            "number (1-indexed).", default="last"
        ).strip().lower()

    # Star-count file -> {field ID: Nstar}
    def _resolve_col(spec):
        if spec == "first":
            return 0
        if spec == "last":
            return -1
        return int(spec) - 1

    def _parse_bulk_nstar_table(path, scale, id_col_spec, val_col_spec):
        table = {}
        _id_idx, _val_idx = _resolve_col(id_col_spec), _resolve_col(val_col_spec)
        with open(path) as _fh:
            _raw_lines = [l.rstrip('\n') for l in _fh]
        _delim = None
        _first_data_line = None
        for _line in _raw_lines:
            _line = _line.strip()
            if not _line or _line.startswith('#'):
                continue
            _delim = ',' if ',' in _line else None
            _first_data_line = _line
            break
        _n_tried, _n_failed = 0, 0
        for _line in _raw_lines:
            _line = _line.strip()
            if not _line or _line.startswith('#'):
                continue
            _parts = [p.strip() for p in (_line.split(_delim) if _delim else _line.split())]
            if len(_parts) < 2:
                continue
            _n_tried += 1
            try:
                table[_parts[_id_idx]] = float(_parts[_val_idx]) * scale
            except (ValueError, IndexError):
                _n_failed += 1
                continue
        if _n_tried > 0 and _n_failed / _n_tried > 0.5:
            _example_parts = [p.strip() for p in (_first_data_line.split(_delim) if _delim
                                                   else _first_data_line.split())]
            raise ValueError(
                f"Column index problem reading '{path}': {_n_failed}/{_n_tried} data lines failed "
                f"to parse with id_col='{id_col_spec}', val_col='{val_col_spec}'. The first data "
                f"line split into {len(_example_parts)} token(s):\n  {_example_parts}\n"
                f"A common cause: a documented byte-format column number (e.g. from a 'ra_h, ra_m, "
                f"ra_s' style description) doesn't match this WHITESPACE-split parser's own column "
                f"count when fields like RA/DEC are written as one colon-joined token (e.g. "
                f"'05:19:00') rather than space-separated pieces. Re-count against the token list "
                f"shown above, not the original format description, and try again."
            )
        return table

    if _split_baselines:
        _nstar_table1 = _parse_bulk_nstar_table(_nstar_path, _nstar_scale, _nstar_id_col, _nstar_val1_col)
        _nstar_table2 = _parse_bulk_nstar_table(_nstar_path, _nstar_scale, _nstar_id_col, _nstar_val2_col)
        print(f"-> Read {len(_nstar_table1)} field(s) with a long-baseline star count "
              f"(total {sum(_nstar_table1.values()):,.0f} stars) and {len(_nstar_table2)} field(s) "
              f"with a short-baseline star count (total {sum(_nstar_table2.values()):,.0f} stars) "
              f"from the star-count file.")
    else:
        _nstar_table = _parse_bulk_nstar_table(_nstar_path, _nstar_scale, _nstar_id_col, _nstar_val_col)
        print(f"-> Read {len(_nstar_table)} field(s) from the star-count file "
              f"(total {sum(_nstar_table.values()):,.0f} stars).")

    # Efficiency file -> eps(tE): rows [log10 tE_min, log10 tE_max, eps], interpolated through
    # bin log-midpoints; field ID from a '# ... field <ID>' header line, else the file name.
    def _parse_one_field_efficiency_file(path):
        _field_id = None
        _rows = []
        with open(path) as _fh:
            for _line in _fh:
                _line = _line.strip()
                if not _line:
                    continue
                if _line.startswith('#'):
                    if _field_id is None:
                        _m = re.search(r'\bfield\s+(\S+)', _line, re.IGNORECASE)
                        if _m:
                            _field_id = _m.group(1)
                    continue
                _parts = _line.split()
                if len(_parts) >= 3:
                    try:
                        _rows.append([float(_parts[0]), float(_parts[1]), float(_parts[2])])
                    except ValueError:
                        continue
        if _field_id is None:
            _field_id = os.path.splitext(os.path.basename(path))[0]
        if not _rows:
            return _field_id, None, None, None
        _arr = np.array(_rows, dtype=float)
        _order = np.argsort(_arr[:, 0])
        _arr = _arr[_order]
        _logtE_mid = 0.5 * (_arr[:, 0] + _arr[:, 1])
        _tE_centers = 10.0 ** _logtE_mid
        _effs = np.clip(_arr[:, 2], 0.0, 1.0)
        _lo, _hi = float(_tE_centers.min()), float(_tE_centers.max())
        _interp = interp1d(_tE_centers, _effs, kind='cubic' if len(_tE_centers) >= 4 else 'linear',
                           bounds_error=False, fill_value=0.0)
        def _fn(tE, _interp=_interp, _lo=_lo, _hi=_hi):
            if _lo < tE <= _hi:
                return float(np.clip(_interp(tE), 0.0, 1.0))
            return 0.0
        return _field_id, _fn, _lo, _hi

    def _strip_archive_ext(path):
        for _ext in (".tar.gz", ".tar.bz2", ".tar.xz", ".tgz", ".zip", ".tar"):
            if path.lower().endswith(_ext):
                return path[:-len(_ext)]
        return os.path.splitext(path)[0]

    def _read_eff_folder(tag=""):
        """Read a folder or archive of per-field efficiency files; returns (eff_by_field, bounds_by_field)."""
        _eff_dir = prompt_str(
            f"\nPath to the FOLDER containing one efficiency file per field{tag}, OR an\n"
            "archive of that folder (zip, .tar, .tar.gz/.tgz, .tar.bz2, or .tar.xz are\n"
            "all fine) useful in notebook environments where only a single file can be\n"
            "uploaded, not a whole folder. It's auto-detected (by the file's actual\n"
            "contents, not its name) and extracted for you, and any folder nesting\n"
            "inside the archive is handled automatically. Every file found is read (any\n"
            "extension; a file with no usable data rows is skipped with a one-line note,\n"
            "not treated as an error), expected to look like:\n"
            "    # OGLE-III/OGLE-IV detection efficiency in the field LMC501\n"
            "    # (for sources brighter than I = 22 mag)\n"
            "    # logtE_min logtE_max efficiency\n"
            "    0.00 0.16 0.000245\n"
            "    0.16 0.32 0.001605\n"
            "    ...\n"
            "i.e. '#' lines are comments (skipped when reading data, but the FIRST one\n"
            "matching '... field <ID> ...' is used to identify which field the whole\n"
            "file belongs to (here, 'LMC501'); if no line matches that pattern, the\n"
            "filename without its extension is used instead, e.g. 'LMC501.txt' ->\n"
            "'LMC501'), and each remaining row is [logtE_min, logtE_max, efficiency] \n"
            "i.e. efficiency binned in log10(tE/day), not tE itself directly. Each bin\n"
            "is converted to one (tE, efficiency) point at that bin's own log-midpoint\n"
            "tE, then interpolated the same way every other efficiency curve in this\n"
            "tool is (so this is a smooth curve through bin centers, not a literal step\n"
            "function constant within each bin).",
            default=""
        )
        if os.path.isfile(_eff_dir) and zipfile.is_zipfile(_eff_dir):
            _extract_dir = _strip_archive_ext(_eff_dir) + "_extracted"
            with zipfile.ZipFile(_eff_dir) as _zf:
                _zf.extractall(_extract_dir)
            print(f"-> '{_eff_dir}' is a zip file; extracted it to '{_extract_dir}'.")
            _eff_dir = _extract_dir
        elif os.path.isfile(_eff_dir) and tarfile.is_tarfile(_eff_dir):
            _extract_dir = _strip_archive_ext(_eff_dir) + "_extracted"
            with tarfile.open(_eff_dir) as _tf:
                _tf.extractall(_extract_dir)
            print(f"-> '{_eff_dir}' is a tar archive (.tar/.tar.gz/.tar.bz2/etc.); "
                  f"extracted it to '{_extract_dir}'.")
            _eff_dir = _extract_dir

        _eff_by_field, _eff_bounds_by_field = {}, {}
        _eff_files = sorted(glob.glob(os.path.join(_eff_dir, "**", "*"), recursive=True))
        _eff_files = [p for p in _eff_files if os.path.isfile(p)]
        for _path in _eff_files:
            _fid, _fn, _lo, _hi = _parse_one_field_efficiency_file(_path)
            if _fn is None:
                print(f"   [skipped] '{os.path.basename(_path)}': no usable data rows found.")
                continue
            _eff_by_field[_fid] = _fn
            _eff_bounds_by_field[_fid] = (_lo, _hi)
        print(f"-> Read {len(_eff_by_field)} field(s) from {len(_eff_files)} file(s) "
              f"in the efficiency folder{tag}.")
        return _eff_by_field, _eff_bounds_by_field

    if _split_baselines:
        print("\n--- Long-baseline efficiency (paired with the long-baseline star count "
              "and exposure) ---")
        _eff_by_field1, _eff_bounds1 = _read_eff_folder(" (long baseline)")
        print("\n--- Short-baseline efficiency (paired with the short-baseline star "
              "count and exposure) ---")
        _eff_by_field2, _eff_bounds2 = _read_eff_folder(" (short baseline)")

        _ids1 = sorted(set(_nstar_table1) & set(_eff_by_field1))
        _ids2 = sorted(set(_nstar_table2) & set(_eff_by_field2))
        _unmatched1 = sorted(set(_nstar_table1) - set(_eff_by_field1))
        _unmatched2 = sorted(set(_nstar_table2) - set(_eff_by_field2))
        if _unmatched1:
            print(f"   [NOTE] {len(_unmatched1)} field(s) have a long-baseline star count but no "
                  f"matching long-baseline efficiency file, excluded: {_unmatched1[:10]}"
                  f"{' ...' if len(_unmatched1) > 10 else ''}")
        if _unmatched2:
            print(f"   [NOTE] {len(_unmatched2)} field(s) have a short-baseline star count but no "
                  f"matching short-baseline efficiency file, excluded: {_unmatched2[:10]}"
                  f"{' ...' if len(_unmatched2) > 10 else ''}")
        if not _ids1 and not _ids2:
            raise ValueError("No field IDs matched in EITHER population -- check that the star-count "
                             "file and both efficiency folders use the same field-ID convention.")

        # One FIELDS entry per (field, baseline); short-baseline star counts are rescaled by the
        # exposure ratio so that a single global Exposure (the long baseline) applies to both.
        _short_over_long = _exposure_short / _exposure_long
        FIELDS = (
            [dict(DS=_DS_shared, l_deg=_l_shared, b_deg=_b_shared, Nstar=_nstar_table1[_fid],
                  field_id=f"{_fid}_long") for _fid in _ids1]
            + [dict(DS=_DS_shared, l_deg=_l_shared, b_deg=_b_shared,
                    Nstar=_nstar_table2[_fid] * _short_over_long,
                    field_id=f"{_fid}_short") for _fid in _ids2]
        )
        FIELD_EFFICIENCIES = [_eff_by_field1[_fid] for _fid in _ids1] + [_eff_by_field2[_fid] for _fid in _ids2]
        tE_eff_min = min([_eff_bounds1[_fid][0] for _fid in _ids1] + [_eff_bounds2[_fid][0] for _fid in _ids2])
        tE_eff_max = max([_eff_bounds1[_fid][1] for _fid in _ids1] + [_eff_bounds2[_fid][1] for _fid in _ids2])
        _FIELD_EFFICIENCIES_PREBUILT = True
        _true_total_nstar = sum(_nstar_table1[f] for f in _ids1) + sum(_nstar_table2[f] for f in _ids2)
        print(f"-> {len(_ids1)} long-baseline + {len(_ids2)} short-baseline field-population "
              f"entries; true total Nstar = {_true_total_nstar:,.0f} (short-baseline entries "
              f"internally rescaled by {_short_over_long:.4f} = short/long exposure ratio, so "
              f"the single global Exposure below, set to the long baseline, applies correctly "
              f"to both); combined efficiency tE range [{tE_eff_min:.3f}, {tE_eff_max:.3f}] days.")
    else:
        _eff_by_field, _eff_bounds_by_field = _read_eff_folder()

        _ids_nstar_only = sorted(set(_nstar_table) - set(_eff_by_field))
        _ids_eff_only = sorted(set(_eff_by_field) - set(_nstar_table))
        _ids_matched = sorted(set(_nstar_table) & set(_eff_by_field))
        if _ids_nstar_only:
            print(f"   [NOTE] {len(_ids_nstar_only)} field(s) have a star count but no matching "
                  f"efficiency file, excluded: {_ids_nstar_only[:10]}"
                  f"{' ...' if len(_ids_nstar_only) > 10 else ''}")
        if _ids_eff_only:
            print(f"   [NOTE] {len(_ids_eff_only)} field(s) have an efficiency file but no matching "
                  f"star count, excluded: {_ids_eff_only[:10]}"
                  f"{' ...' if len(_ids_eff_only) > 10 else ''}")
        if not _ids_matched:
            raise ValueError("No field IDs matched between the star-count file and the efficiency "
                             "folder -- check that both use the same field-ID convention.")

        FIELDS = [dict(DS=_DS_shared, l_deg=_l_shared, b_deg=_b_shared, Nstar=_nstar_table[_fid],
                       field_id=_fid)
                  for _fid in _ids_matched]
        FIELD_EFFICIENCIES = [_eff_by_field[_fid] for _fid in _ids_matched]
        tE_eff_min = min(_eff_bounds_by_field[_fid][0] for _fid in _ids_matched)
        tE_eff_max = max(_eff_bounds_by_field[_fid][1] for _fid in _ids_matched)
        _FIELD_EFFICIENCIES_PREBUILT = True
        print(f"-> {len(FIELDS)} matched field(s), total Nstar = "
              f"{sum(_f['Nstar'] for _f in FIELDS):,.0f}, combined efficiency tE range "
              f"[{tE_eff_min:.3f}, {tE_eff_max:.3f}] days.")

if FIELD_DATA_MODE == "per_field_bulk" and _split_baselines:
    Exposure = _exposure_long
    print(f"\n-> Exposure = {Exposure} days (the long baseline given above; the "
          f"short-baseline population's star count was already rescaled to match).")
else:
    Exposure = prompt_float(
        "\nSurvey exposure: total number of days the experiment was taking data\n"
        "(shared by every field above).\n"
        "(Default: the OGLE-IV baseline, 2010 June 29 to 2020 March 15 = 3546 days.)",
        default=3546.0, min_val=1e-6
    )

if len(FIELDS) <= 12:
    print(f"\n-> {len(FIELDS)} field(s):")
    for _k, _f in enumerate(FIELDS):
        print(f"     [{_k+1}] DS={_f['DS']:g} kpc, (l,b)=({_f['l_deg']:g},{_f['b_deg']:g}) deg, "
              f"Nstar={_f['Nstar']:.0f}")
else:
    _nstars_all = [_f['Nstar'] for _f in FIELDS]
    print(f"\n-> {len(FIELDS)} fields (too many to list individually): "
          f"DS={FIELDS[0]['DS']:g} kpc, (l,b)=({FIELDS[0]['l_deg']:g},{FIELDS[0]['b_deg']:g}) deg "
          f"(shared by all); Nstar per field ranges {min(_nstars_all):,.0f} to "
          f"{max(_nstars_all):,.0f} (total {sum(_nstars_all):,.0f}).")
print(f"   Exposure = {Exposure} days (shared by every field)")


# =============================================================================
# Halo density profiles and velocity distributions (lens & source)
# =============================================================================

def _prompt_density_profile(label, is_source, default_shape, default_rho0, default_rscale,
                             default_v0, default_l=None, default_b=None, allow_contracted=False):
    print(f"\n--- {label} halo ---")
    _valid_shapes = ("cored", "cuspy", "nfw", "custom") + (("contracted_nfw",) if allow_contracted else ())
    _contracted_line = (
        "'contracted_nfw' is the full Cautun et al. (2020) model: the NFW halo above,\n"
        "adiabatically contracted by the Galaxy's baryons (bulge, discs, gas, CGM). It is\n"
        "denser than the uncontracted NFW, i.e. a less conservative choice than 'nfw'.\n"
    ) if allow_contracted else ""
    _contracted_name_fragment = ", 'contracted_nfw'," if allow_contracted else ""
    shape = prompt_str(
        f"Density profile shape for the {label} halo: 'cored', 'cuspy', 'nfw'"
        f"{_contracted_name_fragment} or 'custom'.\n"
        f"'nfw' is rho(r) = rho_s / [(r/r_s)(1+r/r_s)^2], the standard Navarro-Frenk-White\n"
        f"form e.g. Cautun et al. (2020)'s own (pre-contraction) best fit to the Milky\n"
        f"Way's halo (which Mroz et al. 2024 use for the LMC analysis) is this shape, not a\n"
        f"cored isothermal sphere.\n" + _contracted_line,
        default=default_shape
    ).strip().lower()

    if shape not in _valid_shapes:
        print(f"  '{shape}' not recognized; falling back to '{default_shape}'.")
        shape = default_shape

    params = {}
    if shape == "contracted_nfw":
        params['M200_DM_Msun'] = prompt_float(
            "M200 of the dark-matter-only halo, in Msun (Cautun et al. 2020's own "
            "best fit: 0.97e12).", default=0.97e12, min_val=1e9)
        params['c_nfw'] = prompt_float(
            "NFW concentration c (Cautun et al. 2020's own best fit: 9.4).",
            default=9.4, min_val=0.5)
        params['velocity'] = prompt_str(
            "Halo velocity distribution (Maxwellian with velocity scale v0 = sqrt(2) x 1D dispersion):\n"
            "  'circular' = v0(r) = V_c(r), the model's own circular velocity (as in Mroz et al. 2024)\n"
            "  'jeans'    = v0(r) = sqrt(2) sigma(r), from the isotropic Jeans equation in the\n"
            "               model's total (DM + baryon) potential\n"
            "  'constant' = a single v0 that you enter",
            default="circular").strip().lower()
        while params['velocity'] not in ("circular", "jeans", "constant"):
            params['velocity'] = prompt_str("  Please enter circular, jeans or constant.",
                                            default="circular").strip().lower()
        v0_kms = None
        if params['velocity'] == "constant":
            v0_kms = prompt_float(
                f"Velocity scale v0 of the {label} halo's Maxwellian, in km/s (v0 = sqrt(2) x 1D dispersion,\n"
                f"i.e. the circular speed of an isothermal halo).",
                default=default_v0, min_val=1e-9
            )
        l_deg = b_deg = None
        if not is_source:
            l_deg = prompt_float(
                "Galactic longitude l of the line of sight to the source, in degrees.\n"
                "(Default is the LMC's direction.)", default=default_l
            )
            b_deg = prompt_float(
                "Galactic latitude b of the line of sight to the source, in degrees.",
                default=default_b
            )
        return shape, params, v0_kms, l_deg, b_deg
    if shape == "custom":
        expr = prompt_str(
            "Enter rho(x) as a one-line NumPy expression in x (Msun/kpc^3), "
            "e.g. 1.0e7/(1.0+(50*(1-x)/12.0)**2)",
        )
        def _custom_func(x, _expr=expr):
            x = np.atleast_1d(np.asarray(x, dtype=float))
            try:
                val = eval(_expr, {"np": np, "__builtins__": {}}, {"x": x})
                val = np.asarray(val, dtype=float)
                val = np.where(np.isfinite(val) & (val >= 0), val, 0.0)
                return val
            except Exception as e:
                print(f"  [warning] custom density function raised {e} at some x; "
                      f"returning 0 there.")
                return np.zeros_like(x)
        params['func'] = _custom_func
    else:
        _rho0_label = "characteristic density rho_s" if shape == "nfw" else "central density rho0"
        params['rho0_GeVcm3'] = prompt_float(
            f"{_rho0_label} for the {label} halo, in GeV/cm^3.", default=default_rho0, min_val=0.0)
        params['r_scale_kpc'] = prompt_float(
            f"Scale radius for the {label} halo, in kpc"
            f"{' (= R_s = R_200/concentration for NFW)' if shape == 'nfw' else ''}.",
            default=default_rscale, min_val=1e-9)

    v0_kms = prompt_float(
        f"Velocity scale v0 of the {label} halo's Maxwellian, in km/s (v0 = sqrt(2) x 1D dispersion,\n"
        f"i.e. the circular speed of an isothermal halo).",
        default=default_v0, min_val=1e-9
    )

    l_deg = b_deg = None
    if not is_source:
        l_deg = prompt_float(
            "Galactic longitude l of the line of sight to the source, in degrees.\n"
            "(Default is the LMC's direction.)", default=default_l
        )
        b_deg = prompt_float(
            "Galactic latitude b of the line of sight to the source, in degrees.",
            default=default_b
        )

    return shape, params, v0_kms, l_deg, b_deg

_MW_MODEL_DEFAULTS = {
    "S": dict(rho0_msun_pc3=0.008, r_c_kpc=5.0, v_c_kms=220.0),
    "B": dict(beta=-0.2, v_a_kms=200.0, r_c_kpc=5.0),
    "C": dict(beta=+0.2, v_a_kms=180.0, r_c_kpc=5.0),
}

def _parse_mw_model_list(raw):
    """Parse a Milky-Way model selection such as 'S, B, C' into a list of letters (None if invalid)."""
    toks = [t for t in re.split(r"[\s,;\[\](){}'\"]+", raw.strip().upper()) if t]
    if not toks or any(t not in ("S", "B", "C", "U") for t in toks):
        return None
    return toks

def _prompt_mw_model_parameters(choice, k, n):
    """Prompt for the parameters of one Milky-Way halo model; returns a spec dict."""
    if n > 1:
        print(f"\n--- Parameters for Milky-Way model {choice}  (entry {k} of {n}) ---")

    if choice == "U":
        shape, params, v0_kms, _, _ = _prompt_density_profile(
            "Milky Way (foreground lens), model U", is_source=True,
            default_shape="nfw", default_rho0=0.1513, default_rscale=23.8, default_v0=220.0,
            allow_contracted=True
        )
        return dict(model="U", shape=shape, params=params, v0_kms=v0_kms)

    d = dict(_MW_MODEL_DEFAULTS[choice])
    use_defaults = prompt_yes_no(
        f"\nUse the paper's default parameter values for model {choice}?\n"
        f"(Answer n to enter every parameter yourself, one by one.)", default=True
    )
    if not use_defaults:
        if choice == "S":
            d["rho0_msun_pc3"] = prompt_float(
                "Local dark-matter density at the Sun, rho0, in Msun/pc^3 "
                "(0.008 Msun/pc^3 = 0.304 GeV/cm^3).",
                default=d["rho0_msun_pc3"], min_val=1e-12)
            d["r_c_kpc"] = prompt_float(
                "Core radius rc, in kpc.", default=d["r_c_kpc"], min_val=1e-6)
            d["v_c_kms"] = prompt_float(
                "Circular speed v_c (Maxwellian parameter, same at every radius), in km/s.",
                default=d["v_c_kms"], min_val=1e-6)
        else:
            d["beta"] = prompt_float(
                "Rotation-curve slope beta (v_c ~ r^-beta far outside the core: beta < 0 "
                "rising, 0 flat, > 0 falling); must lie in (-2, 1).",
                default=d["beta"], min_val=-1.99, max_val=0.99)
            d["v_a_kms"] = prompt_float(
                "Normalisation velocity v_a, in km/s.", default=d["v_a_kms"], min_val=1e-6)
            d["r_c_kpc"] = prompt_float(
                "Core radius rc, in kpc.", default=d["r_c_kpc"], min_val=1e-6)
    return dict(model=choice, **d)

def _prompt_mw_halo_models():
    """Prompt for the Milky-Way halo model(s); returns a list of spec dicts."""
    print("\n--- Milky Way (foreground lens) halo ---")
    print("""
Choose the Milky-Way halo model (models S, B, C of Green 2017, arXiv:1705.10818):
  S : standard halo: cored isothermal sphere, rho(r) = rho0 (rc^2+r0^2)/(rc^2+r^2),
      isotropic Maxwellian velocities with the SAME circular speed v_c at all radii.
      Parameters: rho0 (local density), rc (core radius), v_c.
      Defaults: rho0 = 0.008 Msun/pc^3, rc = 5 kpc, v_c = 220 km/s.
  B : spherical Evans power-law halo, massive (M(<50 kpc) ~ 7e11 Msun), rising rotation
      curve. The lens velocity distribution varies with radius (it is derived from the
      halo's exact distribution function, not assumed to be a Maxwellian).
      Parameters: beta, v_a, rc.  Defaults: beta = -0.2, v_a = 200 km/s, rc = 5 kpc.
  C : spherical Evans power-law halo, light (M(<50 kpc) ~ 2e11 Msun), falling rotation
      curve; radius-dependent velocity distribution as for B.
      Defaults: beta = +0.2, v_a = 180 km/s, rc = 5 kpc.
  U : user-defined density (built-in shape, contracted NFW, or your own expression)
      with a single constant Maxwellian v0.

You can enter a single model (e.g.  S ) or a list of models (e.g.  [S, B, C] ) to draw
one curve per model in the same plot: curves of different models get different LINE
STYLES (first model solid, then dashed, dash-dot, dotted, ...), while curves of
different R90 values get different COLOURS. A letter may be repeated (e.g. [S, S])
to compare two parameter sets of the same model; each entry asks for its own
parameters. Computing time grows with (number of models) x (number of R90 values).
""")
    while True:
        raw = prompt_str("Milky-Way halo model(s): S, B, C, U, or a list such as [S, B, C].",
                         default="S")
        models = _parse_mw_model_list(raw)
        if models is not None:
            break
        print("  Could not read that. Use the letters S, B, C and/or U, e.g.  S  or  [S, B, C].")

    specs = [_prompt_mw_model_parameters(m, k + 1, len(models)) for k, m in enumerate(models)]

    for i in range(len(specs)):
        for j in range(i):
            if specs[i] == specs[j]:
                print(f"  [note] entries {j+1} and {i+1} are identical (same model and parameters); "
                      f"their curves will lie exactly on top of each other.")
    return specs

_CONTRACTED_V0_LABEL = {"circular": "v0(r) = V_c(r)", "jeans": "v0(r) = sqrt(2) sigma_Jeans(r)"}

def _describe_mw_spec(spec):
    m = spec["model"]
    if m == "S":
        return (f"model S (cored isothermal, constant Maxwellian): rho0={spec['rho0_msun_pc3']} "
                f"Msun/pc^3 (local), rc={spec['r_c_kpc']} kpc, v_c={spec['v_c_kms']} km/s")
    if m in ("B", "C"):
        return (f"model {m} (Evans power law, radius-dependent velocities): beta={spec['beta']}, "
                f"v_a={spec['v_a_kms']} km/s, rc={spec['r_c_kpc']} kpc")
    if spec["v0_kms"] is None:
        return f"model U (contracted NFW density, Maxwellian {_CONTRACTED_V0_LABEL[spec['params']['velocity']]})"
    return f"model U (user-defined {spec['shape']} density, constant Maxwellian v0={spec['v0_kms']} km/s)"

def _mw_legend_label(spec):
    """Legend label (mathtext) for one Milky-Way model."""
    m = spec["model"]
    if m == "S":
        return (rf"S: $\rho_0$={spec['rho0_msun_pc3']:g} $M_\odot$/pc$^3$, "
                rf"$r_c$={spec['r_c_kpc']:g} kpc, $v_c$={spec['v_c_kms']:g} km/s")
    if m in ("B", "C"):
        return (rf"{m}: $\beta$={spec['beta']:g}, $v_a$={spec['v_a_kms']:g} km/s, "
                rf"$r_c$={spec['r_c_kpc']:g} kpc")
    if spec["v0_kms"] is None:
        return {"circular": r"U: contracted NFW, $v_0=V_c(r)$",
                "jeans": r"U: contracted NFW, $v_0=\sqrt{2}\,\sigma_{\rm J}(r)$"}[spec["params"]["velocity"]]
    return rf"U: {spec['shape']} $\rho(r)$, $v_0$={spec['v0_kms']:g} km/s"

def _make_mw_labels(specs):
    """Unique (plain, legend) labels for the list of Milky-Way models."""
    letters = [s["model"] for s in specs]
    plain, seen = [], {}
    for L in letters:
        seen[L] = seen.get(L, 0) + 1
        plain.append(L if letters.count(L) == 1 else f"{L}#{seen[L]}")
    raw = [_mw_legend_label(s) for s in specs]
    legend, seen = [], {}
    for lab in raw:
        seen[lab] = seen.get(lab, 0) + 1
        legend.append(lab if raw.count(lab) == 1 else f"{lab}  (#{seen[lab]})")
    return plain, legend


print("\nSTEP 3 of 6: Halo density profiles and velocities")
print("=" * 70)
print("\nExample (OGLE-IV/LMC defaults) shown at each prompt; press Enter to accept.")

mw_specs = _prompt_mw_halo_models()
mw_plain_labels, mw_legend_labels = _make_mw_labels(mw_specs)

RSol_kpc = prompt_float(
    "\nSun -> Galactic-Center distance, in kpc. (For Milky-Way model S, the local density rho0\n"
    "entered above is the density at exactly this distance.)", default=8.5, min_val=1e-6
)

INCLUDE_SOURCE_HALO = prompt_yes_no(
    "\nInclude source (e.g. LMC) self-lensing from its own dark matter halo, in\n"
    "addition to Milky Way lenses? Both the EROS-2 and OGLE-IV analyses in Croon,\n"
    "McKeen & Raj (2020) use Milky Way lenses only (answer 'n' to match that paper).",
    default=True
)
if INCLUDE_SOURCE_HALO:
    source_shape, source_params, v0_source_kms, _, _ = _prompt_density_profile(
        "Source (e.g. LMC) self-lensing", is_source=True,
        default_shape="cuspy", default_rho0=0.180, default_rscale=17.1, default_v0=92.0
    )
    v0_source_kpcyr = v0_source_kms * KMS_TO_KPCYR
else:
    source_shape = source_params = v0_source_kms = v0_source_kpcyr = None
    print("\n-> Source self-lensing disabled: only Milky Way lenses contribute to the rate.")

print(f"\n-> Milky Way halo model(s): {len(mw_specs)}")
for _k, _spec in enumerate(mw_specs):
    print(f"     [{_k+1}] {_describe_mw_spec(_spec)}")
print(f"   RSol={RSol_kpc} kpc  (line of sight and distance are taken from the Step-2 field table)")
if INCLUDE_SOURCE_HALO:
    print(f"-> Source:    {source_shape} profile, v0={v0_source_kms} km/s")
else:
    print("-> Source:    disabled (Milky Way lenses only)")


# =============================================================================
# Build the halo density functions rho(x) from the above user inputs
# =============================================================================

# Contracted Milky-Way halo (Cautun et al. 2020): baryonic model (bulge, thin/thick
# discs, HI, H2, CGM) and the contraction relation (their Eq. 11)
#   M_DM(<r) = M_DMO(<r) [0.45 + 0.38 (eta_bar + 1.16)^0.53],
#   eta_bar = M_bar(<r) / (f_bar M_tot,DMO(<r)).
# R200 of the DM-only halo follows from M200_DM and the cosmology; R200 of the total
# (contracted DM + baryons) halo, which normalises the CGM (their Eqs. 5-6), is solved
# self-consistently. Surface densities are in Msun/pc^2 and converted once, inside each
# density function.
_CAUTUN_BULGE = dict(r0=0.075, rcut=2.1, alpha=1.8, q=0.5, rho0_msun_pc3=103.0)
_CAUTUN_THIN = dict(zd=0.300, Sigma0_msun_pc2=731.0, Rd=2.63)
_CAUTUN_THICK = dict(zd=0.900, Sigma0_msun_pc2=101.0, Rd=3.80)
_CAUTUN_HI = dict(zd=0.085, Rm=4.0, Rd=7.0, Sigma0_msun_pc2=53.0)
_CAUTUN_H2 = dict(zd=0.045, Rm=12.0, Rd=1.5, Sigma0_msun_pc2=2200.0)
_CAUTUN_CGM = dict(A=0.190, beta=-1.46)
_CAUTUN_FBAR = 0.048 / 0.307          # Omega_b/Omega_m, Planck 2014 (Cautun et al. 2020, Sec. 3.1)
_CAUTUN_H = 0.6777                    # Hubble parameter h of the same cosmology
_CAUTUN_RHO_CRIT = 3.0 * (0.1 * _CAUTUN_H)**2 / (8.0 * np.pi * G_KPC)   # Msun/kpc^3

def _cautun_rho_bulge(R, z):
    p = _CAUTUN_BULGE
    rp = np.sqrt(R**2 + (z / p['q'])**2)
    rho0 = p['rho0_msun_pc3'] * 1.0e9  # Msun/pc^3 -> Msun/kpc^3
    return rho0 / (1.0 + rp / p['r0'])**p['alpha'] * np.exp(-(rp / p['rcut'])**2)

def _cautun_rho_disc(R, z, p):
    Sigma0 = p['Sigma0_msun_pc2'] * 1.0e6  # Msun/pc^2 -> Msun/kpc^2
    return Sigma0 / (2.0 * p['zd']) * np.exp(-np.abs(z) / p['zd'] - R / p['Rd'])

def _cautun_rho_gas(R, z, p):
    R = np.maximum(R, 1e-6)
    Sigma0 = p['Sigma0_msun_pc2'] * 1.0e6
    x = np.clip(np.abs(z) / (2.0 * p['zd']), 0, 350)
    sech2 = 4.0 / (np.exp(x) + np.exp(-x))**2  # avoids cosh() overflow at large |z|/zd
    return Sigma0 / (4.0 * p['zd']) * np.exp(-p['Rm'] / R - R / p['Rd']) * sech2

def _enclosed_mass_axisymmetric(rho_func, r_max, n_R=60, n_z=60):
    """M(<r_max) [Msun] of an axisymmetric rho(R, z) [Msun/kpc^3], by 2D Gauss-Legendre quadrature."""
    gR, wR = np.polynomial.legendre.leggauss(n_R)
    R_nodes = 0.5 * r_max * (gR + 1.0)
    R_weights = 0.5 * r_max * wR
    total = 0.0
    for R, wR_i in zip(R_nodes, R_weights):
        z_max = np.sqrt(max(r_max**2 - R**2, 0.0))
        if z_max <= 0:
            continue
        gz, wz = np.polynomial.legendre.leggauss(n_z)
        z_nodes = 0.5 * z_max * (gz + 1.0)
        z_weights = 0.5 * z_max * wz
        integrand = rho_func(R, z_nodes) * 2.0 * np.pi * R
        total += wR_i * np.sum(z_weights * integrand) * 2.0
    return total

def _cautun_M_bar_enclosed(r):
    """Enclosed stellar + gas mass [Msun] of the Cautun et al. (2020) baryonic model (CGM excluded)."""
    if r <= 0:
        return 0.0
    r = min(r, 100.0)   # all stars and cold gas lie within 100 kpc; larger r only adds quadrature error
    return (_enclosed_mass_axisymmetric(_cautun_rho_bulge, r, 60, 60)
            + _enclosed_mass_axisymmetric(lambda R, z: _cautun_rho_disc(R, z, _CAUTUN_THIN), r, 60, 60)
            + _enclosed_mass_axisymmetric(lambda R, z: _cautun_rho_disc(R, z, _CAUTUN_THICK), r, 60, 60)
            + _enclosed_mass_axisymmetric(lambda R, z: _cautun_rho_gas(R, z, _CAUTUN_HI), r, 80, 60)
            + _enclosed_mass_axisymmetric(lambda R, z: _cautun_rho_gas(R, z, _CAUTUN_H2), r, 80, 60))

# The table takes ~20-40 s to build and make_density_function() is called once per field,
# so it is cached per (M200, c, grid).
_CONTRACTED_TABLE_CACHE = {}

def build_contracted_mw_density_table(M200_DM, c_nfw, n_table=150, r_lo=0.05, r_hi=300.0,
                                      progress=True):
    """Contracted Milky-Way halo (Cautun et al. 2020) tabulated in r; cached per (M200, c, grid)."""
    _cache_key = (float(M200_DM), float(c_nfw), int(n_table), float(r_lo), float(r_hi))
    if _cache_key in _CONTRACTED_TABLE_CACHE:
        return _CONTRACTED_TABLE_CACHE[_cache_key]
    fb, rho200 = _CAUTUN_FBAR, 200.0 * _CAUTUN_RHO_CRIT
    # DM-only NFW halo: its total mass M200_DM/(1 - f_bar) has mean density 200 rho_crit
    # within R200_DMO; the concentration refers to this pre-contraction halo, r_s = R200_DMO / c.
    R200_DMO = (M200_DM / (1.0 - fb) / (4.0 / 3.0 * np.pi * rho200))**(1.0 / 3.0)
    r_s = R200_DMO / c_nfw

    def M_DM_DMO(r):
        x = r / r_s
        return M200_DM * (np.log(1.0 + x) - x / (1.0 + x)) / (np.log(1.0 + c_nfw) - c_nfw / (1.0 + c_nfw))

    def M_CGM_enclosed(r, R200):
        # Eq. 6, with M200_tot = (4 pi / 3) 200 rho_crit R200^3
        A, beta = _CAUTUN_CGM['A'], _CAUTUN_CGM['beta']
        M200_tot = 4.0 / 3.0 * np.pi * rho200 * R200**3
        return (3.0 * A / (beta + 3.0)) * fb * M200_tot * (r / R200)**(beta + 3.0)

    def M_DM_contracted(r, M_bar):
        eta_bar = M_bar / (fb * M_DM_DMO(r) / (1.0 - fb))
        return M_DM_DMO(r) * (0.45 + 0.38 * (eta_bar + 1.16)**0.53)

    def M_total(r, R200):
        M_bar = _cautun_M_bar_enclosed(r) + M_CGM_enclosed(r, R200)
        return M_DM_contracted(r, M_bar) + M_bar, M_bar

    if progress:
        print(f"  Building the contracted Milky Way halo mass table "
              f"({n_table} points, M200={M200_DM:.3e} Msun, c={c_nfw:.2f}; ~20-40 s, done once)...")
    # Total-halo R200: mean density of (contracted DM + baryons) within R200 is 200 rho_crit
    R200 = brentq(lambda R: M_total(R, R)[0] - 4.0 / 3.0 * np.pi * rho200 * R**3,
                  0.5 * R200_DMO, 2.0 * R200_DMO, xtol=1e-3)

    r_table = np.logspace(np.log10(r_lo), np.log10(r_hi), n_table)
    M_DM_table, M_tot_table = np.empty(n_table), np.empty(n_table)
    for i, r in enumerate(r_table):
        M_tot_table[i], M_bar = M_total(r, R200)
        M_DM_table[i] = M_tot_table[i] - M_bar
    result = dict(spline=CubicSpline(r_table, M_DM_table), r=r_table, M_tot=M_tot_table,
                  R200=R200, M200_tot=4.0 / 3.0 * np.pi * rho200 * R200**3, R200_DMO=R200_DMO)
    if progress:
        print(f"  ...done: R200 = {R200:.1f} kpc, M200(total) = {result['M200_tot']:.3e} Msun.")
    _CONTRACTED_TABLE_CACHE[_cache_key] = result
    return result


# Velocity scale v0(r) of the contracted halo's Maxwellian (v0 = sqrt(2) x 1D dispersion):
# 'circular' uses the model's circular velocity V_c(r) = sqrt(G M_tot(<r) / r), as in
# Mroz et al. (2024); 'jeans' solves the isotropic Jeans equation in the total potential,
#   sigma^2(r) = (1 / rho_DM) int_r^inf rho_DM G M_tot(<r') / r'^2 dr'.
_CONTRACTED_V0_CACHE = {}

def contracted_v0_table(M200_DM, c_nfw, mode):
    """(r [kpc], v0 [km/s]) on the contracted-halo grid for mode 'circular' or 'jeans'."""
    key = (float(M200_DM), float(c_nfw), mode)
    if key in _CONTRACTED_V0_CACHE:
        return _CONTRACTED_V0_CACHE[key]
    t = build_contracted_mw_density_table(M200_DM, c_nfw)
    r, M_tot = t['r'], t['M_tot']
    if mode == "circular":
        v0 = np.sqrt(G_KPC * M_tot / r)
    else:
        rho = t['spline'](r, 1) / (4.0 * np.pi * r**2)
        f = rho * G_KPC * M_tot / r**2 * r                     # integrand per d(ln r)
        seg = 0.5 * (f[1:] + f[:-1]) * np.diff(np.log(r))
        integral = np.concatenate([np.cumsum(seg[::-1])[::-1], [0.0]])
        integral += rho[-1] * G_KPC * M_tot[-1] / (4.0 * r[-1])   # tail beyond r_hi (rho ~ r^-3)
        v0 = np.sqrt(2.0 * integral / rho)
    _CONTRACTED_V0_CACHE[key] = (r, v0)
    return r, v0


def make_density_function(kind, params, DS, RSol=None, l_deg=None, b_deg=None, is_source=False):
    """Return rho(x) [Msun/kpc^3] along the line of sight, x = D_L/D_S in (0, 1)."""
    if kind == "custom":
        return params['func']

    # Radius at which the halo density is evaluated: galactocentric radius along the line
    # of sight for the Milky Way, distance from the source centre for the source halo.
    if not is_source:
        cos_val = np.cos(np.radians(l_deg)) * np.cos(np.radians(b_deg))
        def rKpc_of_x(x):
            x = np.asarray(x, dtype=float)
            return np.sqrt(np.clip(RSol**2 - 2.0*x*RSol*DS*cos_val + (x*DS)**2, 1e-9, None))
    else:
        def rKpc_of_x(x):
            x = np.asarray(x, dtype=float)
            return np.maximum((1.0 - x) * DS, 1e-3)

    if kind == "contracted_nfw":
        if is_source:
            raise ValueError("'contracted_nfw' is a Milky-Way-specific model "
                             "(it is not meaningful for a source's own halo).")
        r_table_min, r_table_max = 0.05, 300.0
        spline = build_contracted_mw_density_table(
            params['M200_DM_Msun'], params['c_nfw'], n_table=150,
            r_lo=r_table_min, r_hi=r_table_max)['spline']
        def rho(x):
            r = rKpc_of_x(x)
            r_c = np.clip(r, r_table_min, r_table_max)
            dMdr = spline(r_c, 1)  # rho = (dM/dr) / (4 pi r^2)
            return np.maximum(dMdr / (4.0 * np.pi * r_c**2), 0.0)
        return rho

    rho0 = params['rho0_GeVcm3'] * GEV_CM3_TO_MSUN_KPC3
    rscale = params['r_scale_kpc']

    if kind == "cored":
        def rho(x):
            r = rKpc_of_x(x)
            return rho0 / (1.0 + (r / rscale)**2)
    elif kind == "cuspy":
        def rho(x):
            r = rKpc_of_x(x)
            return rho0 / ((r / rscale) * (1.0 + r / rscale)**3)
    elif kind == "nfw":
        def rho(x):
            r = rKpc_of_x(x)
            return rho0 / ((r / rscale) * (1.0 + r / rscale)**2)
    else:
        raise ValueError(f"Unknown density profile kind: {kind}")
    return rho

# =============================================================================
# Lens "populations": density AND velocity distribution of the lenses
# =============================================================================
# Each population provides density(x) and G_perp(x, v), the density-weighted
# distribution of the lens velocity transverse to the line of sight, which enters
# the rate integral d2GammaDxdtE().

def _los_radius_function(DS, RSol, l_deg, b_deg):
    """Galactocentric radius r(x) [kpc] along the line of sight to (DS, l, b)."""
    cos_val = np.cos(np.radians(l_deg)) * np.cos(np.radians(b_deg))
    def r_of_x(x):
        x = np.asarray(x, dtype=float)
        return np.sqrt(np.clip(RSol**2 - 2.0*x*RSol*DS*cos_val + (x*DS)**2, 1e-9, None))
    return r_of_x


class MaxwellianPopulation:
    """Lens population: density rho(x) with an isotropic Maxwellian velocity distribution (parameter v0)."""

    def __init__(self, rho_func, v0):
        self._rho = rho_func     # callable: array x -> density [Msun/kpc^3]
        self._v0 = v0            # float [kpc/yr] or callable v0(x) [kpc/yr]

    def v0_of_x(self, x):
        x = np.asarray(x, dtype=float)
        return np.asarray(self._v0(x), dtype=float) if callable(self._v0) \
               else np.full_like(x, float(self._v0))

    def density(self, x):
        x = np.atleast_1d(np.asarray(x, dtype=float))
        rho = np.asarray(self._rho(x), dtype=float)
        return np.where(np.isfinite(rho) & (rho >= 0.0), rho, 0.0)

    def G_perp(self, x, v):
        x = np.atleast_1d(np.asarray(x, dtype=float))
        v = np.asarray(v, dtype=float)
        v0 = self.v0_of_x(x)
        return self.density(x) * np.exp(-np.clip((v / v0)**2, 0.0, 700.0)) / (np.pi * v0**2)


class EvansPowerLawPopulation:
    """Evans (1993, 1994) power-law halo (models B and C of Alcock et al. 1995)."""

    def __init__(self, beta, v_a_kms, r_c_kpc, DS, RSol, l_deg, b_deg):
        if not (-2.0 < beta < 1.0):
            raise ValueError("beta must lie in (-2, 1) for a positive-definite density and distribution function.")
        if v_a_kms <= 0 or r_c_kpc <= 0:
            raise ValueError("v_a and rc must be positive.")
        self.beta, self.v_a, self.r_c = float(beta), float(v_a_kms), float(r_c_kpc)
        self._isothermal = abs(self.beta) < 1e-6
        self._r_of_x = _los_radius_function(DS, RSol, l_deg, b_deg)
        self._pref = self.v_a**2 / (4.0 * np.pi * G_KPC)       # Msun / kpc

    def _reff2(self, r):
        return self.r_c**2 + np.asarray(r, dtype=float)**2

    def potential(self, r):
        """Psi(r) [(km/s)^2] (beta != 0 only)."""
        return (self.v_a**2 * self.r_c**self.beta / self.beta) * self._reff2(r)**(-self.beta/2.0)

    def circular_speed(self, r):
        """v_c(r) [km/s]."""
        r = np.asarray(r, dtype=float)
        return np.sqrt(self.v_a**2 * self.r_c**self.beta * r**2 / self._reff2(r)**((self.beta + 2.0)/2.0))

    def mass_enclosed(self, r):
        """M(<r) [Msun] = r v_c^2(r) / G."""
        r = np.asarray(r, dtype=float)
        return r * self.circular_speed(r)**2 / G_KPC

    def _rho_components(self, r):
        b = self.beta; re2 = self._reff2(r)
        rho2 = (1.0 - b) * self._pref * self.r_c**b          * re2**(-(b + 2.0)/2.0)
        rho1 = (2.0 + b) * self._pref * self.r_c**(b + 2.0)  * re2**(-(b + 4.0)/2.0)
        return rho1, rho2

    def density_r(self, r):
        rho1, rho2 = self._rho_components(r)
        return rho1 + rho2

    def density(self, x):
        return self.density_r(self._r_of_x(np.atleast_1d(np.asarray(x, dtype=float))))

    def G_perp(self, x, v):
        # Transverse-velocity distribution derived from the Evans distribution function,
        # as a sum over its two density components.
        x = np.atleast_1d(np.asarray(x, dtype=float))
        v_kms = np.asarray(v, dtype=float) / KMS_TO_KPCYR
        r = self._r_of_x(x)
        rho1, rho2 = self._rho_components(r)
        if self._isothermal:
            vm1sq, vm2sq = self.v_a**2 / 2.0, self.v_a**2
            G = (rho1 * np.exp(-np.clip(v_kms**2 / vm1sq, 0.0, 700.0)) / (np.pi * vm1sq)
                 + rho2 * np.exp(-np.clip(v_kms**2 / vm2sq, 0.0, 700.0)) / (np.pi * vm2sq))
        else:
            b = self.beta
            Psi = self.potential(r)
            arg = 1.0 - v_kms**2 / (2.0 * Psi)
            inside = arg > 0.0                         # beta > 0: v_perp below the escape speed
            log_arg = np.log(np.where(inside, arg, 1.0))
            G = np.zeros_like(r)
            for rho_c, kappa in ((rho1, 4.0/b + 1.0), (rho2, 2.0/b + 1.0)):
                g = kappa / (2.0 * np.pi * Psi) * np.exp((kappa - 1.0) * log_arg)
                G = G + rho_c * np.where(inside, g, 0.0)
        return G / KMS_TO_KPCYR**2                     # per (km/s)^2  ->  per (kpc/yr)^2


def build_lens_population(spec, DS, RSol, l_deg, b_deg):
    """Build the Milky-Way lens population for one spec along one field's line of sight."""
    m = spec["model"]
    if m == "S":
        r_of_x = _los_radius_function(DS, RSol, l_deg, b_deg)
        rho0 = spec["rho0_msun_pc3"] * 1e9          # Msun/pc^3 -> Msun/kpc^3
        rc = spec["r_c_kpc"]
        rho_S = lambda x: rho0 * (rc**2 + RSol**2) / (rc**2 + r_of_x(x)**2)
        return MaxwellianPopulation(rho_S, spec["v_c_kms"] * KMS_TO_KPCYR)
    if m in ("B", "C"):
        return EvansPowerLawPopulation(spec["beta"], spec["v_a_kms"], spec["r_c_kpc"],
                                        DS, RSol, l_deg, b_deg)
    rho_U = make_density_function(spec["shape"], spec["params"], DS, RSol=RSol,
                                   l_deg=l_deg, b_deg=b_deg, is_source=False)
    if spec["v0_kms"] is not None:
        return MaxwellianPopulation(rho_U, spec["v0_kms"] * KMS_TO_KPCYR)
    # contracted halo with a position-dependent v0(r), interpolated in log r along the sightline
    r_tab, v0_tab = contracted_v0_table(spec["params"]["M200_DM_Msun"], spec["params"]["c_nfw"],
                                        spec["params"]["velocity"])
    r_of_x = _los_radius_function(DS, RSol, l_deg, b_deg)
    log_r, log_v0 = np.log(r_tab), np.log(v0_tab)
    def v0_of_x(x):
        r = np.clip(r_of_x(x), r_tab[0], r_tab[-1])
        return np.exp(np.interp(np.log(r), log_r, log_v0)) * KMS_TO_KPCYR
    return MaxwellianPopulation(rho_U, v0_of_x)


def describe_lens_population(spec, pop, RSol, tag=""):
    """Print diagnostics (local density, rotation speed, enclosed mass) for a Milky-Way model."""
    m = spec["model"]
    if m == "U" and spec["shape"] == "contracted_nfw":
        p = spec["params"]
        t = build_contracted_mw_density_table(p['M200_DM_Msun'], p['c_nfw'])
        rho_sun = float(t['spline'](RSol, 1)) / (4.0 * np.pi * RSol**2) / GEV_CM3_TO_MSUN_KPC3
        v0_sun = float(pop.v0_of_x(np.array([0.0]))[0]) / KMS_TO_KPCYR
        print(f"  Milky-Way model U{tag}: contracted NFW (Cautun et al. 2020): R200 = {t['R200']:.1f} kpc, "
              f"M200(total) = {t['M200_tot']:.3e} Msun, local DM density = {rho_sun:.3f} GeV/cm^3 "
              f"at r = {RSol} kpc, v0 there = {v0_sun:.1f} km/s.")
        return
    if m == "U":
        print(f"  Milky-Way model U{tag}: user-defined density, constant Maxwellian "
              f"v0 = {spec['v0_kms']} km/s (no further summary available).")
        return
    if m == "S":
        rc, rho0 = spec["r_c_kpc"], spec["rho0_msun_pc3"] * 1e9
        vc_sun = vc_50 = spec["v_c_kms"]
        rho_sun = rho0
        M50 = 4*np.pi*rho0*(rc**2 + RSol**2) * (50.0 - rc*np.arctan(50.0/rc))
    else:
        vc_sun = float(pop.circular_speed(RSol)); vc_50 = float(pop.circular_speed(50.0))
        rho_sun = float(pop.density_r(RSol)); M50 = float(pop.mass_enclosed(50.0))
    print(f"  Milky-Way model {m}{tag}: rho(R_sun) = {rho_sun/1e9:.5f} Msun/pc^3 "
          f"= {rho_sun/GEV_CM3_TO_MSUN_KPC3:.4f} GeV/cm^3;  "
          f"v_c(R_sun) = {vc_sun:.1f} km/s,  v_c(50 kpc) = {vc_50:.1f} km/s;  "
          f"M(<50 kpc) = {M50:.2e} Msun")

# Fields sharing (DS, l, b) share the expensive line-of-sight integral; group them.
_FIELD_GEOM_GROUPS = {}
for _i, _f in enumerate(FIELDS):
    _key = (_f["DS"], _f["l_deg"], _f["b_deg"])
    _FIELD_GEOM_GROUPS.setdefault(_key, []).append(_i)
if len(_FIELD_GEOM_GROUPS) < len(FIELDS):
    print(f"-> Performance note: {len(FIELDS)} fields share only {len(_FIELD_GEOM_GROUPS)} distinct "
          f"(DS, l, b) geometries; the rate calculation will exploit this (see dGammaDtE) rather than "
          f"repeating the expensive part once per field.")

lens_populations_by_model = [
    [build_lens_population(_spec, _f["DS"], RSol_kpc, _f["l_deg"], _f["b_deg"]) for _f in FIELDS]
    for _spec in mw_specs
]
# One population per (Milky-Way model, field); dGammaDtE() sums over CURRENT_MODEL_FIELD_POPS,
# which the main loop reassigns for each model.
CURRENT_MODEL_FIELD_POPS = lens_populations_by_model[0]

# Source (self-lensing) halo, one population per field
if INCLUDE_SOURCE_HALO:
    source_populations_by_field = [
        MaxwellianPopulation(make_density_function(source_shape, source_params, _f["DS"], is_source=True),
                             v0_source_kpcyr)
        for _f in FIELDS
    ]
else:
    source_populations_by_field = None

for _k, _spec in enumerate(mw_specs):
    describe_lens_population(_spec, lens_populations_by_model[_k][0], RSol_kpc,
                             tag=(f" [{_k+1}/{len(mw_specs)}]" if len(mw_specs) > 1 else ""))

# Sanity check: every density is finite and non-negative along the line of sight
_x_check = np.linspace(0.001, 0.999, 25)
_checks = []
for _lab, _pops in zip(mw_plain_labels, lens_populations_by_model):
    for _fi, _pop in enumerate(_pops):
        _tag = f" (field {_fi+1})" if len(FIELDS) > 1 else ""
        _checks.append((f"Milky Way model {_lab}{_tag}", _pop.density))
if INCLUDE_SOURCE_HALO:
    for _fi, _pop in enumerate(source_populations_by_field):
        _tag = f" (field {_fi+1})" if len(FIELDS) > 1 else ""
        _checks.append((f"source{_tag}", _pop.density))
for _name, _rho in _checks:
    _vals = np.asarray(_rho(_x_check), dtype=float)
    if not np.all(np.isfinite(_vals)) or np.any(_vals < 0):
        print(f"  [warning] {_name} density function produced non-finite or negative "
              f"values somewhere in 0.001<x<0.999; those points will be treated as 0 "
              f"density during integration.")
print("Halo density functions built.")


print("\nSTEP 4 of 6: Observed data")
print("=" * 70)

if ANALYSIS_METHOD == "likelihood":
    # =============================================================================
    # Observed Einstein crossing times (tE)
    # =============================================================================

    _EXAMPLE_TE = [35.165, 165.941, 52.127, 13.497, 102.756, 64.595, 41.474,
                   45.051, 7.007, 88.188, 31.278, 41.800, 13.790, 101.230, 8.200]

    tEObserved = prompt_list_of_floats(
        "\nEnter your observed tE values (days) as a Python list, e.g.\n"
        f"  {_EXAMPLE_TE}\n"
        "or type 'none'/'[]' for zero observed events (projected-sensitivity mode).\n"
        "There is no default; this is specific to your data.",
        default_list=None, allow_empty=True
    )
    print(f"\n-> {len(tEObserved)} observed event(s) entered.")


elif ANALYSIS_METHOD == "poisson_ul":
    print("No event-timescale list is needed for this method (zero background is\n"
          "assumed).\n")

    N_OBS = prompt_int(
        "Number of candidate microlensing events your survey observed (e.g. 0 for\n"
        "a non-detection).",
        default=0, min_val=0
    )
    CONFIDENCE_LEVEL = prompt_float(
        "\nConfidence level for the upper limit, as a fraction (e.g. 0.95 for 95%).",
        default=0.95, min_val=0.5, max_val=0.999999
    )
    POISSON_UL_THRESHOLD = 0.5 * chi2.ppf(CONFIDENCE_LEVEL, 2 * (N_OBS + 1))
    print(f"\n-> {N_OBS} observed event(s) at {CONFIDENCE_LEVEL*100:g}% CL -> the predicted\n"
          f"   number of dark-object-induced events must not exceed "
          f"{POISSON_UL_THRESHOLD:.4f} (zero background assumed).")

elif ANALYSIS_METHOD == "binned":
    print("No raw list of individual observed event timescales is needed for this\n"
          "method: only a choice of how to bin the continuous background curve\n"
          "you'll enter in Step 6 below (shared with method L).\n")

    while True:
        BIN_EDGES = prompt_list_of_floats(
            "Enter the tE bin edges (days) as a sorted list, e.g. the one below, which\n"
            "defines 5 bins. At least 2 edges (1 bin) are required; this has no default,\n"
            "since the right choice of bins depends on your data.\n"
            "  [0.1, 0.5, 2.0, 10.0, 50.0, 300.0]",
            default_list=None, allow_empty=False
        )
        BIN_EDGES = np.sort(np.asarray(BIN_EDGES, dtype=float))
        if len(BIN_EDGES) < 2:
            print("  Need at least 2 edges (i.e. at least 1 bin). Please re-enter.")
            continue
        if np.any(np.diff(BIN_EDGES) <= 0):
            print("  Edges must be strictly increasing once sorted (no duplicates). Please re-enter.")
            continue
        break
    print(f"\n-> {len(BIN_EDGES) - 1} bin(s) entered, spanning tE in "
          f"[{BIN_EDGES[0]:.3g}, {BIN_EDGES[-1]:.3g}] days.")

    BINNED_NFG_MODE = prompt_str(
        "\nDo you have a CONTINUOUS dN_bkg/dtE curve (same format method L would use,\n"
        "integrated over these bins below), or do you already have PER-BIN N_FG values\n"
        "DIRECTLY (e.g. already-binned foreground counts, one value\n"
        "per bin above; do NOT use this if what you actually have is a continuous curve\n"
        "you happened to evaluate at some points, only if the bins themselves are already\n"
        "fixed and given)? Enter 'curve' or 'direct'.", default="curve"
    ).strip().lower()
    while BINNED_NFG_MODE not in ("curve", "direct"):
        BINNED_NFG_MODE = prompt_str("  Please enter 'curve' or 'direct'.", default="curve").strip().lower()

    if BINNED_NFG_MODE == "direct":
        while True:
            BIN_NFG_DIRECT = prompt_list_of_floats(
                f"\nEnter the {len(BIN_EDGES) - 1} per-bin N_FG values directly, in the SAME\n"
                f"order as the bin edges above (first value for "
                f"[{BIN_EDGES[0]:.3g}, {BIN_EDGES[1]:.3g}], etc.).\n"
                f"No default, this has to match your own data.",
                default_list=None, allow_empty=False
            )
            BIN_NFG_DIRECT = np.asarray(BIN_NFG_DIRECT, dtype=float)
            if len(BIN_NFG_DIRECT) != len(BIN_EDGES) - 1:
                print(f"  Need exactly {len(BIN_EDGES) - 1} values (one per bin), got "
                      f"{len(BIN_NFG_DIRECT)}. Please re-enter.")
                continue
            if np.any(BIN_NFG_DIRECT < 0):
                print("  N_FG values cannot be negative. Please re-enter.")
                continue
            break
        print(f"\n-> {len(BIN_NFG_DIRECT)} per-bin N_FG value(s) entered directly; "
              f"total N_FG = {BIN_NFG_DIRECT.sum():.3g}.")

    CONFIDENCE_LEVEL = prompt_float(
        "\nConfidence level for the upper limit, as a fraction (e.g. 0.95 for 95%).",
         default=0.90, min_val=0.5, max_val=0.999999
    )
    KAPPA_DOF = prompt_int(
        "\nDegrees of freedom for the kappa threshold (Croon, McKeen & Raj 2020, and\n"
        "this tool's likelihood method, both use 2 even though this is nominally a\n"
        "1-parameter f_EDO scan; change only if you have a specific reason to).",
        default=2, min_val=1
    )
    KAPPA_THRESHOLD = chi2.ppf(CONFIDENCE_LEVEL, KAPPA_DOF)
    print(f"\n-> {CONFIDENCE_LEVEL*100:g}% CL, {KAPPA_DOF} dof -> the binned deviance "
          f"(kappa) must not exceed {KAPPA_THRESHOLD:.4f}.")


# =============================================================================
# Detector/survey efficiency vs tE
# =============================================================================

print("\nSTEP 5 of 6: Detector efficiency vs tE")
print("=" * 70)

# Default: digitized OGLE-IV detection efficiency [tE (days), eps]
_DEFAULT_EFF = [
    [3.383376,0.010457],[3.622332,0.011588],[3.825592,0.012859],[4.040258,0.014324],
    [4.266969,0.015977],[4.506402,0.017798],[4.759270,0.019826],[5.026327,0.022057],
    [5.272266,0.024309],[5.492628,0.026483],[5.761383,0.029262],[6.043289,0.032417],
    [6.295876,0.035316],[6.603935,0.039072],[6.927067,0.043303],[7.216593,0.047095],
    [7.621538,0.052156],[8.215697,0.057390],[8.916821,0.063158],[9.677778,0.069624],
    [10.503675,0.076622],[11.478119,0.083773],[12.715333,0.091656],[14.182361,0.100962],
    [15.926970,0.111281],[18.008668,0.122020],[20.362449,0.133796],[23.023876,0.145371],
    [26.033159,0.156507],[29.435762,0.168111],[33.283096,0.180368],[37.633286,0.191975],
    [42.552057,0.204095],[48.113726,0.216236],[54.402319,0.225069],[61.512848,0.233327],
    [69.552742,0.242303],[78.643473,0.252635],[88.922386,0.264163],[100.544781,0.276058],
    [113.686254,0.285369],[128.545352,0.294825],[145.346575,0.304419],[164.343763,0.312532],
    [185.823935,0.320494],[210.111622,0.329224],[237.573776,0.339355],[268.625306,0.350800],
    [303.735356,0.360768],[343.434383,0.365538]
]

def _dedupe_sorted_xy(arr):
    """Merge rows with duplicate x (averaging y) so that x is strictly increasing."""
    x_rounded = np.round(arr[:, 0], 9)
    uniq_x, inverse = np.unique(x_rounded, return_inverse=True)
    if len(uniq_x) == len(arr):
        return arr
    merged = np.array([[arr[inverse == k, 0].mean(), arr[inverse == k, 1].mean()]
                       for k in range(len(uniq_x))])
    print(f"  [note] {len(arr) - len(uniq_x)} duplicate tE value(s) found and merged "
          f"(averaging their y-value) before interpolating.")
    return merged

def _prompt_one_efficiency(tag=""):
    """Prompt for one [tE, efficiency] table; returns (eps(tE), tE_min, tE_max), zero outside the range."""
    arr = prompt_2d_array(
        f"\nEnter your [tE, efficiency] table{tag} as a Python literal or a data-file path.\n"
        f"Press Enter to use the built-in OGLE-IV example table.",
        default_array=_DEFAULT_EFF,
        example_str="  [[3.4, 0.010], [10.5, 0.077], [50.0, 0.216], [300.0, 0.36]]"
    )
    arr = arr[np.argsort(arr[:, 0])]
    arr = _dedupe_sorted_xy(arr)
    lo, hi = float(arr[:, 0].min()), float(arr[:, 0].max())
    interp = interp1d(arr[:, 0], np.clip(arr[:, 1], 0.0, 1.0),
                      kind='cubic' if len(arr) >= 4 else 'linear',
                      bounds_error=False, fill_value=0.0)
    def _f(tE, _interp=interp, _lo=lo, _hi=hi):
        if _lo < tE <= _hi:
            return float(np.clip(_interp(tE), 0.0, 1.0))
        return 0.0
    return _f, lo, hi

if _FIELD_EFFICIENCIES_PREBUILT:
    print(f"\n-> Already read from the efficiency folder in Step 2: {len(FIELD_EFFICIENCIES)} "
          f"field(s), tE in [{tE_eff_min:.3f}, {tE_eff_max:.3f}] days. Nothing further to enter here.")
elif len(FIELDS) == 1:
    efficiency, tE_eff_min, tE_eff_max = _prompt_one_efficiency()
    FIELD_EFFICIENCIES = [efficiency]
    print(f"\n-> Efficiency table spans tE in [{tE_eff_min:.3f}, {tE_eff_max:.3f}] days; "
          f"this sets the signal-integral bounds.")
else:
    print(f"\nThis analysis combines {len(FIELDS)} fields (Step 2). Enter a separate\n"
          f"efficiency table for EACH one below. Different fields can genuinely have\n"
          f"different detection efficiency if they were observed with a different cadence\n"
          f"or strategy.")
    FIELD_EFFICIENCIES, _mins, _maxs = [], [], []
    for _fi, _fld in enumerate(FIELDS):
        _tag = f" for field {_fi + 1}/{len(FIELDS)} (DS={_fld['DS']:g} kpc)"
        _fn, _lo, _hi = _prompt_one_efficiency(tag=_tag)
        FIELD_EFFICIENCIES.append(_fn)
        _mins.append(_lo); _maxs.append(_hi)
    tE_eff_min, tE_eff_max = min(_mins), max(_maxs)
    print(f"\n-> Combined signal-integral bounds (union across {len(FIELDS)} field(s)): "
          f"tE in [{tE_eff_min:.3f}, {tE_eff_max:.3f}] days.")


print("\nSTEP 6 of 6: Background")
print("=" * 70)

if ANALYSIS_METHOD == "likelihood" or (ANALYSIS_METHOD == "binned" and BINNED_NFG_MODE == "curve"):
    # =============================================================================
    # Background timescale distribution dN_bkg/dtE (optional)
    # =============================================================================

    # Default: digitized OGLE-IV/LMC background table [tE (days), value]
    _DEFAULT_BKG = [
        [5.415801,0.010046],[6.988625,0.024968],[8.706219,0.052606],[10.905560,0.097458],
        [13.692023,0.184014],[17.122605,0.294968],[21.626467,0.445310],[26.869709,0.623081],
        [34.138017,0.764308],[43.059649,0.941676],[54.222959,1.005343],[68.126260,1.043914],
        [86.638505,1.063538],[109.172217,0.999832],[137.554095,0.956577],[172.935184,0.878490],
        [218.525108,0.765408],[276.792154,0.669811],[346.672267,0.505699]
    ]

    def _prompt_one_background(tag=""):
        """Prompt for one background table (or 'none'); returns (dN_bkg/dtE or None, tE_min, tE_max)."""
        _use_default = prompt_yes_no(
            f"\nUse the default OGLE-IV/LMC background table{tag}? (Choose 'n' to enter "
            f"your own table or 'none'.)", default=True
        )
        if _use_default:
            arr = np.array(_DEFAULT_BKG, dtype=float)
        else:
            arr = prompt_2d_array(
                f"\nEnter your [tE, dN_bkg/dtE] table{tag} as a Python literal or a "
                f"data-file path,\nor type 'none' to assume zero background for this field.",
                default_array=None, allow_none=True,
                example_str="  [[5.4, 0.010], [20.0, 0.45], [80.0, 1.06], [300.0, 0.60]]"
            )
        if arr is None:
            return None, 0.0, 0.0
        arr = arr[np.argsort(arr[:, 0])]
        arr = _dedupe_sorted_xy(arr)

        _is_histogram = prompt_yes_no(
            f"\nIs this a HISTOGRAM{tag} i.e. each point is the COUNT of events already\n"
            f"summed within that tE bin (e.g. straight off a published figure), rather than\n"
            f"a continuous density dN_bkg/dtE? Answer 'y' for a digitized histogram like\n"
            f"Mroz et al. (2024)'s Extended Data Fig. 6b; answer 'n' if your points already\n"
            f"are a dN_bkg/dtE density (e.g. you computed them yourself, or divided by bin\n"
            f"width already).", default=True
        )
        if _is_histogram:
            _log_binwidth = prompt_float(
                "Bin width in log10(tE), i.e. the spacing between consecutive bin centers\n"
                "(0.1 dex is typical for this kind of figure; check your own points' spacing\n"
                "if unsure).", default=0.1, min_val=1e-6
            )
            # counts per log10 bin -> density: dN/dtE = N_bin / (tE ln(10) dlog10tE)
            _bin_width_natural_log = np.log(10.0) * _log_binwidth
            arr = arr.copy()
            arr[:, 1] = arr[:, 1] / (arr[:, 0] * _bin_width_natural_log)
            print(f"-> Converted {len(arr)} histogram bin count(s) to a dN_bkg/dtE density "
                  f"(divided each by tE*ln(10)*{_log_binwidth:g}).")

        lo, hi = float(arr[:, 0].min()), float(arr[:, 0].max())
        interp = interp1d(arr[:, 0], np.clip(arr[:, 1], 0.0, None),
                          kind='cubic' if len(arr) >= 4 else 'linear',
                          bounds_error=False, fill_value=0.0)
        def _f(tE, _interp=interp, _lo=lo, _hi=hi):
            if _lo < tE <= _hi:
                return float(max(0.0, _interp(tE)))
            return 0.0
        return _f, lo, hi

    if len(FIELDS) == 1 or FIELD_DATA_MODE == "per_field_bulk":
        _bkg_fn, tE_bkg_min, tE_bkg_max = _prompt_one_background()
        if _bkg_fn is None:
            tE_bkg_min = tE_bkg_max = 0.0
            def dNbkgDtE(tE):
                return 0.0
            print("\n-> No background: dN_bkg/dtE = 0 everywhere.")
        else:
            dNbkgDtE = _bkg_fn
            print(f"\n-> Background table spans tE in [{tE_bkg_min:.3f}, {tE_bkg_max:.3f}] days.")
    else:
        print(f"\nThis analysis combines {len(FIELDS)} fields (Step 2). Enter a separate\n"
              f"background table for EACH one below; they will be summed into one combined\n"
              f"dN_bkg/dtE (the observed-event list stays a single pooled list either way).")
        _field_bkg_fns, _mins, _maxs = [], [], []
        for _fi, _fld in enumerate(FIELDS):
            _tag = f" for field {_fi + 1}/{len(FIELDS)} (DS={_fld['DS']:g} kpc)"
            _fn, _lo, _hi = _prompt_one_background(tag=_tag)
            _field_bkg_fns.append(_fn)
            if _fn is not None:
                _mins.append(_lo); _maxs.append(_hi)
        def dNbkgDtE(tE, _fns=_field_bkg_fns):
            return sum(_fn(tE) for _fn in _fns if _fn is not None)
        tE_bkg_min = min(_mins) if _mins else 0.0
        tE_bkg_max = max(_maxs) if _maxs else 0.0
        if _mins:
            print(f"\n-> Combined background (summed over fields) spans tE in "
                  f"[{tE_bkg_min:.3f}, {tE_bkg_max:.3f}] days.")
        else:
            print("\n-> No background: dN_bkg/dtE = 0 everywhere for every field.")

else:
    if ANALYSIS_METHOD == "poisson_ul":
        print("Not asked: the Poisson upper-limit method assumes zero background (see\n"
              "the method-choice prompt near the top of this notebook).")
    else:
        print("Not asked: you already entered per-bin N_FG values directly in Step 4.")

# Binned method: foreground count per bin, N_FG,i (background curve integrated over each bin)
if ANALYSIS_METHOD == "binned":
    if BINNED_NFG_MODE == "curve":
        BIN_NFG = np.array([
            quad(dNbkgDtE, BIN_EDGES[i], BIN_EDGES[i + 1], epsabs=1e-8, epsrel=1e-4, limit=100)[0]
            for i in range(len(BIN_EDGES) - 1)
        ])
        BIN_NFG = np.clip(BIN_NFG, 0.0, None)
    else:
        BIN_NFG = BIN_NFG_DIRECT

    print(f"\n-> Binned foreground: N_FG per bin = "
          f"{np.array2string(BIN_NFG, precision=3)};\n   total N_FG = {BIN_NFG.sum():.3g}.")

    _zero_bins = [i for i, n in enumerate(BIN_NFG) if n <= 0.0]
    if _zero_bins:
        if BINNED_NFG_MODE == "curve":
            _outside = [i for i in _zero_bins
                        if BIN_EDGES[i + 1] <= tE_bkg_min or BIN_EDGES[i] >= tE_bkg_max]
            if _outside:
                print(f"   [NOTE] bin(s) {[b + 1 for b in _outside]} of {len(BIN_NFG)} fall entirely outside "
                      f"the background curve's own\n"
                      f"   digitized range [{tE_bkg_min:.3f}, {tE_bkg_max:.3f}] days, so N_FG=0 there by "
                      f"construction (not\n"
                      f"   because the background is known to vanish there). This is handled exactly by the "
                      f"kappa statistic\n"
                      f"   below (a finite 2*N_DM,i penalty, not a divergence), but consider choosing bin "
                      f"edges within\n"
                      f"   the curve's digitized range if that wasn't intentional.")
        else:
            print(f"   [NOTE] bin(s) {[b + 1 for b in _zero_bins]} of {len(BIN_NFG)} have N_FG=0. This is "
                  f"handled exactly\n"
                  f"   by the kappa statistic below (a finite 2*N_DM,i penalty, not a divergence), but it's "
                  f"still worth\n"
                  f"   double-checking those are genuinely zero rather than just small and imprecisely read.")


# =============================================================================
# Mass-profile interpolation and extended-lens formulas
# =============================================================================
# tau: image-plane radius in units of the Einstein radius; tau_M = R90/R_E.

# m(x): enclosed projected-mass fraction at x = s_factor*tau/tau_M, interpolated from the table.
def build_mass_profile_interpolator(points):
    pts = np.array(points, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError("Mass profile must be a 2D array of [x, mass_fraction] pairs.")
    pts = pts[np.argsort(pts[:, 0])]
    _, uniq_idx = np.unique(pts[:, 0], return_index=True)
    pts = pts[np.sort(uniq_idx)]
    xs, ms = pts[:, 0], pts[:, 1]
    if xs[0] > 0:
        xs = np.insert(xs, 0, 0.0); ms = np.insert(ms, 0, 0.0)
    if xs[-1] < 1:
        xs = np.append(xs, 1.0); ms = np.append(ms, 1.0)
    ms = np.clip(ms, 0.0, 1.0)
    kind = 'cubic' if len(xs) >= 4 else 'linear'
    f = interp1d(xs, ms, kind=kind, bounds_error=False, fill_value=(0.0, 1.0))
    def m_of_x(x):
        """Enclosed mass fraction m(x), clipped to [0, 1] and non-decreasing."""
        x = np.clip(np.atleast_1d(x).astype(float), 0.0, 1.0)
        out = np.clip(f(x), 0.0, 1.0)
        if out.size > 1:
            order = np.argsort(x)
            out_sorted = np.maximum.accumulate(out[order])
            result = np.empty_like(out)
            result[order] = out_sorted
            out = result
        return out
    return m_of_x


class ProfilePhysics:
    """Lensing functions for the EDO mass profile, including the threshold impact parameter u_T(tau_M)."""

    def __init__(self, mass_profile_points, s_factor):
        self.m_of_x = build_mass_profile_interpolator(mass_profile_points)
        self.s_factor = float(s_factor)

    def mProjected_vec(self, tau, tauM):
        """Enclosed mass fraction at image-plane radius tau (1 outside the object)."""
        tau = np.atleast_1d(tau).astype(float)
        a = np.abs(tau)
        out = np.ones_like(a)
        if tauM > 0:
            inside = (a >= 0) & (a <= tauM / self.s_factor)
            if np.any(inside):
                out[inside] = self.m_of_x(self.s_factor * a[inside] / tauM)
        return out

    # Magnification of the image at tau: mu = |(1 - m/tau^2)(1 + m/tau^2 - m'/tau)|^-1
    def muExtended_vec(self, tau, tauM):
        tau = np.atleast_1d(tau).astype(float)
        if tauM <= SMALL_TAUM_EXACT_THRESHOLD:
            with np.errstate(divide='ignore', invalid='ignore'):
                u = tau - 1.0 / tau  # exact point-mass limit: m(u)=1 for all u>0
                mu = (u**2 + 2.0) / (u * np.sqrt(u**2 + 4.0))
            return mu
        mVal = self.mProjected_vec(tau, tauM)
        h = 1e-4
        dmVal = (self.mProjected_vec(tau + h, tauM) - self.mProjected_vec(tau - h, tauM)) / (2 * h)
        with np.errstate(divide='ignore', invalid='ignore'):
            denom = np.abs((1.0 - mVal / tau**2) * (1.0 + mVal / tau**2 - dmVal / tau))
            mu = np.where(denom > 1e-300, 1.0 / np.where(denom > 1e-300, denom, 1.0), 1e14)
        return mu

    @staticmethod
    def _monotonic_branches(u_vals, mu_vals):
        """Split (u, mu) into runs over which u is monotonic."""
        finite = np.isfinite(u_vals) & np.isfinite(mu_vals)
        branches = []
        n = len(u_vals)
        i = 0
        while i < n:
            if not finite[i]:
                i += 1
                continue
            j = i + 1
            direction = 0
            while j < n and finite[j]:
                d = u_vals[j] - u_vals[j - 1]
                if d == 0:
                    j += 1
                    continue
                s = 1 if d > 0 else -1
                if direction == 0:
                    direction = s
                elif s != direction:
                    break
                j += 1
            if j - i >= 2:
                u_seg, mu_seg = u_vals[i:j], mu_vals[i:j]
                if u_seg[0] > u_seg[-1]:
                    u_seg, mu_seg = u_seg[::-1], mu_seg[::-1]
                branches.append((u_seg, mu_seg))
            i = j - 1 if j > i + 1 else i + 1
        return branches

    def uT(self, tauM, tau_min=1e-4, n_scan=6000, max_expand=10, u_scan_cap=50.0):
        """Threshold impact parameter u_T: total magnification >= MU_THRESHOLD for all u <= u_T."""
        # Scan the image plane, map each monotonic branch tau -> u(tau), sum the magnifications
        # of all images at each u, and locate the first downward crossing of MU_THRESHOLD.
        if tauM <= 1e-3:
            return 1.0
        tau_min_eff = min(tau_min, tauM * 1e-3)
        tau_max = max(10.0, 3.0 * tauM)
        for _ in range(max_expand):
            taus_pos = np.geomspace(tau_min_eff, tau_max, n_scan)
            taus = np.concatenate([-taus_pos[::-1], taus_pos])
            u_vals = taus - self.mProjected_vec(taus, tauM) / taus   # lens equation u = tau - m/tau
            mu_vals = self.muExtended_vec(taus, tauM)
            n_half = len(taus_pos)
            branches = (self._monotonic_branches(u_vals[:n_half], mu_vals[:n_half])
                        + self._monotonic_branches(u_vals[n_half:], mu_vals[n_half:]))
            pos_branches = [b for b in branches if b[0][-1] > 0]
            if not pos_branches:
                tau_max *= 2.0
                continue
            u_grid_min = max(1e-6, min(max(b[0][0], 0.0) for b in pos_branches))
            u_scan_max = min(max(b[0][-1] for b in pos_branches), u_scan_cap)
            if u_scan_max <= u_grid_min:
                tau_max *= 2.0
                continue
            u_grid = np.linspace(u_grid_min, u_scan_max, 6000)
            mu_tot_grid = np.zeros_like(u_grid)
            for u_seg, mu_seg in pos_branches:
                mu_tot_grid += np.interp(u_grid, u_seg, mu_seg, left=0.0, right=0.0)
            if mu_tot_grid[-1] < MU_THRESHOLD:
                break
            tau_max *= 2.0
        else:
            return 0.0

        g = mu_tot_grid - MU_THRESHOLD
        sign_change = (g[:-1] > 0) & (g[1:] <= 0)
        idxs = np.nonzero(sign_change)[0]
        if len(idxs) == 0:
            return 0.0
        i = idxs[0]
        a, b = u_grid[i], u_grid[i + 1]

        def mu_tot_scalar(uq):
            total = 0.0
            for u_seg, mu_seg in pos_branches:
                if u_seg[0] <= uq <= u_seg[-1]:
                    total += np.interp(uq, u_seg, mu_seg)
            return total

        try:
            u_sol = brentq(lambda uu: mu_tot_scalar(uu) - MU_THRESHOLD, a, b, xtol=1e-10)
        except Exception:
            return max(0.0, a)
        return max(0.0, u_sol)

    def build_uT_interpolator(self, tauM_table_max=100.0, tauM_step=0.05, verbose=True):
        """Tabulate u_T(tau_M) and return a cubic interpolator."""
        tauM_grid = np.arange(1e-4, tauM_table_max + tauM_step, tauM_step)
        uT_vals = np.empty_like(tauM_grid)
        for i, tm in enumerate(tauM_grid):
            uT_vals[i] = self.uT(tm)
        if verbose:
            nz = np.nonzero(uT_vals > 1e-8)[0]
            if len(nz) and nz[-1] < len(tauM_grid) - 1:
                print(f"  This profile's threshold impact parameter u_T naturally reaches "
                      f"zero at tauM ~ {tauM_grid[nz[-1] + 1]:.2f} "
                      f"(within the {tauM_table_max:.0f}-cap table): objects this "
                      f"extended, relative to the Einstein radius, no longer produce "
                      f"detectable magnification.")
            else:
                print(f"  This profile's u_T has NOT reached zero by tauM={tauM_table_max:.0f}; "
                      f"the table is truncated there. This is expected for cuspy/cored "
                      f"profiles whose lensing cross-section keeps growing with size; "
                      f"the rate integral's own velocity suppression makes the truncation "
                      f"safe (raise tauM_table_max above to check).")
        self.uT_interp = interp1d(tauM_grid, uT_vals, kind='cubic',
                                   bounds_error=False, fill_value=(1.0, 0.0))
        self.tauM_grid = tauM_grid
        self.uT_vals = uT_vals
        return self.uT_interp


# One ProfilePhysics object (with its u_T(tau_M) table) per EDO type
edo_physics_list = []
for _edo_idx, (_edo_label, _mp_points, _edo_sfactor) in enumerate(edo_profiles_raw):
    print(f"\nBuilding the profile physics engine and the u_T(tauM) lookup table for "
          f"{_edo_label} [{_edo_idx + 1}/{len(edo_profiles_raw)}] "
          f"(this runs once per EDO type, may take a few seconds)...")
    _this_physics = ProfilePhysics(_mp_points, _edo_sfactor)
    _this_physics.build_uT_interpolator(tauM_table_max=100.0, tauM_step=0.05, verbose=True)
    edo_physics_list.append((_edo_label, _this_physics))
    print("Done.")

profile_physics = edo_physics_list[0][1]   # EDO type read by dGammaDtE(); reassigned in the main loop


# =============================================================================
# Microlensing rate integrals and limit-setting statistics (methods L, P, B)
# =============================================================================

def REinstein(x, M, DS):
    """Einstein radius [kpc] at line-of-sight fraction x for lens mass M [Msun], source distance DS [kpc]."""
    x = np.asarray(x, dtype=float)
    return x * DS * np.sqrt(np.clip((4.0 * G_KPC * M * (1.0 - x)) / (C_KMS**2 * x * DS), 0, None))

# tau_M = R90 / R_E
def calctauM(x, M, DS):
    RE = REinstein(x, M, DS)
    RE = np.where(RE > 1e-300, RE, 1e-300)
    return R90_kpc / RE


# Composite Gauss-Legendre quadrature over x with the substitution x = sin^2(theta),
# which concentrates nodes near both ends of the line of sight.
def _make_x_quadrature(n_panels=40, n_gl=16, x_lo=0.001, x_hi=0.999):
    th_lo, th_hi = np.arcsin(np.sqrt(x_lo)), np.arcsin(np.sqrt(x_hi))
    g_x, g_w = np.polynomial.legendre.leggauss(n_gl)
    edges = np.linspace(th_lo, th_hi, n_panels + 1)
    a, b = edges[:-1, None], edges[1:, None]
    theta = (0.5*(b - a)*g_x[None, :] + 0.5*(a + b)).ravel()
    w_theta = (0.5*(b - a)*g_w[None, :]).ravel()
    return np.sin(theta)**2, w_theta*np.sin(2.0*theta)

_X_NODES, _X_WEIGHTS = _make_x_quadrature()

def d2GammaDxdtE(x, tE, M, population, DS):
    """d^2 Gamma/(dx dtE) per source star [day^-2] for one lens population."""
    x = np.atleast_1d(np.asarray(x, dtype=float))
    RE = REinstein(x, M, DS)
    tauM = calctauM(x, M, DS)
    uTVal = np.where(tauM >= profile_physics.tauM_grid[-1], 0.0,
                     profile_physics.uT_interp(np.maximum(tauM, 1e-4)))
    uTVal = np.where(np.isfinite(uTVal) & (uTVal > 0.0), uTVal, 0.0)
    # Rate of Croon, McKeen & Raj (2020, Eq. 7), with tE the Einstein-RADIUS crossing time:
    # v_E = u_T R_E / tE,  d2Gamma/(dx dtE) = (4 pi DS / M) v_E^4 G_perp(x, v_E)
    vE = uTVal * RE / (tE / DAYS_PER_YEAR)                 # [kpc/yr]
    G = np.asarray(population.G_perp(x, vE), dtype=float)
    G = np.where(np.isfinite(G) & (G > 0.0), G, 0.0)
    rate = (4.0 * np.pi * DS / M) * vE**4 * G
    return rate / (DAYS_PER_YEAR**2)

def dGammaDtE(tE, M):
    """Efficiency-weighted dGamma/dtE [day^-1], summed over fields (Nstar_f * eps_f(tE) per field)."""
    # The x-integral is evaluated once per (DS, l, b) group and weighted by the group's
    # sum of Nstar_f * eps_f(tE); Milky-Way lenses first, then (optionally) source lenses.
    total = 0.0
    for _indices in _FIELD_GEOM_GROUPS.values():
        _i0 = _indices[0]
        per_star_rate = np.sum(d2GammaDxdtE(_X_NODES, tE, M, CURRENT_MODEL_FIELD_POPS[_i0],
                                            FIELDS[_i0]["DS"]) * _X_WEIGHTS)
        if per_star_rate <= 0.0:
            continue
        _weight = 0.0
        for _i in _indices:
            _e = FIELD_EFFICIENCIES[_i](tE)
            if _e > 0.0:
                _weight += FIELDS[_i]["Nstar"] * _e
        total += per_star_rate * _weight
    if INCLUDE_SOURCE_HALO:
        for _indices in _FIELD_GEOM_GROUPS.values():
            _i0 = _indices[0]
            per_star_rate = np.sum(d2GammaDxdtE(_X_NODES, tE, M, source_populations_by_field[_i0],
                                                FIELDS[_i0]["DS"]) * _X_WEIGHTS)
            if per_star_rate <= 0.0:
                continue
            _weight = 0.0
            for _i in _indices:
                _e = FIELD_EFFICIENCIES[_i](tE)
                if _e > 0.0:
                    _weight += FIELDS[_i]["Nstar"] * _e
            total += per_star_rate * _weight
    return max(0.0, float(total))

# --- Method L: unbinned Poisson likelihood --------------------------------------
# ln L(f) = sum_i ln[dN_exp/dtE(tE_i)] - N_exp,  dN_exp/dtE = f Exposure dGamma/dtE + dN_bkg/dtE
if ANALYSIS_METHOD == "likelihood":
    def dNsigDtE(tE, M, fEDO):
        return fEDO * Exposure * dGammaDtE(tE, M)

    def dNexpDtE(tE, M, fEDO):
        return dNsigDtE(tE, M, fEDO) + dNbkgDtE(tE)

    def NexpTotal(M, fEDO):
        nSig, _ = quad(lambda tE: dNsigDtE(tE, M, fEDO), tE_eff_min, tE_eff_max,
                        epsabs=1e-8, epsrel=1e-4, limit=100)
        nBkg = 0.0
        if tE_bkg_max > tE_bkg_min:
            nBkg, _ = quad(dNbkgDtE, tE_bkg_min, tE_bkg_max, epsabs=1e-8, epsrel=1e-4, limit=100)
        return nSig + nBkg

    def UnbinnedLogLikelihood(M, fEDO):
        s = 0.0
        for tE in tEObserved:
            s += safe_log(dNexpDtE(tE, M, fEDO))
        return s - NexpTotal(M, fEDO)

    # q(f) = 2[ln L(0) - ln L(f)]; the frequentist limit solves q(f) = DELTA_CHI2_95
    def TestStatisticQ(M, fEDO, logLHat=None):
        if logLHat is None:
            logLHat = UnbinnedLogLikelihood(M, 0.0)
        return 2.0 * (logLHat - UnbinnedLogLikelihood(M, fEDO))

    def FindLimitF95(M):
        """Frequentist 95% CL upper limit on f_EDO at mass M (likelihood-ratio threshold)."""
        logLHat = UnbinnedLogLikelihood(M, 0.0)

        def obj(f):
            return TestStatisticQ(M, f, logLHat=logLHat) - DELTA_CHI2_95

        try:
            f0 = obj(1e-6)
        except Exception:
            f0 = -DELTA_CHI2_95
        try:
            f1 = obj(0.999)
        except Exception:
            f1 = 1e6

        if f0 * f1 > 0:
            return 1.0 if f1 < 0 else 0.0
        try:
            sol = brentq(obj, 1e-6, 0.999, xtol=1e-5)
            return float(min(1.0, max(0.0, sol)))
        except Exception:
            return 1.0

    print("Rate-integral and likelihood functions defined.")
    if len(tEObserved) > 0 and tE_bkg_max <= tE_bkg_min:
        print("\n[NOTE] You entered observed events but no background model."
              "This typically makes the 95% CL limit unconstrained "
              "(f_limit = 1) at every mass, because the analysis has no way to explain "
              "the observed events except via signal. Consider adding a background "
              "table, or clearing tEObserved to compute a projected sensitivity instead.")

    # Bayesian alternative (Mroz et al. 2024, Eq. 5): flat prior on f in [0, F_PRIOR_MAX].
    # The posterior is evaluated on a grid that is widened until it has decayed at the upper
    # edge; the limit is the 95% point of its CDF.
    F_PRIOR_MAX = 10.0

    def _bayesian_logL_closure(M):
        """logL(f) at mass M with all f-independent terms precomputed once (exact: the signal is linear in f)."""
        sig_at_events = [Exposure * dGammaDtE(tE, M) for tE in tEObserved]
        bkg_at_events = [dNbkgDtE(tE) for tE in tEObserved]
        S_total, _ = quad(lambda tE: Exposure * dGammaDtE(tE, M), tE_eff_min, tE_eff_max,
                          epsabs=1e-8, epsrel=1e-4, limit=100)
        B_total = 0.0
        if tE_bkg_max > tE_bkg_min:
            B_total, _ = quad(dNbkgDtE, tE_bkg_min, tE_bkg_max, epsabs=1e-8, epsrel=1e-4, limit=100)

        def logL_fast(f):
            s = 0.0
            for sig_i, bkg_i in zip(sig_at_events, bkg_at_events):
                s += safe_log(f * sig_i + bkg_i)
            return s - (f * S_total + B_total)

        return logL_fast

    def FindLimitF95_Bayesian(M, f_seed=None, n_grid=500, max_widen=6):
        """Bayesian 95% credible upper limit on f_EDO at mass M (flat prior on [0, F_PRIOR_MAX]), capped at 1."""
        if f_seed is None or not np.isfinite(f_seed) or f_seed <= 0:
            f_seed = 0.01
        logL_fast = _bayesian_logL_closure(M)
        f_hi = min(F_PRIOR_MAX, max(20.0 * f_seed, 0.05))
        for _ in range(max_widen):
            fs = np.linspace(1e-10, f_hi, n_grid)
            logLs = np.array([logL_fast(f) for f in fs])
            logLs -= logLs.max()
            post = np.exp(logLs)
            edge_ratio = post[-1] / post.max() if post.max() > 0 else 0.0
            if edge_ratio < 1e-6 or f_hi >= F_PRIOR_MAX - 1e-9:
                break
            f_hi = min(F_PRIOR_MAX, f_hi * 4.0)
        cdf = np.concatenate([[0.0], np.cumsum(0.5 * (post[1:] + post[:-1]) * np.diff(fs))])
        cdf /= cdf[-1]
        return float(min(1.0, np.interp(0.95, cdf, fs)))

    FindLimitFast = FindLimitF95   # cheap limit, used only by the mass-range edge scan

    STAT_METHOD_CHOICE = prompt_str(
        "\nWithin the likelihood (Pipeline L) method, compute the 95% CL limit via:\n"
        "  'F' = frequentist likelihood-ratio threshold (Delta(2lnL) = 5.991)\n"
        "  'B' = Bayesian 95% credible value, flat prior f in [0,10] (Mroz et al. 2024,\n"
        "        Eq. 5); this generally differs from 'F'.",
        default="F"
    ).strip().upper()
    while STAT_METHOD_CHOICE not in ("F", "B"):
        STAT_METHOD_CHOICE = prompt_str("  Please enter F or B.", default="F").strip().upper()
    STAT_METHOD = {"F": "frequentist", "B": "bayesian"}[STAT_METHOD_CHOICE]
    print(f"-> Statistical method: {STAT_METHOD}.")

    if STAT_METHOD == "bayesian":
        def FindLimit(M):
            f_seed = FindLimitF95(M)
            return FindLimitF95_Bayesian(M, f_seed=f_seed)
    else:
        FindLimit = FindLimitF95

# --- Method P: Poisson upper limit (zero background) ------------------------------
# N_events(f) = f N_events(1), so the limit is f = POISSON_UL_THRESHOLD / N_events(1).
elif ANALYSIS_METHOD == "poisson_ul":

    def NeventsAtFullF(M):
        """Expected number of EDO events at f_EDO = 1 (Eq. 8)."""
        val, _ = quad(lambda tE: dGammaDtE(tE, M),
                      tE_eff_min, tE_eff_max, epsabs=0.0, epsrel=1e-6, limit=200)
        return Exposure * max(0.0, val)

    def FindLimitPoissonUL(M):
        """f_EDO at which the expected number of events equals POISSON_UL_THRESHOLD."""
        n1 = NeventsAtFullF(M)
        if n1 <= 0.0:
            return 1.0
        return min(1.0, POISSON_UL_THRESHOLD / n1)

    FindLimit = FindLimitPoissonUL
    FindLimitFast = FindLimitPoissonUL
    print("Poisson-upper-limit rate function defined.")

# --- Method B: binned deviance (kappa statistic) ----------------------------------
else:

    def dNsigDtE(tE, M, fEDO):
        return fEDO * Exposure * dGammaDtE(tE, M)

    def _bin_N_DM(M, fEDO, lo, hi):
        val, _ = quad(lambda tE: dNsigDtE(tE, M, fEDO), lo, hi, epsabs=0.0, epsrel=1e-6, limit=100)
        return max(0.0, val)

    def KappaStatistic(M, fEDO):
        """Binned likelihood-ratio statistic: sum_i 2[N_DM,i - N_FG,i ln(1 + N_DM,i/N_FG,i)]."""
        kappa = 0.0
        for lo, hi, n_fg in zip(BIN_EDGES[:-1], BIN_EDGES[1:], BIN_NFG):
            n_dm = _bin_N_DM(M, fEDO, lo, hi)
            if n_dm <= 0.0:
                continue
            if n_fg <= 0.0:
                kappa += 2.0 * n_dm
            else:
                kappa += 2.0 * (n_dm - n_fg * np.log1p(n_dm / n_fg))
        return kappa

    def FindLimitBinned(M):
        """f_EDO at which KappaStatistic equals KAPPA_THRESHOLD."""
        def obj(f):
            return KappaStatistic(M, f) - KAPPA_THRESHOLD

        try:
            f0 = obj(1e-6)
        except Exception:
            f0 = -KAPPA_THRESHOLD
        try:
            f1 = obj(0.999)
        except Exception:
            f1 = 1e12

        if f0 * f1 > 0:
            return 1.0 if f1 < 0 else 0.0
        try:
            sol = brentq(obj, 1e-6, 0.999, xtol=1e-5)
            return float(min(1.0, max(0.0, sol)))
        except Exception:
            return 1.0

    FindLimit = FindLimitBinned
    FindLimitFast = FindLimitBinned
    print("Binned-deviance (kappa) rate function defined.")

CL_PERCENT = 95.0 if ANALYSIS_METHOD == "likelihood" else CONFIDENCE_LEVEL * 100.0
CL_LABEL = f"{CL_PERCENT:g}% CL"


# =============================================================================
# Mass grid and main computation loop
# =============================================================================

# The grid starts from 10^-7 - 10^2 Msun and is extended until every curve reaches
# f_EDO ~ 1 at both ends (checked with the cheap FindLimitFast).
POINTS_PER_DECADE = 10

_BASELINE_LOG_LO, _BASELINE_LOG_HI = -7.0, 2.0

def _mass_edge_scan(_label, f_threshold=0.99, max_steps=8, step_decades=1.5):
    """Extend the mass range outward until f_EDO reaches the threshold at both ends."""
    lo, hi = _BASELINE_LOG_LO, _BASELINE_LOG_HI
    f_lo = FindLimitFast(10.0 ** lo)
    for _ in range(max_steps):
        if f_lo >= f_threshold:
            break
        lo -= step_decades
        f_lo = FindLimitFast(10.0 ** lo)
    else:
        print(f"  [note] {_label}: f_EDO did not reach {f_threshold:g} on the low-mass "
              f"side by M=10^{lo:.1f} Msun; using this as its edge anyway (that curve's "
              f"left side may still look cut off).")
    f_hi = FindLimitFast(10.0 ** hi)
    for _ in range(max_steps):
        if f_hi >= f_threshold:
            break
        hi += step_decades
        f_hi = FindLimitFast(10.0 ** hi)
    else:
        print(f"  [note] {_label}: f_EDO did not reach {f_threshold:g} on the high-mass "
              f"side by M=10^{hi:.1f} Msun; using this as its edge anyway (that curve's "
              f"right side may still look cut off).")
    return lo, hi

print("\nScanning the mass range needed for f_EDO to reach ~1 at both ends, for every\n"
      "EDO type / Milky-Way model / R90 combination (quick boundary checks only, not\n"
      "the full grid)...")
_overall_log_lo, _overall_log_hi = _BASELINE_LOG_LO, _BASELINE_LOG_HI
for _edo_label, _edo_physics in edo_physics_list:
    profile_physics = _edo_physics
    for _mlabel, _field_pops in zip(mw_plain_labels, lens_populations_by_model):
        CURRENT_MODEL_FIELD_POPS = _field_pops
        for _R90_val in R90_list_solar_radii:
            R90_kpc = float(_R90_val) * RSUN_KPC
            _lo, _hi = _mass_edge_scan(f"{_edo_label}/{_mlabel}/R90={_R90_val:g}")
            _overall_log_lo = min(_overall_log_lo, _lo)
            _overall_log_hi = max(_overall_log_hi, _hi)
profile_physics = edo_physics_list[0][1]
CURRENT_MODEL_FIELD_POPS = lens_populations_by_model[0]
R90_kpc = float(R90_list_solar_radii[0]) * RSUN_KPC

_n_decades = _overall_log_hi - _overall_log_lo
_n_points = max(2, round(_n_decades * POINTS_PER_DECADE)) + 1
massGrid = np.logspace(_overall_log_lo, _overall_log_hi, _n_points)
print(f"-> Mass range: 10^{_overall_log_lo:.2f} to 10^{_overall_log_hi:.2f} Msun "
      f"({_n_decades:.1f} decades, {_n_points} points at {POINTS_PER_DECADE}/decade).")

def _prompt_plot_filename(default="EDO_constraint_plot.png"):
    """Prompt for the output plot file name."""
    while True:
        raw = prompt_str(
            f"\nFile name for the saved plot (PNG; written to the current directory,\n"
            f"{os.getcwd()}, unless you give a path). An existing file with the same\n"
            f"name is overwritten. Type 'none' to skip saving.", default=default).strip()
        if raw.lower() == "none":
            return None
        name = raw if raw.lower().endswith(".png") else raw + ".png"
        folder = os.path.dirname(os.path.abspath(name))
        if os.path.isdir(folder) and os.access(folder, os.W_OK):
            return name
        print(f"  Cannot write to the folder '{folder}' (it does not exist or is not writable). "
              f"Please try again.")

plot_filename = _prompt_plot_filename()

def _prompt_curve_save_prefix(default="none"):
    """Prompt for whether (and where) to save each curve as a text file."""
    raw = prompt_str(
        "\nSave each computed curve as its own M-vs-f_EDO text file (one per EDO type x\n"
        "Milky-Way model x R90), for combining later with the separate interpolator tool?\n"
        "Enter a filename PREFIX to save them, or leave blank/type 'none' to skip.",
        default=default).strip()
    return None if raw.lower() in ("", "none") else raw

curve_save_prefix = _prompt_curve_save_prefix()

n_edo = len(edo_physics_list)
n_radii = len(R90_list_solar_radii)
n_models = len(lens_populations_by_model)
_edo_labels = [lbl for lbl, _ in edo_physics_list]
f_limits_by_edo_model_R90 = {}

print(f"Computing {CL_LABEL} limits on f_EDO (method: {ANALYSIS_METHOD}), combining "
      f"{len(FIELDS)} field(s), at "
      f"{len(massGrid)} mass points, for "
      f"{n_edo} EDO type(s) {_edo_labels} x "
      f"{n_models} Milky-Way halo model(s) {mw_plain_labels} x {n_radii} R90 value(s) "
      f"{[float(r) for r in R90_list_solar_radii]} = {n_edo*n_models*n_radii} curve(s)\n"
      f"(this can take a while... progress is printed below)...\n")

# Main loop: one limit curve per (EDO type, Milky-Way model, R90). The globals
# profile_physics, CURRENT_MODEL_FIELD_POPS and R90_kpc select what dGammaDtE() evaluates.
_t_start = time.time()
for e_idx, (_edo_label, _edo_physics) in enumerate(edo_physics_list):
    profile_physics = _edo_physics
    print(f"\n#####  EDO type: {_edo_label}  [{e_idx + 1}/{n_edo}]  #####")
    _results_this_edo = {}

    for m_idx, _mlabel in enumerate(mw_plain_labels):
        CURRENT_MODEL_FIELD_POPS = lens_populations_by_model[m_idx]
        print(f"=== Milky-Way halo model {_mlabel}  [{m_idx+1}/{n_models}]: "
              f"{_describe_mw_spec(mw_specs[m_idx])} ===")
        _results_this_model = {}

        for r_idx, R90_val in enumerate(R90_list_solar_radii):
            R90_kpc = float(R90_val) * RSUN_KPC
            print(f"--- EDO {_edo_label}, model {_mlabel}, R90 = {R90_val:g} solar radii  "
                  f"[{r_idx+1}/{n_radii}] ---")

            f_limits = np.empty_like(massGrid)
            for i, M in enumerate(massGrid):
                _t0 = time.time()
                f_limits[i] = FindLimit(M)
                print(f"  [{i+1:>3d}/{len(massGrid)}] M = {M:.4e} Msun  ->  "
                      f"f_EDO({CL_LABEL}) = {f_limits[i]:.4e}   "
                      f"({time.time()-_t0:.1f}s)")

            _results_this_model[R90_val] = f_limits.copy()
        _results_this_edo[_mlabel] = _results_this_model
    f_limits_by_edo_model_R90[_edo_label] = _results_this_edo

# Restore the first configuration
profile_physics = edo_physics_list[0][1]
CURRENT_MODEL_FIELD_POPS = lens_populations_by_model[0]

_radius_word = "radius" if n_radii == 1 else "radii"
_model_word = "model" if n_models == 1 else "models"
_edo_word = "type" if n_edo == 1 else "types"
print(f"\nDone with {n_edo} EDO {_edo_word} x {n_models} {_model_word} x {n_radii} {_radius_word} "
      f"in {(time.time()-_t_start)/60:.1f} minutes.")

# =============================================================================
# Save each (EDO type, Milky-Way model, R90) curve as its own text file
# =============================================================================
if curve_save_prefix is not None:
    def _sanitize_for_filename(s):
        return "".join(c if (c.isalnum() or c in "._-") else "_" for c in str(s))

    _n_saved = 0
    for _edo_label in _edo_labels:
        for _mlabel in mw_plain_labels:
            for R90_val in R90_list_solar_radii:
                _curve = f_limits_by_edo_model_R90[_edo_label][_mlabel][R90_val]
                _fname = (f"{curve_save_prefix}__{_sanitize_for_filename(_edo_label)}__"
                          f"model={_sanitize_for_filename(_mlabel)}__"
                          f"R90={_sanitize_for_filename(f'{float(R90_val):g}')}.txt")
                with open(_fname, "w") as _fh:
                    _fh.write("#Columns: M [M_sun], f_EDO\n")
                    for _M, _f in zip(massGrid, _curve):
                        _fh.write(f"{_M} {_f}\n")
                _n_saved += 1
    print(f"\n-> Saved {_n_saved} curve file(s) with prefix '{curve_save_prefix}' to "
          f"{os.getcwd()}.")


# =============================================================================
# Plot the constraint — dark object mass vs. CL dark-matter fraction
# =============================================================================
# Colour encodes R90 and line style the Milky-Way model; one panel per EDO type.

_color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
_LINESTYLES = ['-', '--', '-.', ':', (0, (5, 1)), (0, (3, 1, 1, 1)), (0, (1, 1)), (0, (5, 2, 1, 2, 1, 2))]
_multi_model, _multi_radius, _multi_edo = n_models > 1, n_radii > 1, n_edo > 1
_R90_vals = list(R90_list_solar_radii)
_ls = lambda m: _LINESTYLES[m % len(_LINESTYLES)]
_col = lambda r: _color_cycle[r % len(_color_cycle)]

_radius_handles = [Line2D([], [], color=_col(r), linestyle='-', linewidth=2,
                          label=fr'$R_{{90}} = {R90_val:g}\ R_\odot$')
                   for r, R90_val in enumerate(_R90_vals)]
_model_handles = [Line2D([], [], color=('0.2' if _multi_radius else _col(0)), linestyle=_ls(m),
                         linewidth=2, label=mw_legend_labels[m])
                  for m in range(n_models)]

_fixed = []
if not _multi_edo:
    _fixed.append(_edo_labels[0])
if not _multi_model:
    _fixed.append('Milky Way halo model ' + mw_legend_labels[0])
if not _multi_radius:
    _fixed.append(fr'$R_{{90}} = {_R90_vals[0]:g}\ R_\odot$')
_suptitle = f'{CL_LABEL} Constraint on Extended Dark Object Dark-Matter Fraction'
if _fixed:
    _suptitle += '\n' + ';  '.join(_fixed)

ASSUMED_DISPLAY_WIDTH_IN = 6.3   # inches; a typical one-column article \textwidth
TARGET_CAPTION_FONT_PT = 10      # match this to your own document's caption size
_axis_fontsize = lambda fig_width_in: TARGET_CAPTION_FONT_PT * fig_width_in / ASSUMED_DISPLAY_WIDTH_IN

# Single EDO type: one axis with legends at the side
if not _multi_edo:
    fig, ax = plt.subplots(figsize=(8, 6))
    for m_idx, _mlabel in enumerate(mw_plain_labels):
        for r_idx, R90_val in enumerate(_R90_vals):
            _curve = f_limits_by_edo_model_R90[_edo_labels[0]][_mlabel][R90_val]
            _valid = np.isfinite(_curve) & (_curve > 0)
            ax.loglog(massGrid[_valid], _curve[_valid], linestyle=_ls(m_idx), linewidth=2, color=_col(r_idx))

    _fs = _axis_fontsize(8)
    ax.set_xlabel(r'$M\ (M_\odot)$', fontsize=_fs)
    ax.set_ylabel(r'$f_{\mathrm{DM}}$', fontsize=_fs)
    ax.tick_params(labelsize=_fs * 0.8)
    ax.set_title(_suptitle, fontsize=11)
    ax.grid(True, which='both', linestyle='--', alpha=0.6)
    ax.set_ylim(top=1.2)

    _sec_ax = ax.secondary_xaxis('top', functions=(lambda M: M * MSUN_TO_G, lambda Mg: Mg / MSUN_TO_G))
    _sec_ax.set_xlabel(r'$M\ (\mathrm{g})$', fontsize=_fs * 0.9)
    _sec_ax.tick_params(labelsize=_fs * 0.72)

    if _multi_model and _multi_radius:
        _leg1 = ax.legend(handles=_radius_handles, title='Object radius (colour)',
                          loc='upper left', bbox_to_anchor=(1.02, 1.0), fontsize=8.5, title_fontsize=9)
        ax.add_artist(_leg1)
        fig.canvas.draw()
        _bb = _leg1.get_window_extent().transformed(ax.transAxes.inverted())
        ax.legend(handles=_model_handles, title='Milky Way halo model (line style)',
                  loc='upper left', bbox_to_anchor=(1.02, _bb.y0 - 0.05), fontsize=8.5, title_fontsize=9)
    elif _multi_model:
        ax.legend(handles=_model_handles, title='Milky Way halo model', loc='best', fontsize=8.5, title_fontsize=9)
    else:
        ax.legend(handles=_radius_handles, loc='best')
    fig.tight_layout()

# Several EDO types: a grid of panels with shared axes and legend rows below
else:
    _n_cols = min(2, n_edo)
    _n_rows = -(-n_edo // _n_cols)   # ceil division
    _fs = _axis_fontsize(6.0 * _n_cols)   # font size scaled to the figure width

    _model_handles_wrapped = [
        Line2D([], [], color=h.get_color(), linestyle=h.get_linestyle(), linewidth=2,
               label=h.get_label().replace(', ', '\n'))
        for h in _model_handles
    ]
    _radius_lines = 1
    _model_lines = max((lab.count(', ') + 1 for lab in mw_legend_labels), default=1)

    _legend_fontsize = _fs * 0.8          # matches the scaling already used for tick labels
    _legend_title_fontsize = _fs * 0.85
    _row_h_in = lambda n_lines: 0.40 + (_fs / 72.0) * 1.35 * n_lines   # 0.40in: padding + title row

    _n_legend_rows = int(_multi_model) + int(_multi_radius)
    _legend_gap_in, _bottom_pad_in = 0.08, 0.12
    _radius_row_h_in = _row_h_in(_radius_lines) if _multi_radius else 0.0
    _model_row_h_in = _row_h_in(_model_lines) if _multi_model else 0.0
    _legend_rows_h_in = _radius_row_h_in + _model_row_h_in
    _fig_h = (4.6 * _n_rows + _legend_rows_h_in
              + _legend_gap_in * max(0, _n_legend_rows - 1) + _bottom_pad_in + 0.55)
    fig, _axes = plt.subplots(_n_rows, _n_cols, figsize=(6.0 * _n_cols, _fig_h),
                              squeeze=False, sharex=True, sharey=True)
    _axes = _axes.ravel()
    for _extra_ax in _axes[n_edo:]:
        fig.delaxes(_extra_ax)         # e.g. the 4th slot of a 2x2 grid when n_edo=3
    _axes = _axes[:n_edo]

    _bottom_row_for_col = {}
    for _idx in range(n_edo):
        _r, _c = _idx // _n_cols, _idx % _n_cols
        _bottom_row_for_col[_c] = max(_bottom_row_for_col.get(_c, -1), _r)

    for e_idx, _edo_label in enumerate(_edo_labels):
        ax = _axes[e_idx]
        _r, _c = e_idx // _n_cols, e_idx % _n_cols
        for m_idx, _mlabel in enumerate(mw_plain_labels):
            for r_idx, R90_val in enumerate(_R90_vals):
                _curve = f_limits_by_edo_model_R90[_edo_label][_mlabel][R90_val]
                _valid = np.isfinite(_curve) & (_curve > 0)
                ax.loglog(massGrid[_valid], _curve[_valid], linestyle=_ls(m_idx), linewidth=2, color=_col(r_idx))

        ax.text(0.04, 0.05, _edo_label, transform=ax.transAxes, fontsize=10.5,
                fontweight='bold', va='bottom', ha='left', color='0.15',
                bbox=dict(facecolor='white', alpha=0.78, edgecolor='none', boxstyle='round,pad=0.22'))
        ax.grid(True, which='both', linestyle='--', alpha=0.6)
        ax.set_ylim(top=1.2)

        if _r == _bottom_row_for_col[_c]:
            ax.set_xlabel(r'$M\ (M_\odot)$', fontsize=_fs)
            ax.tick_params(labelbottom=True, labelsize=_fs * 0.8)
        else:
            ax.tick_params(labelbottom=False)
        if _c == 0:
            ax.set_ylabel(r'$f_{\mathrm{DM}}$', fontsize=_fs)
            ax.tick_params(labelleft=True, labelsize=_fs * 0.8)
        else:
            ax.tick_params(labelleft=False)
        if _r == 0:
            _sec_ax = ax.secondary_xaxis('top', functions=(lambda M: M * MSUN_TO_G, lambda Mg: Mg / MSUN_TO_G))
            _sec_ax.set_xlabel(r'$M\ (\mathrm{g})$', fontsize=_fs * 0.9)
            _sec_ax.tick_params(labelsize=_fs * 0.72)

    fig.suptitle(_suptitle, fontsize=12, y=0.995)

    _top_of_legends_in = _bottom_pad_in + _legend_rows_h_in + _legend_gap_in * max(0, _n_legend_rows - 1)
    _top_of_legends = _top_of_legends_in / _fig_h
    fig.tight_layout(rect=[0, _top_of_legends + 0.01, 1, 0.965])

    fig.canvas.draw()
    _left_col_axes = [_axes[i] for i in range(n_edo) if (i % _n_cols) == 0]
    _right_col_axes = [_axes[i] for i in range(n_edo) if (i % _n_cols) == (_n_cols - 1)]
    _LEGEND_X0 = min(a.get_position().x0 for a in _left_col_axes)
    _LEGEND_W = max(a.get_position().x1 for a in _right_col_axes) - _LEGEND_X0

    _row_defs = []
    if _multi_model and _multi_radius:
        _row_defs = [(_radius_handles, 'Object radius', _radius_row_h_in),
                     (_model_handles_wrapped, 'Milky Way halo model', _model_row_h_in)]
    elif _multi_model:
        _row_defs = [(_model_handles_wrapped, 'Milky Way halo model', _model_row_h_in)]
    elif _multi_radius:
        _row_defs = [(_radius_handles, 'Object radius', _radius_row_h_in)]

    _cursor_in = _bottom_pad_in
    _placements = []
    for (_handles, _title, _row_h_in_val) in reversed(_row_defs):
        _placements.append((_handles, _title, _cursor_in, _row_h_in_val))
        _cursor_in += _row_h_in_val + _legend_gap_in
    _placements.reverse()

    for (_handles, _title, _y_in, _row_h_in_val) in _placements:
        fig.legend(handles=_handles, title=_title, loc='lower center',
                  bbox_to_anchor=(_LEGEND_X0, _y_in / _fig_h, _LEGEND_W, _row_h_in_val / _fig_h),
                  mode='expand', ncol=len(_handles),
                  fontsize=_legend_fontsize, title_fontsize=_legend_title_fontsize, frameon=True)


if plot_filename is not None:
    try:
        fig.savefig(plot_filename, dpi=200, bbox_inches='tight', facecolor='white')
        print(f"Plot saved to: {os.path.abspath(plot_filename)}")
    except Exception as _e:        # a failed save must never cost you the computed results
        print(f"[warning] could not save the plot to '{plot_filename}': {_e}\n"
              f"The results are still in memory; you can save the figure yourself with "
              f"fig.savefig('name.png', bbox_inches='tight').")
plt.show()

# Summary: minimum of each curve
print()
for _edo_label in _edo_labels:
    for _mlabel in mw_plain_labels:
        for R90_val in _R90_vals:
            _curve = f_limits_by_edo_model_R90[_edo_label][_mlabel][R90_val]
            _valid = np.isfinite(_curve) & (_curve > 0)
            _prefix = (f"[EDO {_edo_label}] " if _multi_edo else "") + (f"[model {_mlabel}] " if _multi_model else "")
            if not np.any(_valid):
                print(f"{_prefix}R90 = {R90_val:g} Rsun:  no valid (finite, positive) points")
                continue
            print(f"{_prefix}R90 = {R90_val:g} Rsun:  minimum f_EDO({CL_LABEL}) = {np.min(_curve[_valid]):.4e} "
                  f"at M = {massGrid[_valid][np.argmin(_curve[_valid])]:.4e} Msun")


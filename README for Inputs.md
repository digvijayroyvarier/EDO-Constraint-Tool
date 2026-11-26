# Input data

Example inputs for the EDO microlensing constraint tool, one folder per survey and source population. Every file starts with `#` comment lines describing its contents, units, source and the prompt it answers.

| File | Contents | Tool prompt |
|---|---|---|
| `EROS2_LMC/efficiency.txt` | EROS-2 LMC detection efficiency vs $t_E$ (150 points) | Step 5: file path |
| `EROS2_SMC/efficiency.txt` | EROS-2 SMC detection efficiency vs $t_E$ (151 points) | Step 5: file path |
| `OGLE-IV_LMC/efficiency.txt` | OGLE-IV LMC detection efficiency vs $t_E$ (129 points) | Step 5: file path |
| `OGLE-IV_LMC/foreground.txt` | OGLE-IV LMC expected foreground events per 0.1-dex $t_E$ bin (33 points) | Step 6: `n` to the default-table question, then file path; histogram `y`, bin width `0.1` |
| `OGLE-IV_LMC/observed_tE.txt` | $t_E$ of the 13 observed OGLE-IV LMC events | Step 4, method L: paste the list |
| `OGLE-IV_MW/efficiency.txt` | OGLE-IV Milky Way detection efficiency vs $t_E$ (25 points) | Step 5: file path |
| `OGLE-IV_MW/bin_edges.txt` | 26 $t_E$ bin edges (25 bins, 0.1–300 days) | Step 4, method B: paste the list |
| `OGLE-IV_MW/foreground_bin_counts.txt` | Foreground counts in those 25 bins | Step 4, method B: answer `direct`, paste the list |

## Formats

- **Tables** (`efficiency.txt`, `foreground.txt`): two whitespace-separated columns, $t_E$ in days and the tabulated value. Enter the file's path at the table prompt. For the background (Step 6), first answer `n` when asked whether to use the default OGLE-IV/LMC table.
- **Lists** (`observed_tE.txt`, `bin_edges.txt`, `foreground_bin_counts.txt`): the last line is a Python list. Copy that line and paste it at the prompt.

All timescales are in days. `foreground.txt` is a histogram (expected events per 0.1-dex bin in $\log_{10} t_E$), not a density. `foreground_bin_counts.txt` and `efficiency.txt` in `OGLE-IV_MW/` use the bins defined by `bin_edges.txt`, in the same order.

## Sources

- `OGLE-IV_LMC/foreground.txt`: digitized from Mróz et al. (2024), Nature 632, 749, Extended Data Fig. 6b.
- `OGLE-IV_LMC/observed_tE.txt`: statistical sample of Mróz et al. (2024), ApJS, doi:10.3847/1538-4365/ad452e.
- Per-field OGLE-IV LMC efficiencies and star counts: see the next section.
- Other files: see the `Source:` line in each file's header.

## OGLE-IV LMC per-field data (not included here)

The source-star counts and efficiencies of Mróz et al. (2024, ApJS 273, 4) are not redistributed in this repository. Download them from the archived copy at doi:10.5281/zenodo.10879576 (mirror: https://www.astrouw.edu.pl/ogle/ogle4/LMC_OPTICAL_DEPTH/):

- `table3.txt`: error-bar coefficients per subfield, listed for OGLE-III data only where a subfield has OGLE-III photometry
- `table8.txt`: microlensing source stars per subfield, corrected for blending and Milky Way foreground stars
- `eff_o3_o4.tar.gz`: efficiencies of the 43 fields observed during OGLE-III and OGLE-IV
- `eff_o4.tar.gz`: efficiencies of the 195 fields observed during OGLE-IV only

The efficiencies are per source star brighter than I = 22 mag, so they must be paired with the source counts of `table8.txt`, not the raw star counts of `table1.txt`. `OGLE-IV_LMC/make_source_counts.py` combines `table3.txt` and `table8.txt` into one per-field file:

```
python make_source_counts.py table3.txt table8.txt source_counts.txt
```

Each subfield (one of the 32 CCD chips of a field) has its corrected I < 22 source count assigned to the OGLE-III+IV baseline if `table3.txt` lists OGLE-III coefficients for it, and to the OGLE-IV-only baseline otherwise; the counts are then summed per field (78.682 million in total, as in the paper). This chip-level division reproduces the per-field OGLE-III+IV and OGLE-IV-only star counts of `table1.txt`, including in the 25 fields on the edge of the OGLE-III footprint.

At Step 2 of the tool, choose `per_field_bulk`, then answer:

| Prompt | Answer |
|---|---|
| Two sub-populations with different observing baselines? | `y` |
| Long-baseline star-count column | `2` (default) |
| Long-baseline exposure (days) | `7000` (default) |
| Short-baseline star-count column | `3` (default) |
| Short-baseline exposure (days) | `4000` (default) |
| Star-count file | path to `source_counts.txt` |
| Multiplier | `1` (default) |
| Field-ID column | `first` (default) |
| Long-baseline efficiencies | path to `eff_o3_o4.tar.gz` |
| Short-baseline efficiencies | path to `eff_o4.tar.gz` |

The exposures are the time windows over which Mróz et al. drew the peak times of their simulated events (JD 2,452,000–2,459,000 and 2,455,000–2,459,000), as their efficiencies require. The tool reports fields with a zero count in one baseline as having no efficiency file in that baseline; these contribute no events and can be ignored. Ten outer fields have no source stars at all, because all their stars are Milky Way foreground.

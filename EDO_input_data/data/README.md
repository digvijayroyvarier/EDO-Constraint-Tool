# Input data

Example inputs for the EDO microlensing constraint tool, one folder per survey and source population. Every file starts with `#` comment lines describing its contents, units, source and the prompt it answers.

| File | Contents | Tool prompt |
|---|---|---|
| `EROS2_LMC/efficiency.txt` | EROS-2 LMC detection efficiency vs $t_E$ (150 points) | Step 5: file path |
| `EROS2_SMC/efficiency.txt` | EROS-2 SMC detection efficiency vs $t_E$ (151 points) | Step 5: file path |
| `OGLE-IV_LMC/efficiency.txt` | OGLE-IV LMC detection efficiency vs $t_E$ (129 points) | Step 5: file path |
| `OGLE-IV_LMC/foreground.txt` | OGLE-IV LMC expected foreground events per 0.1-dex $t_E$ bin (33 points) | Step 6: n to the default-table question, then file path; histogram y, bin width 0.1 |
| `OGLE-IV_LMC/observed_tE.txt` | $t_E$ of the 13 observed OGLE-IV LMC events | Step 4, method L: paste the list |
| `OGLE-IV_MW/efficiency.txt` | OGLE-IV Milky Way detection efficiency vs $t_E$ (25 points) | Step 5: file path |
| `OGLE-IV_MW/bin_edges.txt` | 26 $t_E$ bin edges (25 bins, 0.1–300 days) | Step 4, method B: paste the list |
| `OGLE-IV_MW/foreground_bin_counts.txt` | Foreground counts in those 25 bins | Step 4, method B: answer `direct`, paste the list |

## Formats

- **Tables** (`efficiency.txt`, `foreground.txt`): two whitespace-separated columns, $t_E$ in days and the tabulated value. Enter the file's path at the prompt.
- **Lists** (`observed_tE.txt`, `bin_edges.txt`, `foreground_bin_counts.txt`): the last line is a Python list. Copy that line and paste it at the prompt.

All timescales are in days. `foreground.txt` is a histogram (expected events per 0.1-dex bin in $\log_{10} t_E$), not a density. `foreground_bin_counts.txt` and `efficiency.txt` in `OGLE-IV_MW/` use the bins defined by `bin_edges.txt`, in the same order.

## OGLE-III and IV LMC per-field data
Download from Mróz et al. (2024, ApJS 273, 4), archived at doi:10.5281/zenodo.10879576
(mirror: https://www.astrouw.edu.pl/ogle/ogle4/LMC_OPTICAL_DEPTH/):
`table1.txt`, `eff_o3_o4.tar.gz`, `eff_o4.tar.gz`.

At Step 2, choose `per_field_bulk`, then answer:
- two sub-populations with different baselines: `y`
- long-baseline star-count column: `7`, exposure: `6839`
- short-baseline star-count column: `8`, exposure: `3546`
- star-count file: path to `table1.txt`; multiplier: `1000`; field-ID column: `first`
- long-baseline efficiencies: path to `eff_o3_o4.tar.gz`
- short-baseline efficiencies: path to `eff_o4.tar.gz`

## Sources

- `OGLE-IV_LMC/foreground.txt`: digitized from Mróz et al. (2024), Nature 632, 749, Extended Data Fig. 6b.
- `OGLE-IV_LMC/observed_tE.txt`: statistical sample of Mróz et al. (2024), ApJS, doi:10.3847/1538-4365/ad452e.
- Per-field OGLE-IV LMC efficiencies and star counts, for the tool's bulk field mode, are published by OGLE at https://www.astrouw.edu.pl/ogle/ogle4/LMC_OPTICAL_DEPTH/.
- Other files: see the `Source:` line in each file's header.

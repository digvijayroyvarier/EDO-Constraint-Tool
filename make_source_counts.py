"""Per-field OGLE-IV LMC microlensing source counts for the EDO constraint tool.

Combines two tables of Mroz et al. (2024, ApJS 273, 4), both given per subfield
(one of the 32 CCD chips of an OGLE-IV field, e.g. LMC501.08):
  table3.txt  error-bar coefficients; those for OGLE-III data are present only
              for subfields with OGLE-III photometry
  table8.txt  source stars, corrected for blending and for Milky Way foreground

Each subfield's corrected count of sources brighter than I = 22 mag (the
magnitude limit of the published efficiency files) is assigned to the
OGLE-III+IV (long) baseline if it has OGLE-III coefficients, and to the
OGLE-IV-only (short) baseline otherwise, then summed per field.

Usage:  python make_source_counts.py [table3.txt] [table8.txt] [output.txt]
"""
import sys
from collections import defaultdict

table3 = sys.argv[1] if len(sys.argv) > 1 else "table3.txt"
table8 = sys.argv[2] if len(sys.argv) > 2 else "table8.txt"
output = sys.argv[3] if len(sys.argv) > 3 else "source_counts.txt"

def data_rows(path):
    with open(path) as f:
        return [line.split() for line in f if line.strip() and not line.startswith("#")]

# table3: subfield, then (gamma, epsilon) for OGLE-III if available, then for OGLE-IV
has_ogle3 = {r[0]: len(r) == 5 for r in data_rows(table3)}

# table8: subfield, RA, Dec, then total/corrected pairs for I < 21, 21.5, 22 mag
n_corr = {r[0]: int(r[8]) for r in data_rows(table8)}       # n22_corr

if set(has_ogle3) != set(n_corr):
    sys.exit(f"Subfield lists differ between {table3} and {table8}.")

counts = defaultdict(lambda: [0, 0])                         # field -> [long, short]
for subfield, n in n_corr.items():
    counts[subfield.split(".")[0]][0 if has_ogle3[subfield] else 1] += n

with open(output, "w") as f:
    f.write("# OGLE-IV LMC microlensing source stars per field (I < 22 mag, corrected for\n"
            "# blending and Milky Way foreground), from table8 of Mroz et al. (2024, ApJS 273, 4);\n"
            "# subfields with OGLE-III data (their table3) are counted in the OGLE-III+IV column.\n"
            "# Columns: field  N_OGLE-III+IV  N_OGLE-IV_only\n")
    for field in sorted(counts):
        f.write(f"{field}  {counts[field][0]:8d}  {counts[field][1]:8d}\n")

n_long = sum(v[0] for v in counts.values())
n_short = sum(v[1] for v in counts.values())
print(f"Wrote {output}: {len(counts)} fields, {(n_long + n_short) / 1e6:.3f} million source stars "
      f"({n_long / 1e6:.3f} M OGLE-III+IV, {n_short / 1e6:.3f} M OGLE-IV only).")

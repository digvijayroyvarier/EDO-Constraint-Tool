# Extended Dark Object Microlensing Constraint Calculator
Computes upper limits on the dark-matter fraction $f_{\mathrm{EDO}}$ in extended dark objects (EDOs) of a given mass profile (e.g. boson stars, ultra-compact minihalos, NFW or self-similar subhalos, uniform spheres, or point lenses) from gravitational microlensing survey data.

**Usage:** Run the code cell below and answer the prompts. Pressing Enter accepts the default shown, which corresponds to an OGLE-IV/LMC configuration.

## Workflow

0. **Statistical method:** (L) unbinned Poisson likelihood using the observed event timescales and a background model, with a frequentist ($\Delta\,2\ln\mathcal{L}=5.991$) or Bayesian (flat prior $f\in[0,10]$) 95% limit; (P) Poisson upper limit with zero background; (B) binned likelihood-ratio deviance against a binned foreground.
1. **EDO mass profile:** Built-in profile(s) or a custom $\rho(r)$, and one or more values of $R_{90}$ (radius enclosing 90% of the mass).
2. **Survey geometry:** Source fields $(D_S, l, b, N_\star)$ and exposure, entered as a table or read in bulk from a star-count file plus per-field efficiency files (optionally split into long- and short-baseline populations).
3. **Halos:** Milky-Way lens halo models S, B, C of Green (2017), or U which is a user-defined density (cored, cuspy, NFW, the contracted NFW of Cautun et al. 2020, or a custom expression) with a Maxwellian velocity distribution. Optionally, also the source's own halo.
4. **Observed data:** Event timescales $t_E$ (L), number of candidates and confidence level (P), or bin edges (B).
5. **Detection efficiency** $\varepsilon(t_E)$, per field.
6. **Background** $dN_{\mathrm{bkg}}/dt_E$ (L and B), as a density or a binned histogram.

The tool then tabulates the threshold impact parameter $u_T(\tau_M)$ for each profile, evaluates the event rate $d\Gamma/dt_E$ summed over fields, and computes the limit on a mass grid that is extended automatically until $f_{\mathrm{EDO}} \approx 1$ at both ends, for every combination of EDO type, Milky-Way model and $R_{90}$.

## Input file formats

- **Tables** ($[t_E, \varepsilon]$, $[t_E, dN/dt_E]$, field tables): a Python list literal, or a path to a whitespace- or comma-delimited text file.
- **Bulk star counts:** one row per field; the columns holding the field ID and star count are selected at the prompt.
- **Per-field efficiency:** a folder or archive (.zip, .tar, .tar.gz, ...) with one file per field: rows `log10(tE_min) log10(tE_max) efficiency`, and a header comment line containing `field <ID>` (otherwise the file name is used as the ID).

## Output

- A PNG of the limit versus mass (one panel per EDO type; colour encodes $R_{90}$, line style the Milky-Way model).
- Optionally, one text file per curve with columns `M [M_sun]` and `f_EDO`.

## References

Croon, McKeen & Raj (2020), arXiv:2002.08962 · Mróz et al. (2024), Nature, arXiv:2403.02386 · Mróz et al. (2024), ApJS 273, 4, arXiv:2403.02398 · Green (2017), arXiv:1705.10818 · Cautun et al. (2020), MNRAS · Evans (1993, 1994) · Alcock et al. (1995), ApJ 449, 28 · Niikura et al. (2019)


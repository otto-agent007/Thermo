# A finite local lattice-gas model for adsorption

```toml
id = "schneider-local-adsorption"
title = "Phase transitions in disordered mesoporous solids"
kind = "paper"
status = "asserted"
last_checked = 2026-10-20

[source]
url = "https://doi.org/10.1038/s41598-017-07406-2"
full_text = "https://pmc.ncbi.nlm.nih.gov/articles/PMC5543148/"
verified_text = "https://www.ebi.ac.uk/europepmc/webservices/rest/PMC5543148/fullTextXML"
authors = "Daniel Schneider, Daria Kondrashova and Rustem Valiullin"
published = "Scientific Reports 7, 7216 (2017)"
locator = "Methods: GCMC simulations of gas sorption, equations 22–23"

[[claims]]
id = "C1"
status = "asserted"
text = "Equation 22 specifies a simple-cubic binary-occupation lattice gas with nearest-neighbor attraction and local external fields."

[[claims]]
id = "C2"
status = "asserted"
text = "The printed Hamiltonian has +sum_i phi_i n_i; the stated field is phi_i=3 epsilon beside pore walls and zero elsewhere."
```

## Model convention

The published equation is

\[
H(n)=-\epsilon\sum_{\langle ij\rangle}n_i n_j+\sum_i\phi_i n_i,
\qquad n_i\in\{0,1\}.
\]

Equation 23 uses grand-canonical Metropolis occupancy changes. Thermo's explicit
grand energy is therefore defined as \(G=H-\mu\sum_i n_i\).
With the printed positive wall field, occupancy at a wall costs more energy:
the literal convention is **repulsive**. Do not silently change its sign to
obtain an attractive wall or describe that alteration as a reproduction.

## Relevance and limits

Our derived spin map, for \(n_i=(1+s_i)/2\) and energy
\(-\sum J_{ij}s_i s_j-\sum h_i s_i\), is
\(J_{ij}=\epsilon/4\) and
\(h_i=(\mu-\phi_i)/2+\epsilon z_i/4\), with actual boundary coordination \(z_i\).
This is an algebraic derivation, not a paper result about Thermo.

The proposed open 3×3×16 rectangular pore, chemical-potential points, coefficient
rounding and hardware placement are our model-class screen. They are not the
paper's geometry or a calibrated real-fluid workload. Published hysteresis and
scanning branches do not establish equilibrium accuracy for our sampler.

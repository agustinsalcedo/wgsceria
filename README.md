# Microkinetic Model for WGS on CeO<sub>2</sub>

![wgsceria](figures/wgsceria.png "Mechanism for the WGS reaction on CeO2(111)")

This repository contains a microkinetic model for the Water–Gas Shift (WGS) reaction on the CeO<sub>2</sub>(111) surface, originally developed during my PhD and used in a published study.  
It is not intended as a general-purpose tool, but rather as a study of a specific catalytic system.

---

## Requirements

- Python ≥ 3.8
- NumPy
- SciPy
- Matplotlib

---

## Model Description

The model includes:

- Elementary reaction network for WGS on CeO<sub>2</sub>(111).
- Thermodynamic consistency between forward and reverse rate constants.
- Adsorption/desorption corrections (2D gas approximation).

Main components of the model implemented in `wgsceria.py`:

- `get_k(T, P)` → equilibrium and rate constants
- `get_rates(theta, kf, kr)` → elementary reaction rates
- `get_odes(rate)` → time derivatives of surface coverages
- `int_conv(kf, kr, theta0)` → ODE solver (BDF method)
- `get_drc(K, kf, theta0, step_idx)` → degree of rate control (DRC) calculation

The Jupyter notebooks (`T_900K.ipynb` and `T_screening.ipynb`) provide example applications of the model.

For more details, see:

A. Salcedo and B. Irigoyen, “Unraveling the Origin of Ceria Activity in Water-Gas Shift by First-Principles Microkinetic Modeling”, *J. Phys. Chem. C*, 124, 14, 7823–7834, **2020**. https://doi.org/10.1021/acs.jpcc.0c00229

---

## License

MIT License

---

## Author

Agustin Salcedo

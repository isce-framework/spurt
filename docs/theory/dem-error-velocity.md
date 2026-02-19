# DEM Error and Velocity Estimation

Spurt estimates per-pixel velocity and DEM error as part of 3D phase unwrapping. This page derives the underlying model and explains how each component maps to code in the `spurt.links` module.

---

## The linear phase model

An interferometric phase observation between two SAR acquisitions can be decomposed as:

$$
\phi = \underbrace{\frac{4\pi}{\lambda} \, v \, \Delta t}_{\text{displacement}} \;+\; \underbrace{\frac{4\pi}{\lambda} \, \frac{B_\perp}{R \sin\theta} \, \varepsilon_{\text{DEM}}}_{\text{DEM error}} \;+\; \phi_{\text{atmo}} + \phi_{\text{noise}}
$$

where:

| Symbol | Meaning | Units |
|--------|---------|-------|
| $\lambda$ | Radar wavelength | m |
| $v$ | Line-of-sight velocity | mm/yr |
| $\Delta t$ | Temporal baseline | days |
| $B_\perp$ | Perpendicular baseline (secondary minus reference) | m |
| $R$ | Slant range distance | m |
| $\theta$ | Incidence angle | rad |
| $\varepsilon_{\text{DEM}}$ | DEM error | m |

We collect the two deterministic terms into a **design matrix** $\mathbf{A}$ of shape $(N_{\text{ifg}}, 2)$ and a parameter vector $\mathbf{x} = [v, \, \varepsilon_{\text{DEM}}]^T$:

$$
\boldsymbol{\phi} = \mathbf{A} \, \mathbf{x} + \boldsymbol{\phi}_{\text{atmo}} + \boldsymbol{\phi}_{\text{noise}}
$$

Each row of $\mathbf{A}$ corresponds to one interferogram:

$$
A_{k,0} = \frac{4\pi}{\lambda} \cdot \frac{\Delta t_k}{365.25 \cdot 1000}
\qquad
A_{k,1} = \frac{4\pi}{\lambda} \cdot \frac{\Delta B_{\perp,k}}{R \sin\theta}
$$

Column 0 gives **radians per mm/yr of velocity**; column 1 gives **radians per meter of DEM error** \[cite:Ferretti2001PermanentScatterers\].

---

## Why per-edge estimation

Rather than fitting the model to individual pixel phases, spurt fits to **spatial gradients** (phase differences between neighboring pixels on the Delaunay graph). For an edge connecting pixels $i$ and $j$:

$$
\Delta\phi_k^{(ij)} = \phi_k^{(j)} - \phi_k^{(i)} = \mathbf{A}_k \, (\mathbf{x}^{(j)} - \mathbf{x}^{(i)}) + \Delta\phi_{\text{atmo}} + \Delta\phi_{\text{noise}}
$$

On short baselines, the atmospheric contribution $\Delta\phi_{\text{atmo}}$ nearly cancels. This is the same data structure that MCF operates on, so per-edge estimation slots directly into the EMCF pipeline without any additional spatial processing.

---

## The design matrix in code

`spurt.links.build_design_matrix()` in `src/spurt/links/_design_matrix.py` constructs $\mathbf{A}$:

```python
from spurt.links import build_design_matrix

amat = build_design_matrix(
    ifg_edges=ifg_edges,     # (nifgs, 2) array of SLC index pairs
    dates=dates,             # datetime64[D] per SLC
    bperp_m=bperp_m,         # perpendicular baseline per SLC (meters)
    wavelength_m=0.031,      # Capella X-band: 3.1 cm
    slant_range_m=550_000,
    incidence_rad=0.61,
)
# amat.shape == (nifgs, 2)
# amat[:, 0] -> rad per mm/yr   (velocity sensitivity)
# amat[:, 1] -> rad per meter   (DEM error sensitivity)
```

The function iterates over interferogram pairs, computing temporal baseline $\Delta t$ and perpendicular baseline difference $\Delta B_\perp$ for each. The common factor $4\pi/\lambda$ is applied once, and the geometric DEM scaling $1/(R \sin\theta)$ is computed once for the scene.

---

## Temporal coherence as objective

The model parameters are estimated by maximizing **temporal coherence** \[cite:Ferretti2001PermanentScatterers\], defined as:

$$
\gamma = \frac{\left|\sum_{k=1}^{N} w_k \, \exp\!\bigl(j\,(\mathbf{A}_k \mathbf{x} - \phi_k)\bigr)\right|}{\sum_{k=1}^{N} w_k}
$$

where $w_k$ are per-interferogram weights and $\phi_k$ are the observed (wrapped) phase gradients. When the model perfectly explains the data, all residual phasors align and $\gamma = 1$. When the model is wrong, they scatter and $\gamma \to 0$.

For optimization with `scipy.minimize`, spurt implements the **negative** temporal coherence in `src/spurt/links/_common.py`:

```python
def neg_temporal_coherence(x, amat, b, wts):
    res = amat.dot(x) - b
    return -np.abs(np.sum(wts * np.exp(1j * res)))
```

A variant with analytic Jacobian (`neg_temporal_coherence_with_jacobian`) is also provided for gradient-based refinement.

---

## Grid search + Nelder-Mead

The temporal coherence surface is **non-convex** because the observed phases are wrapped. A local optimizer starting from a poor initial guess will find the wrong basin. Spurt uses a two-stage strategy via `GridSearchLinearModel` in `src/spurt/links/_grid_search.py`:

1. **Brute-force grid search** (`scipy.optimize.brute`): evaluate $\gamma$ on a coarse grid over user-defined parameter ranges. The ranges are specified as Python `slice` objects, e.g. `slice(-50, 50, 0.5)` for DEM error in meters. This finds the global basin.

2. **Nelder-Mead refinement**: starting from the grid optimum, `scipy.optimize.fmin` refines to sub-grid precision. The result is clipped to the search bounds to prevent aliasing.

This is conceptually different from \[cite:Pepe2006ExtensionMinimumCost\], which solves a full MCF problem for each parameter hypothesis. The per-link approach is simpler, parallelizes trivially, and avoids the need to define a spatial cost function over parameter space.

The solver processes links in batches via `estimate_model_many()`, distributing work across CPU cores with Python multiprocessing.

---

## Model-guided wrapping

After estimating per-edge parameters, the model prediction is used to **guide phase wrapping disambiguation**. This is the key integration point between the `links` module and the EMCF solver.

The function `phase_diff()` in `src/spurt/mcf/utils.py` computes:

$$
d = \phi_1 - \phi_0 - m
$$

$$
\text{phase\_diff} = m + d - 2\pi \left\lfloor \frac{d}{2\pi} \right\rceil
$$

where $m$ is the model prediction. Without a model ($m = 0$), this is standard wrapped phase differencing: the result lies in $(-\pi, \pi]$. With a model, the result lies in $(m - \pi, \, m + \pi]$. The gradient is wrapped **around the model prediction** rather than around zero.

In code:

```python
def phase_diff(z0, z1, model=0.0):
    d = z1 - z0 - model
    return model + d - np.round(d / (2 * np.pi)) * 2 * np.pi
```

This "flattening" means MCF only needs to resolve **residual** ambiguities---the small integer cycles left after accounting for velocity and DEM error. For interferograms with large perpendicular baselines, this can dramatically reduce the number of residues MCF must handle.

The integration point is `EMCFSolver.unwrap_gradients_in_time()` in `src/spurt/workflows/emcf/_solver.py` (lines 271--308). After estimating model parameters for a batch of links, it recomputes the spatial gradients with the model prediction:

```python
# Recompute gradients using model to guide wrapping
grad_space[:, i_start:i_end] = utils.phase_diff(
    wrap_data[:, inds[:, 0]],
    wrap_data[:, inds[:, 1]],
    model=model_pred,
)
```

---

## From edges to pixels

The per-edge estimated parameters (velocity gradient, DEM error gradient) are integrated to per-point values via **weighted least-squares** (WLS). Given the incidence matrix $\mathbf{D}$ of the spatial graph where each row represents an edge $i \to j$:

$$
\mathbf{D} \, \mathbf{v} \approx \mathbf{g}
$$

where $\mathbf{g}$ is the vector of per-edge gradients. This is solved by `scipy.sparse.linalg.lsqr` with weights $\sqrt{\gamma}$ (square root of temporal coherence), so that well-fitting edges contribute more to the solution.

The implementation in `src/spurt/workflows/emcf/_output.py` (function `_integrate_tile_link_params`, lines 323--395):

1. Builds a weighted incidence matrix: multiply each row by $\sqrt{\gamma_{\text{edge}}}$
2. Solves via LSQR for each parameter dimension (velocity, DEM error)
3. Centers the result by removing the median (the integration constant is arbitrary)
4. Computes mean coherence per point for quality assessment

The result is a per-pixel map of velocity (mm/yr) and DEM error (m), plus a coherence map indicating estimation reliability.

---

## References

- \[cite:Ferretti2001PermanentScatterers\]
- \[cite:Pepe2006ExtensionMinimumCost\]

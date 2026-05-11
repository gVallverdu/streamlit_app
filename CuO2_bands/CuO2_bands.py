#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Streamlit app — Band diagram of square lattices
Interactive comparison between:
  - Simple monoatomic square lattice (1 AO/cell)
  - CuO2 layer with second-neighbor O1-O2 interactions (beta_2)

Sliders: beta_1 and beta_2/beta_1
"""


import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from pathlib import Path

IMG_PATH = Path(__file__).parent / "assets" / "CuO2.png"

# ─────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Band Diagram - Square lattices",
    page_icon="atom",
    layout="centered",
)

col_txt, col_img = st.columns([2, 1])
with col_img:
    st.image(str(IMG_PATH), width="stretch")
    # st.image(img, width=300)
with col_txt:
    st.title("Band Diagram of Square Lattices")
    st.markdown(
        "### Effect of second neighbor\n"
        "Comparison between the band diagrams of a **simple monoatomic square lattice**"
        " and a single **CuO2 layer** with second-neighbor oxygen interactions."
    )
    # st.markdown("Description de la structure...")

# st.image(img, width=300)


# ─────────────────────────────────────────────────────────────
# Sidebar — parameters
# ─────────────────────────────────────────────────────────────
st.sidebar.header("Parameters")

st.sidebar.markdown("$\\alpha$ (eV)")
alpha = st.sidebar.number_input(
    "On-site energy",
    min_value=-5.0, max_value=0.0, value=-1.0, step=0.1,
    format="%.2f",
    help="On-site Huckel energy for all atoms (reference level)."
)

st.sidebar.markdown("$\\beta_1$ (eV)")
beta_1 = st.sidebar.slider(
    "Cu - O interaction",
    min_value=-1.0, max_value=-0.025, value=-0.5, step=0.025,
    format="%.2f",
    help="Hopping integral between a copper atom and an adjacent oxygen atom."
)

st.sidebar.markdown("$\\beta_2 / \\beta_1$ ratio")
ratio = st.sidebar.slider(
    "ratio O1 - O2 / Cu - O interaction",
    min_value=-0.8, max_value=0.0, value=-0.45, step=0.05,
    format="%.2f",
    help="Ratio of the second-neighbor O1-O2 interaction with respect to Cu-O."
)

beta_2 = ratio * beta_1

st.sidebar.markdown("---")
st.sidebar.markdown(f"""
***parameters***

* $\\alpha$ = {alpha:.3f} eV
* $\\beta_1$ = {beta_1:.3f} eV
* $\\beta_2$ = {beta_2:.3f} eV
"""
)

# ─────────────────────────────────────────────────────────────
# k-path
# ─────────────────────────────────────────────────────────────
def make_path(N: int = 300):
    t = np.linspace(0.5, 0, N, endpoint=False)
    kx_MG, ky_MG = t, t

    t = np.linspace(0, 0.5, N, endpoint=False)
    kx_GX, ky_GX = t, np.zeros(N)

    t = np.linspace(0, 0.5, N, endpoint=True)
    kx_XM, ky_XM = 0.5 * np.ones(N), t

    kx = np.concatenate([kx_MG, kx_GX, kx_XM])
    ky = np.concatenate([ky_MG, ky_GX, ky_XM])
    s  = np.arange(len(kx))
    ticks  = [0, N, 2 * N, 3 * N - 1]
    labels = ['$M$', r'$\Gamma$', '$X$', '$M$']
    return kx, ky, s, ticks, labels

# ─────────────────────────────────────────────────────────────
# Band functions
# ─────────────────────────────────────────────────────────────
def band_square(kx_n, ky_n, alpha, beta):
    """Simple square lattice: 
        E = alpha + 2*beta*[cos(2*pi*kx) + cos(2*pi*ky)]
    """
    return alpha + 2 * beta * (np.cos(2 * np.pi * kx_n) + np.cos(2 * np.pi * ky_n))

def band_CuO2(kx_n, ky_n, eps0, b1, b2):
    """CuO2 lowest band: 
        E = eps0 + 2*beta_1*[cos(2*pi*kx)+cos(2*pi*ky)] - 4*beta_2*cos(2*pi*kx)*cos(2*pi*ky)
    """
    cx = np.cos(2 * np.pi * kx_n)
    cy = np.cos(2 * np.pi * ky_n)
    return eps0 + 2 * b1 * cx + 2 * b1 * cy - 4 * b2 * cx * cy

# ─────────────────────────────────────────────────────────────
# Compute bands
# ─────────────────────────────────────────────────────────────
kx, ky, s, ticks, labels = make_path(N=300)

E_square = band_square(kx, ky, alpha, beta_1)

# Align Gamma point of CuO2 with that of simple lattice
# E_square(Gamma) = alpha + 4*beta_1
# E_CuO2(Gamma)   = eps0 + 4*beta_1 - 4*beta_2  ->  eps0 = alpha + 4*beta_2
eps0   = alpha + 4 * beta_2
E_CuO2 = band_CuO2(kx, ky, eps0, beta_1, beta_2)

# Fermi level: half-filled simple band (monovalent atom)
E_fermi = alpha  # E(Gamma) + E(M) / 2 = alpha, by symmetry

# ─────────────────────────────────────────────────────────────
# Key energy values for annotation
# ─────────────────────────────────────────────────────────────
def key_values(E, ticks):
    return {
        "M":     E[ticks[0]],
        "Gamma": E[ticks[1]],
        "X":     E[ticks[2]],
    }

# sq = key_values(E_square, ticks)
# cu = key_values(E_CuO2,   ticks)

# ─────────────────────────────────────────────────────────────
# Plot
# ─────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))

for t in ticks:
    ax.axvline(t, color='#aaaaaa', linewidth=0.7, linestyle='--', zorder=1)

ax.axhline(E_fermi, color='C3', linewidth=1.0, linestyle=':', zorder=2)

ax.plot(s, E_square, color='C0', linewidth=2,
        label='Simple square lattice', zorder=3)
ax.plot(s, E_CuO2, color='C1', linewidth=2, linestyle='--',
        label=f'CuO2 (lowest)', zorder=3)

ax.set_xlim(s[0], s[-1])
ax.set_ylim(-5, 3)
ax.set_xticks(ticks, labels=labels)
ax.set_xticklabels(labels, fontsize=13)
ax.set_ylabel('Energy (eV)', fontsize=12)

ax.legend(fontsize=12, bbox_to_anchor=(0.5, 1), ncol=2, loc="lower center")
ax.grid(axis='y', linestyle=':', linewidth=0.5, alpha=0.6)

plt.tight_layout()
st.pyplot(fig)
plt.close(fig)

# ─────────────────────────────────────────────────────────────
# Key values table
# ─────────────────────────────────────────────────────────────
# st.markdown("### Energy at high-symmetry points")

# col1, col2, col3 = st.columns(3)
# for col, pt in zip([col1, col2, col3], ['M', 'Gamma', 'X']):
#     col.metric(
#         label=f"Simple - {pt}",
#         value=f"{sq[pt]:.3f} eV"
#     )
# for col, pt in zip([col1, col2, col3], ['M', 'Gamma', 'X']):
#     delta = cu[pt] - sq[pt]
#     col.metric(
#         label=f"CuO2 - {pt}",
#         value=f"{cu[pt]:.3f} eV",
#         delta=f"{delta:+.3f} eV vs simple",
#     )

# ─────────────────────────────────────────────────────────────
# Equations reminder
# ─────────────────────────────────────────────────────────────
st.markdown(r"""
## Band expressions

### Reciprocal space - First Brillouin Zone

$$
\begin{aligned}
\vec{b}_1 &= \frac{2\pi}{a} \vec{i} &
\vec{b}_2 &= \frac{2\pi}{a} \vec{j} &
\vec{k} & = k_x \vec{b}_1 + k_y \vec{b}_2 &
k_x, k_y & \in \left[-\frac{1}{2} ; \left.\frac{1}{2}\right[\right.
\end{aligned}
$$

Take a look at [this page on EPFL](https://exoset.epfl.ch/gitrepository/constructing-brillouin-zones-of-the-2d-square-lattice)
for an example about the construction of the Brillouin zones of a square lattice.

### Simple square lattice

Here is the expression of the band energy for a square lattice with one
atom per unit cell, considering a single atomic orbital per unit cell.

$$
\epsilon = \alpha + 2 \beta \cos(2 \pi k_x) + 2 \beta \cos(2 \pi k_y)
$$

### CuO2 layer

Here is a simplified expression of the lowest band energy for a CuO2 layer 
including a second neighbor interaction $\beta_2$ between O1 and O2. This
is obtained considering only one AO per atom.

$$
\epsilon = \epsilon_0 + 2 \beta_1 \cos(2 \pi k_x) + 2 \beta_1 \cos(2 \pi k_y)
- 4 \beta_2 \cos(2 \pi k_x)\cos(2 \pi k_y)
$$

""")

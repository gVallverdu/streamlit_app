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
    page_title="1D Band Diagram",
    page_icon="atom",
    layout="centered",
)

st.title("1D Band Diagram")
st.markdown("""
Let's consider the band diagram of an infinit 1D chain with two sites per unit cell.
We consider the case of two identical (or not) sites with one, or two, interactions
in the context of Huckel theory.
"""
)



# ─────────────────────────────────────────────────────────────
# Sidebar — parameters
# ─────────────────────────────────────────────────────────────
st.sidebar.header("Parameters")

# alpha  = 0.
# beta   = -1.0
# beta_s = -0.8
# beta_d = -1.2
alpha_1 = 0.0

st.sidebar.markdown("$\\beta_1$ (eV)")
beta_1 = st.sidebar.slider(
    "First interaction",
    min_value=-1.0, max_value=-0.025, value=-0.5, step=0.025,
    format="%.2f",
)

st.sidebar.markdown("$\\beta_2 / \\beta_1$ ratio")
factor_beta = st.sidebar.slider(
    "second interaction",
    min_value=0.5, max_value=1.5, value=0.9, step=0.05,
    format="%.2f",
    help="Set ratio greater or lower than 1 to open the gap."
)

# st.sidebar.markdown("$\\alpha_1$ (eV)")
# alpha_1 = st.sidebar.slider(
#     "On site energy, atom 1",
#     min_value=-2.0, max_value=-0.025, value=-0.5, step=0.025,
#     format="%.2f",
# )

st.sidebar.markdown("$\\alpha_2 = \\alpha_1 + \\delta$ (eV)")
shift_alpha = st.sidebar.slider(
    "Shift energy",
    min_value=0.0, max_value=1.0, value=0.0, step=0.05,
    format="%.2f",
)

beta_2 = factor_beta * beta_1
alpha_2 = alpha_1 + shift_alpha
gap = np.sqrt((alpha_1 - alpha_2)**2 + 4 * (beta_1 - beta_2)**2)

st.sidebar.markdown("---")
st.sidebar.markdown(f"""
***parameters***

* $\\alpha_1$ = {alpha_1:.3f} eV
* $\\alpha_2$ = {alpha_2:.3f} eV
* $\\beta_1$ = {beta_1:.3f} eV
* $\\beta_2$ = {beta_2:.3f} eV
* gap = {gap:.3f} eV
"""
)


# ─────────────────────────────────────────────────────────────
# Band functions
# ─────────────────────────────────────────────────────────────
def band(k, alpha_1=-1.0, beta_1=-.5, alpha_2=None, beta_2=None, signe="+"):
    """ Compute the energy bands """
    if beta_2 is None:
        beta_2 = beta_1
    if alpha_2 is None:
        alpha_2 = alpha_1
    alpha_part = ((alpha_1 - alpha_2) / 2)**2
    beta_part = beta_1**2 + beta_2**2 + 2. * beta_1 * beta_2 * np.cos(2 * np.pi * k)
    sm = np.sqrt(alpha_part + beta_part)
    if signe == "+":
        e = (alpha_1 + alpha_2) / 2 + sm
    elif signe == "-":
        e = (alpha_1 + alpha_2) / 2 - sm
    else:
        e = None
    return e

# ─────────────────────────────────────────────────────────────
# Compute bands
# ─────────────────────────────────────────────────────────────
npts = 100
valk = np.linspace(-.5, .5, npts, endpoint=False)

plt.style.use("default")
fig, ax = plt.subplots(figsize=(8, 5))
ax.grid(True)
# fig.patch.set_facecolor('#FFFFFF')
# ax.set_facecolor('#FFFFFF')

ax.plot(valk, band(valk, alpha_1=alpha_1, beta_1=beta_1, signe="+"), "C0-", label="Simple case")
ax.plot(valk, band(valk, alpha_1=alpha_1, beta_1=beta_1, signe="-"), "C0-")

ax.plot(valk, band(valk, alpha_1, beta_1, alpha_2, beta_2, signe="+"), "C1-", label="General case")
ax.plot(valk, band(valk, alpha_1, beta_1, alpha_2, beta_2, signe="-"), "C1-")

sm = np.sqrt(((alpha_1 + alpha_2) / 2)**2 + (beta_1 - beta_2)**2)
emax_vb = (alpha_1 + alpha_2) / 2 - sm
emin_cb = (alpha_1 + alpha_2) / 2 + sm
ax.axhline(emax_vb, color="C3", linestyle="--", lw=1)
ax.axhline(emin_cb, color="C3", linestyle="--", lw=1)

if abs(beta_1 - beta_2) > 0.05:
    ax.annotate(
        "",
        xy=(0.05, beta_1 - beta_2),
        xytext=(0.05, beta_2 - beta_1),
        arrowprops=dict(arrowstyle="<|-|>", color="C3", lw=1)
    )
    ax.text(
        0.07, 0,
        "gap",
        color="C3",
        va="center",
        fontsize=10
    )

ax.set_ylabel("Energy (eV)")

ax.set_xlim(-.5, .5)
ax.set_ylim(-2.5, 2.5)
ax.set_xticks([-.5, 0, .5], labels=["X", r"$\Gamma$", "X"])

ax.legend(fontsize=12, bbox_to_anchor=(0.5, 1), ncol=2, loc="lower center")

plt.tight_layout()
st.pyplot(fig)
plt.close(fig)


# ─────────────────────────────────────────────────────────────
# Equations reminder
# ─────────────────────────────────────────────────────────────
st.markdown(r"""
## Band expressions

We consider a 1D infinite chain with to site per unit cell (a dimer). The
first typical exercise is the H2 dimer but also a polyacetylene polymer.

### Reciprocal space - First Brillouin Zone

$$
\begin{aligned}
\vec{a} & = a \vec{i} &
\vec{b} &= \frac{2\pi}{a} \vec{i} &
\vec{k} & = k \vec{b} &
k & \in \left[-\frac{1}{2} ; \left.\frac{1}{2}\right[\right.
\end{aligned}
$$

### Totaly symmetric case

Consider an infinite chain with two identical AO per unit cell and only one kind
of interaction. In this very simple case, the band energy in the context
of simple Huckel is given by:

$$
\epsilon(k) = \alpha \pm 2 \beta \cos(2 \pi k)
$$

This is the special case $\alpha_1=\alpha_2$ et $\beta_1 = \beta_2$.

### Two interactions

Consider now that the distance between the two sites is not the same. For
example it could be an alternating chain of single and double bonds.
The energy bands read:

$$
\epsilon(k) = \alpha \pm \sqrt{
        \beta_1^2 + \beta_2^2 + 2 \beta_1 \beta_2 \cos(2\pi k)
    }
$$

### Two $\alpha$ values

Consider now that the interaction is the same but the two atoms are different.

$$
\epsilon(k)
=
\frac{\alpha_1+\alpha_2}{2}
\pm
\sqrt{
\left(
\frac{\alpha_1-\alpha_2}{2}
\right)^2
+
4\beta^2\cos^2(\pi k)
}
$$

### General case

The general case corresponds to the diagonalization of this matrix
$$
H(k) = 
\begin{pmatrix}
\alpha_1 & \beta_1 + \beta_2 e^{2 i \pi k} \\
\beta_1 + \beta_2 e^{-2 i \pi k} & \alpha_2 \\
\end{pmatrix}
$$
The eigenvalues read:
$$
\epsilon(k)
=
\frac{\alpha_1+\alpha_2}{2}
\pm
\sqrt{
\left(
\frac{\alpha_1-\alpha_2}{2}
\right)^2
+
\beta_1^2+\beta_2^2
+
2\beta_1\beta_2\cos(2\pi k)
}
$$
""")

st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; color: gray; font-size: 0.85em; padding: 10px;">
        <strong>Germain Salvato Vallverdu</strong> &nbsp;|&nbsp;
        <a href="https://www.univ-pau.fr" target="_blank" style="color: gray; text-decoration: none;">Université de Pau et des Pays de l'Adour</a> &nbsp;|&nbsp;
        <a href="https://github.com/gVallverdu" target="_blank" style="color: gray; text-decoration: none;">
            <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="gray" viewBox="0 0 16 16">
                <path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38
                         0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13
                         -.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66
                         .07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15
                         -.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27
                         .68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12
                         .51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48
                         0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0 0 16 8c0-4.42-3.58-8-8-8z"/>
            </svg>
            &nbsp;gVallverdu
        </a>
    </div>
    """,
    unsafe_allow_html=True
)

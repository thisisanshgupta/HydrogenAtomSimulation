"""
File: hydrogen_plot.py
Description: Visualization layer for hydrogenic wavefunctions.
             Reads CSV data produced by the C++ backend (hydrogen_wavefunction.cpp)
             and renders probability density orbital diagrams + radial distributions.

Usage:
    1. Compile and run the C++ backend first:
           g++ -O3 -std=c++17 -o hydrogen_wavefunction hydrogen_wavefunction.cpp -lm
           ./hydrogen_wavefunction
    2. Then run this script:
           python hydrogen_plot.py
"""

import os
import glob
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import PowerNorm
from matplotlib.ticker import MultipleLocator

# ── Aesthetics ────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor":  "#0a0a12",
    "axes.facecolor":    "#0a0a12",
    "text.color":        "#e8e4d8",
    "axes.labelcolor":   "#e8e4d8",
    "axes.edgecolor":    "#2a2a3a",
    "xtick.color":       "#7a7a9a",
    "ytick.color":       "#7a7a9a",
    "grid.color":        "#1a1a2a",
    "font.family":       "monospace",
    "mathtext.fontset":  "cm",
})

CMAP_PSI  = "inferno"          # probability density
CMAP_RADIAL = "plasma"         # radial distribution colour cycle backup
ORBITAL_COLORS = [
    "#ff6b6b", "#ffa36b", "#ffd56b", "#6bffb8",
    "#6be4ff", "#6b8eff", "#d46bff", "#ff6bd4",
]

# Map (n,l,m) → spectroscopic label
def orbital_label(n, l, m):
    sub = ["s", "p", "d", "f", "g", "h"][l]
    return rf"$|{n}{sub}_{{m={m}}}\rangle$"

def latex_psi_label(n, l, m):
    return rf"$\psi_{{{n},{l},{m}}}$"

# ── Load all CSV files produced by the C++ backend ───────────────────────────

def discover_orbitals():
    """Return sorted list of (n,l,m) tuples for which both CSV files exist."""
    pattern = re.compile(r"psi_n(\d+)l(\d+)m(-?\d+)\.csv")
    orbitals = []
    for fn in sorted(glob.glob("psi_n*.csv")):
        mo = pattern.match(os.path.basename(fn))
        if mo:
            n, l, m = int(mo.group(1)), int(mo.group(2)), int(mo.group(3))
            rad_fn = f"radial_n{n}l{l}m{m}.csv"
            if os.path.exists(rad_fn):
                orbitals.append((n, l, m))
    return orbitals

def load_psi_slice(n, l, m, grid_pts=None):
    fn = f"psi_n{n}l{l}m{m}.csv"
    df = pd.read_csv(fn)
    gp = int(np.sqrt(len(df)))
    X  = df["x"].values.reshape(gp, gp)
    Z  = df["z"].values.reshape(gp, gp)
    P  = df["prob_density"].values.reshape(gp, gp)
    return X, Z, P

def load_radial(n, l, m):
    fn = f"radial_n{n}l{l}m{m}.csv"
    df = pd.read_csv(fn)
    return df["r_over_a_mu"].values, df["R_nl"].values, df["radial_prob"].values

# ── Plot 1: Orbital probability-density gallery ───────────────────────────────

def plot_orbital_gallery(orbitals, out_file="orbital_gallery.png"):
    ncols = 5
    nrows = int(np.ceil(len(orbitals) / ncols))
    fig_w = ncols * 3.4
    fig_h = nrows * 3.8 + 0.8

    fig = plt.figure(figsize=(fig_w, fig_h), facecolor="#0a0a12")
    fig.suptitle("Hydrogen Atom — Orbital Probability Densities  $|\\psi_{n,l,m}|^2$",
                 fontsize=15, color="#e8e4d8", y=0.99, fontfamily="monospace")

    gs = gridspec.GridSpec(nrows, ncols, figure=fig,
                           hspace=0.35, wspace=0.15,
                           left=0.04, right=0.96, top=0.94, bottom=0.04)

    for idx, (n, l, m) in enumerate(orbitals):
        row, col = divmod(idx, ncols)
        ax = fig.add_subplot(gs[row, col])

        X, Z, P = load_psi_slice(n, l, m)

        # Normalise to [0,1] for colour mapping
        vmax = np.percentile(P, 99.8)
        im = ax.imshow(
            P,
            origin="lower",
            extent=[X.min(), X.max(), Z.min(), Z.max()],
            cmap=CMAP_PSI,
            norm=PowerNorm(gamma=0.4, vmin=0, vmax=vmax),
            aspect="equal",
            interpolation="bilinear",
        )

        # Cross-hair at nucleus
        ax.axhline(0, color="#ffffff18", lw=0.5, ls="--")
        ax.axvline(0, color="#ffffff18", lw=0.5, ls="--")

        # Labels
        sub  = ["s","p","d","f","g","h"][l]
        title = f"$n={n},\\ l={l},\\ m={m}$   {n}{sub}"
        ax.set_title(title, fontsize=9, color="#c8c4b8", pad=3)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_edgecolor("#2a2a3a")

    fig.savefig(out_file, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print(f"  ✓  Saved {out_file}")
    plt.close(fig)

# ── Plot 2: Individual orbital detail (density + radial dist side by side) ────

def plot_orbital_detail(n, l, m, out_file=None):
    if out_file is None:
        out_file = f"orbital_detail_n{n}l{l}m{m}.png"

    X, Z, P   = load_psi_slice(n, l, m)
    r_au, Rnl, Pr = load_radial(n, l, m)

    sub   = ["s","p","d","f","g","h"][l]
    title = f"Hydrogen  $|{n},{l},{m}\\rangle$ — {n}{sub} orbital"

    fig = plt.figure(figsize=(12, 5.2), facecolor="#0a0a12")
    fig.suptitle(title, fontsize=14, color="#e8e4d8", y=1.01)

    gs = gridspec.GridSpec(1, 3, figure=fig, width_ratios=[1, 0.05, 1.1],
                           wspace=0.12, left=0.06, right=0.97)

    # ── Left: 2-D probability density ────────────────────────────────────────
    ax_orb = fig.add_subplot(gs[0])
    vmax = np.percentile(P, 99.8)
    im   = ax_orb.imshow(
        P,
        origin="lower",
        extent=[X.min(), X.max(), Z.min(), Z.max()],
        cmap=CMAP_PSI,
        norm=PowerNorm(gamma=0.35, vmin=0, vmax=vmax),
        aspect="equal",
        interpolation="bilinear",
    )
    ax_orb.axhline(0, color="#ffffff25", lw=0.6, ls="--")
    ax_orb.axvline(0, color="#ffffff25", lw=0.6, ls="--")
    ax_orb.set_xlabel(r"$x$ / $a_\mu$", fontsize=11)
    ax_orb.set_ylabel(r"$z$ / $a_\mu$", fontsize=11)
    ax_orb.set_title(r"$|\psi_{n,l,m}(x,0,z)|^2$  (y=0 plane)", fontsize=10, color="#a0a0c0")

    # Convert axis ticks to units of a_mu
    a_mu = 5.29465e-11   # m (reduced-mass for H)
    def to_au(val_m): return val_m / a_mu
    x_extent_au = [to_au(X.min()), to_au(X.max())]
    z_extent_au = [to_au(Z.min()), to_au(Z.max())]
    n_ticks = 5
    xt = np.linspace(x_extent_au[0], x_extent_au[1], n_ticks)
    zt = np.linspace(z_extent_au[0], z_extent_au[1], n_ticks)
    ax_orb.set_xticks(np.linspace(X.min(), X.max(), n_ticks))
    ax_orb.set_xticklabels([f"{v:.0f}" for v in xt], fontsize=8)
    ax_orb.set_yticks(np.linspace(Z.min(), Z.max(), n_ticks))
    ax_orb.set_yticklabels([f"{v:.0f}" for v in zt], fontsize=8)

    # Colourbar
    ax_cb = fig.add_subplot(gs[1])
    cb    = plt.colorbar(im, cax=ax_cb)
    cb.set_label(r"$|\psi|^2$ [a.u.]", fontsize=9, color="#a0a0c0")
    cb.ax.yaxis.set_tick_params(color="#7a7a9a", labelcolor="#7a7a9a", labelsize=7)

    # ── Right: Radial wavefunction + probability distribution ────────────────
    ax_r = fig.add_subplot(gs[2])
    color_R  = "#6be4ff"
    color_Pr = "#ff6b6b"

    ax_r.fill_between(r_au, Pr, alpha=0.25, color=color_Pr)
    ax_r.plot(r_au, Pr / max(Pr.max(), 1e-30),
              color=color_Pr, lw=2.0, label=r"$r^2|R_{n,l}|^2$ (norm.)")

    ax_r2 = ax_r.twinx()
    ax_r2.plot(r_au, Rnl, color=color_R, lw=1.5, ls="--", alpha=0.85,
               label=r"$R_{n,l}(r)$")
    ax_r2.axhline(0, color="#ffffff15", lw=0.5)
    ax_r2.set_ylabel(r"$R_{n,l}$  [m$^{-3/2}$]", fontsize=10, color=color_R)
    ax_r2.tick_params(axis="y", colors=color_R, labelsize=8)

    ax_r.set_xlabel(r"$r$ / $a_\mu$", fontsize=11)
    ax_r.set_ylabel(r"$P_{n,l}(r)$  (normalised)", fontsize=10, color=color_Pr)
    ax_r.tick_params(axis="y", colors=color_Pr, labelsize=8)
    ax_r.set_title("Radial distribution  &  $R_{n,l}(r)$", fontsize=10, color="#a0a0c0")
    ax_r.set_xlim(left=0)
    ax_r.grid(True, alpha=0.15)

    # Combined legend
    lines1, labs1 = ax_r.get_legend_handles_labels()
    lines2, labs2 = ax_r2.get_legend_handles_labels()
    ax_r.legend(lines1 + lines2, labs1 + labs2, fontsize=9,
                loc="upper right", facecolor="#12121f", edgecolor="#2a2a3a",
                labelcolor="#e8e4d8")

    fig.savefig(out_file, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print(f"  ✓  Saved {out_file}")
    plt.close(fig)

# ── Plot 3: Radial distribution multi-panel (all orbitals, grouped by n) ──────

def plot_radial_overview(orbitals, out_file="radial_overview.png"):
    ns = sorted(set(n for n, l, m in orbitals))
    nrows = len(ns)
    fig, axes = plt.subplots(nrows, 1, figsize=(11, 3.2 * nrows),
                             facecolor="#0a0a12", sharex=False)
    if nrows == 1:
        axes = [axes]

    fig.suptitle("Hydrogenic Radial Probability Distributions  $P_{n,l}(r) = r^2|R_{n,l}|^2$",
                 fontsize=13, color="#e8e4d8", y=1.005)

    for ax, n_val in zip(axes, ns):
        group = [(l, m) for (n, l, m) in orbitals if n == n_val]
        ax.set_facecolor("#0a0a12")
        ax.set_title(f"$n = {n_val}$", fontsize=10, color="#c0bca8", pad=2)
        ax.set_ylabel(r"$P_{n,l}(r)$  [a.u.]", fontsize=9)

        for ci, (l, m) in enumerate(group):
            r_au, Rnl, Pr = load_radial(n_val, l, m)
            color = ORBITAL_COLORS[ci % len(ORBITAL_COLORS)]
            sub   = ["s","p","d","f","g","h"][l]
            label = f"${n_val}{sub}$   $m={m}$"
            ax.fill_between(r_au, Pr, alpha=0.12, color=color)
            ax.plot(r_au, Pr, color=color, lw=1.8, label=label)

        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)
        ax.set_xlabel(r"$r$ / $a_\mu$", fontsize=9)
        ax.legend(fontsize=8.5, loc="upper right",
                  facecolor="#12121f", edgecolor="#2a2a3a", labelcolor="#e8e4d8",
                  ncol=min(len(group), 4))
        ax.grid(True, alpha=0.12)
        for sp in ax.spines.values(): sp.set_edgecolor("#2a2a3a")
        ax.tick_params(colors="#7a7a9a", labelsize=8)

    fig.tight_layout(rect=[0, 0, 1, 1])
    fig.savefig(out_file, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print(f"  ✓  Saved {out_file}")
    plt.close(fig)

# ── Plot 4: Quantum number comparison strips ───────────────────────────────────

def plot_n_strips(orbitals, out_file="n_comparison_strips.png"):
    """
    Horizontal strip layout: one row per n, orbitals of that n as columns.
    Shows how the wavefunction spreads as n increases.
    """
    from collections import defaultdict
    by_n = defaultdict(list)
    for n, l, m in orbitals:
        by_n[n].append((l, m))

    ns     = sorted(by_n)
    ncols  = max(len(v) for v in by_n.values())
    nrows  = len(ns)

    fig_h = 3.0 * nrows + 0.7
    fig_w = 3.0 * ncols + 0.5
    fig   = plt.figure(figsize=(fig_w, fig_h), facecolor="#0a0a12")
    fig.suptitle("Orbitals Grouped by Principal Quantum Number $n$",
                 fontsize=13, color="#e8e4d8")

    gs = gridspec.GridSpec(nrows, ncols, figure=fig,
                           hspace=0.25, wspace=0.08,
                           left=0.06, right=0.97, top=0.92, bottom=0.03)

    for ri, n_val in enumerate(ns):
        for ci, (l, m) in enumerate(by_n[n_val]):
            ax = fig.add_subplot(gs[ri, ci])
            X, Z, P = load_psi_slice(n_val, l, m)
            vmax = np.percentile(P, 99.8)
            ax.imshow(P, origin="lower",
                      extent=[X.min(), X.max(), Z.min(), Z.max()],
                      cmap=CMAP_PSI,
                      norm=PowerNorm(gamma=0.35, vmin=0, vmax=vmax),
                      aspect="equal", interpolation="bilinear")
            sub   = ["s","p","d","f","g","h"][l]
            ax.set_title(f"$n={n_val},l={l},m={m}$  {n_val}{sub}",
                         fontsize=7.5, color="#c8c4b8", pad=2)
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values(): sp.set_edgecolor("#1a1a2a")

            # Row label on first column
            if ci == 0:
                ax.set_ylabel(f"$n={n_val}$", fontsize=10, color="#a0bbd0", labelpad=4)

    fig.savefig(out_file, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print(f"  ✓  Saved {out_file}")
    plt.close(fig)

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    orbitals = discover_orbitals()
    if not orbitals:
        print("ERROR: No CSV files found. Run the C++ backend first:\n"
              "  g++ -O3 -std=c++17 -o hydrogen_wavefunction hydrogen_wavefunction.cpp -lm\n"
              "  ./hydrogen_wavefunction")
        return

    print(f"Found {len(orbitals)} orbital(s): {orbitals}\n")

    print("── Generating orbital gallery …")
    plot_orbital_gallery(orbitals, "orbital_gallery.png")

    print("\n── Generating radial distribution overview …")
    plot_radial_overview(orbitals, "radial_overview.png")

    print("\n── Generating n-comparison strips …")
    plot_n_strips(orbitals, "n_comparison_strips.png")

    print("\n── Generating individual detail plots …")
    for (n, l, m) in orbitals:
        plot_orbital_detail(n, l, m)

    print("\nDone. All figures written to disk.")

if __name__ == "__main__":
    main()

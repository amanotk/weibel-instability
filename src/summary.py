"""Summary plots for PIC-NIX simulation data."""

import numpy as np


def plot_field_snapshot(run, step, *, save=None):
    """Plot 2D snapshot of field components (mean over z) from a picnix Run.

    Parameters
    ----------
    run : picnix.Run
        Loaded simulation run object.
    step : int
        Simulation step to plot.
    save : str, optional
        Filepath to save the figure.
    """
    from matplotlib import pyplot as plt

    data = run.read_at("field", step)
    mime = run.config["parameter"]["mime"]
    sigma = run.config["parameter"]["sigma"]
    alpha = run.config["parameter"]["alpha"]
    ush = run.config["parameter"]["ush"]
    B0 = np.sqrt(sigma)
    Beq = np.sqrt(alpha * (1 - alpha) * ush**2 * mime)

    uf = data["uf"]
    bx = uf[..., 3].mean(axis=0) / Beq
    by = uf[..., 4].mean(axis=0) / Beq - B0 / Beq
    bz = uf[..., 5].mean(axis=0) / Beq

    vmax = max(np.abs(bx).max(), np.abs(by).max(), np.abs(bz).max())

    xc = run.xc / np.sqrt(mime)
    yc = run.yc / np.sqrt(mime)
    X, Y = np.meshgrid(xc, yc, indexing="xy")

    time = run.get_time_at("field", step) / np.sqrt(mime)

    fig = plt.figure(figsize=(10, 4))
    gs = fig.add_gridspec(
        1,
        3,
        left=0.06,
        right=0.90,
        bottom=0.15,
        top=0.83,
        wspace=0.15,
    )
    axs = [fig.add_subplot(gs[0, i]) for i in range(3)]

    labels = [
        r"$\delta B_x / B_{\mathrm{eq}}$",
        r"$\delta B_y / B_{\mathrm{eq}}$",
        r"$\delta B_z / B_{\mathrm{eq}}$",
    ]
    fields = [bx, by, bz]

    for ax, label, bfield in zip(axs, labels, fields, strict=True):
        ax.pcolormesh(X, Y, bfield, cmap="RdBu_r", vmin=-vmax, vmax=vmax, shading="nearest")
        ax.set_aspect("equal")
        ax.set_title(label)
        ax.set_xlabel(r"$x / (c/\omega_{\mathrm{pi}})$")
        ax.xaxis.set_major_locator(plt.MultipleLocator(2.0))
        ax.xaxis.set_minor_locator(plt.MultipleLocator(0.5))
        ax.yaxis.set_major_locator(plt.MultipleLocator(2.0))
        ax.yaxis.set_minor_locator(plt.MultipleLocator(0.5))
    axs[0].set_ylabel(r"$y / (c/\omega_{\mathrm{pi}})$")

    sm = plt.cm.ScalarMappable(cmap="RdBu_r", norm=plt.Normalize(vmin=-vmax, vmax=vmax))
    sm.set_array([])

    fig.canvas.draw()
    ax_pos = axs[-1].get_position()
    cbar_x0 = ax_pos.x1 + 0.02
    cbar_width = 0.02
    cbar_ax = fig.add_axes([cbar_x0, ax_pos.y0, cbar_width, ax_pos.y1 - ax_pos.y0])
    fig.colorbar(sm, cax=cbar_ax)

    fig.suptitle(rf"$\omega_{{\mathrm{{pi}}}} t = {time:.2f}$")

    if save:
        fig.savefig(save)

    return fig, axs


def plot_moment_snapshot(run, step, *, save=None):
    """Plot 2D snapshot of density and velocity moments (mean over z) from a picnix Run.

    Parameters
    ----------
    run : picnix.Run
        Loaded simulation run object.
    step : int
        Simulation step to plot.
    save : str, optional
        Filepath to save the figure.
    """
    from matplotlib import pyplot as plt

    data = run.read_at("field", step)
    mime = run.config["parameter"]["mime"]
    alpha = run.config["parameter"]["alpha"]
    ush = run.config["parameter"]["ush"]
    vsh = ush / np.sqrt(1.0 + ush**2)

    um = data["um"]
    Ns = um.shape[-2]

    mass = [1, mime, mime]
    bulk = [
        0.0,
        -2 * alpha * vsh / (1 - (1 - 2 * alpha) * vsh**2),
        +2 * (1 - alpha) * vsh / (1 + (1 - 2 * alpha) * vsh**2),
    ]
    species_labels = [r"$ele$", r"$ion$", r"$ref$"]
    moment_labels = [
        r"$n / n_{\mathrm{e}}$",
        r"$\delta V_x / V_{\mathrm{sh}}$",
        r"$\delta V_y / V_{\mathrm{sh}}$",
        r"$\delta V_z / V_{\mathrm{sh}}$",
    ]

    xc = run.xc / np.sqrt(mime)
    yc = run.yc / np.sqrt(mime)
    X, Y = np.meshgrid(xc, yc, indexing="xy")

    time = run.get_time_at("field", step) / np.sqrt(mime)

    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(
        3,
        8,
        width_ratios=[1, 0.05, 1, 0.05, 1, 0.05, 1, 0.05],
        left=0.06,
        right=0.96,
        bottom=0.06,
        top=0.94,
        wspace=0.25,
        hspace=0.15,
    )

    axs = []
    for row in range(3):
        axs.append([fig.add_subplot(gs[row, col]) for col in range(0, 8, 2)])

    all_pos = []

    for s in range(Ns):
        n = um[..., s, 0].mean(axis=0)
        vx = um[..., s, 1].mean(axis=0) / (n + 1e-32)
        vy = um[..., s, 2].mean(axis=0) / (n + 1e-32)
        vz = um[..., s, 3].mean(axis=0) / (n + 1e-32)

        n = n / mass[s]
        vx = (vx - bulk[s]) / vsh
        vy = vy / vsh
        vz = vz / vsh

        v_max = max(np.abs(vx).max(), np.abs(vy).max(), np.abs(vz).max())

        panels = [n, vx, vy, vz]
        vmins = [0.0, -v_max, -v_max, -v_max]
        vmaxs = [n.max(), v_max, v_max, v_max]
        cmaps = ["viridis", "RdBu_r", "RdBu_r", "RdBu_r"]

        for col, (panel, vmin, vmax, cmap) in enumerate(
            zip(panels, vmins, vmaxs, cmaps, strict=True)
        ):
            ax = axs[s][col]
            ax.pcolormesh(X, Y, panel, cmap=cmap, vmin=vmin, vmax=vmax, shading="nearest")
            ax.set_aspect("equal")

            if s == 0:
                ax.set_title(moment_labels[col])
            if col == 0:
                ax.set_ylabel(r"$y / (c/\omega_{\mathrm{pi}})$")
            if s == 2:
                ax.set_xlabel(r"$x / (c/\omega_{\mathrm{pi}})$")

            ax.xaxis.set_major_locator(plt.MultipleLocator(2.0))
            ax.xaxis.set_minor_locator(plt.MultipleLocator(0.5))
            ax.yaxis.set_major_locator(plt.MultipleLocator(2.0))
            ax.yaxis.set_minor_locator(plt.MultipleLocator(0.5))

            all_pos.append((ax, vmin, vmax, cmap))

    for s in range(Ns):
        ax = axs[s][0]
        ax.text(
            -0.25,
            0.5,
            species_labels[s],
            transform=ax.transAxes,
            va="center",
            ha="right",
            fontsize=12,
        )

    fig.canvas.draw()
    for ax, vmin, vmax, cmap in all_pos:
        ax_pos = ax.get_position()
        cbar_x0 = ax_pos.x1 + 0.01
        cbar_width = 0.01
        cbar_ax = fig.add_axes([cbar_x0, ax_pos.y0, cbar_width, ax_pos.y1 - ax_pos.y0])
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
        sm.set_array([])
        fig.colorbar(sm, cax=cbar_ax)

    fig.suptitle(rf"$\omega_{{\mathrm{{pi}}}} t = {time:.2f}$")

    if save:
        fig.savefig(save)

    return fig, axs


if __name__ == "__main__":
    import argparse

    import matplotlib

    matplotlib.use("Agg")

    import picnix

    parser = argparse.ArgumentParser(description="Plot PIC-NIX snapshots")
    parser.add_argument("profile", help="Path to picnix profile (.msg)")
    parser.add_argument("step", type=int, help="Simulation step to plot")
    parser.add_argument(
        "-t",
        "--type",
        choices=["field", "moment"],
        default="field",
        help="Plot type: field (B-field) or moment (density/velocity)",
    )
    parser.add_argument("-o", "--output", default=None, help="Output image filepath")
    args = parser.parse_args()

    run = picnix.Run(args.profile)

    if args.type == "field":
        save_path = args.output or f"field_snapshot_{args.step}.png"
        fig, _ = plot_field_snapshot(run, args.step, save=save_path)
    else:
        save_path = args.output or f"moment_snapshot_{args.step}.png"
        fig, _ = plot_moment_snapshot(run, args.step, save=save_path)

    print(f"Saved to {save_path}")

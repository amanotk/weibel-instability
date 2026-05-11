"""Summary plots for PIC-NIX simulation data."""

import numpy as np


def plot_field_evolution(run, *, save=None):
    """Plot time evolution of volume-averaged magnetic field energy components.

    Parameters
    ----------
    run : picnix.Run
        Loaded simulation run object.
    save : str, optional
        Filepath to save the figure.
    """
    from matplotlib import pyplot as plt

    mime = run.config["parameter"]["mime"]
    alpha = run.config["parameter"]["alpha"]
    ush = run.config["parameter"]["ush"]
    Beq = np.sqrt(alpha * (1 - alpha) * ush**2 * mime)

    steps = run.get_step("field")
    times = run.get_time("field")

    bx2_list = []
    by2_list = []
    bz2_list = []

    for step in steps:
        data = run.read_at("field", step)
        uf = data["uf"]
        bx2_list.append(np.mean(uf[..., 3] ** 2) / Beq**2)
        by2_list.append(np.mean(uf[..., 4] ** 2) / Beq**2)
        bz2_list.append(np.mean(uf[..., 5] ** 2) / Beq**2)

    wt = times / np.sqrt(mime)
    bx2 = np.array(bx2_list)
    by2 = np.array(by2_list)
    bz2 = np.array(bz2_list)

    fig = plt.figure(figsize=(6, 4))
    ax = fig.gca()

    ax.semilogy(wt, bx2, label=r"$B_x^2 / B_{\mathrm{eq}}^2$")
    ax.semilogy(wt, by2, label=r"$B_y^2 / B_{\mathrm{eq}}^2$")
    ax.semilogy(wt, bz2, label=r"$B_z^2 / B_{\mathrm{eq}}^2$")

    ax.set_xlabel(r"$\omega_{\mathrm{pi}} t$")
    ax.set_ylabel(r"$B_i^2 / B_{\mathrm{eq}}^2$")
    ax.legend(loc="best")
    ax.grid(True, which="both", alpha=0.3)
    ax.set_ylim(1e-7, 1e-1)

    fig.tight_layout()

    if save:
        fig.savefig(save)

    return fig, ax


def plot_field_lines(ax, az, X, Y, *, nlines=10):
    """Overlay field-line contours on an axis.

    Parameters
    ----------
    ax : Axes
        Matplotlib axis to draw on.
    az : (Ny, Nx) array
        Vector potential (any absolute scale works).
    X, Y : (Ny, Nx) arrays
        Grid coordinates matching ``az``.
    nlines : int
        Number of contour lines to draw (evenly spaced).
    """
    az_min = az.min()
    az_max = az.max()
    if az_max - az_min < 1e-15:
        return
    levels = np.linspace(az_min, az_max, nlines + 2)[1:-1]
    ax.contour(
        X,
        Y,
        az,
        levels=levels,
        colors="k",
        linewidths=0.7,
        linestyles="solid",
    )


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

    from src.fields import calc_vector_potential_2d

    data = run.read_at("field", step)
    mime = run.config["parameter"]["mime"]
    sigma = run.config["parameter"]["sigma"]
    alpha = run.config["parameter"]["alpha"]
    ush = run.config["parameter"]["ush"]
    B0 = np.sqrt(sigma)
    Beq = np.sqrt(alpha * (1 - alpha) * ush**2 * mime)

    uf = data["uf"]
    bx_raw = uf[..., 3].mean(axis=0)
    by_raw = uf[..., 4].mean(axis=0)
    bz_raw = uf[..., 5].mean(axis=0)
    az = calc_vector_potential_2d(bx_raw, by_raw, run.delh)

    bx = bx_raw / Beq
    by = by_raw / Beq - B0 / Beq
    bz = bz_raw / Beq
    b_mag = np.sqrt(bx**2 + by**2 + bz**2)

    vmax = max(np.abs(bx).max(), np.abs(by).max(), np.abs(bz).max())

    xc = data["xc"] / np.sqrt(mime)
    yc = data["yc"] / np.sqrt(mime)

    X, Y = np.meshgrid(xc, yc, indexing="xy")

    time = run.get_time_at("field", step) / np.sqrt(mime)

    fig = plt.figure(figsize=(16, 3.77))
    gs = fig.add_gridspec(
        1,
        8,
        width_ratios=[1, 0.05, 1, 0.05, 1, 0.05, 1, 0.05],
        left=0.06,
        right=0.96,
        bottom=0.1327,
        top=0.8407,
        wspace=0.25,
    )

    axs = [fig.add_subplot(gs[0, col]) for col in range(0, 8, 2)]

    labels = [
        r"$|B| / B_{\mathrm{eq}}$",
        r"$\delta B_x / B_{\mathrm{eq}}$",
        r"$\delta B_y / B_{\mathrm{eq}}$",
        r"$\delta B_z / B_{\mathrm{eq}}$",
    ]
    fields = [b_mag, bx, by, bz]
    vmins = [0.0, -vmax, -vmax, -vmax]
    vmaxs = [b_mag.max(), vmax, vmax, vmax]
    cmaps = ["viridis", "RdBu_r", "RdBu_r", "RdBu_r"]

    for ax, label, bfield, vmin, vmax_val, cmap in zip(
        axs, labels, fields, vmins, vmaxs, cmaps, strict=True
    ):
        ax.pcolormesh(
            X,
            Y,
            bfield,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax_val,
            shading="nearest",
            rasterized=True,
        )
        ax.set_aspect("equal")
        ax.set_title(label)
        ax.set_xlabel(r"$x / (c/\omega_{\mathrm{pi}})$")
        ax.xaxis.set_major_locator(plt.MultipleLocator(2.0))
        ax.xaxis.set_minor_locator(plt.MultipleLocator(0.5))
        ax.yaxis.set_major_locator(plt.MultipleLocator(2.0))
        ax.yaxis.set_minor_locator(plt.MultipleLocator(0.5))
    axs[0].set_ylabel(r"$y / (c/\omega_{\mathrm{pi}})$")

    for ax in axs:
        plot_field_lines(ax, az, X, Y)

    fig.canvas.draw()
    for ax, vmin, vmax_val, cmap in zip(axs, vmins, vmaxs, cmaps, strict=True):
        ax_pos = ax.get_position()
        cbar_x0 = ax_pos.x1 + 0.01
        cbar_width = 0.01
        cbar_ax = fig.add_axes([cbar_x0, ax_pos.y0, cbar_width, ax_pos.y1 - ax_pos.y0])
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax_val))
        sm.set_array([])
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

    from src.fields import calc_vector_potential_2d

    data = run.read_at("field", step)
    mime = run.config["parameter"]["mime"]
    alpha = run.config["parameter"]["alpha"]
    ush = run.config["parameter"]["ush"]
    vsh = ush / np.sqrt(1.0 + ush**2)

    uf = data["uf"]
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

    xc = data["xc"] / np.sqrt(mime)
    yc = data["yc"] / np.sqrt(mime)

    bx_raw = uf[..., 3].mean(axis=0)
    by_raw = uf[..., 4].mean(axis=0)
    az = calc_vector_potential_2d(bx_raw, by_raw, run.delh)

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
            ax.pcolormesh(
                X,
                Y,
                panel,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                shading="nearest",
                rasterized=True,
            )
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

    for row in range(Ns):
        for ax in axs[row]:
            plot_field_lines(ax, az, X, Y)

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
    parser.add_argument(
        "step",
        type=int,
        nargs="?",
        help="Simulation step (required for field/moment)",
    )
    parser.add_argument(
        "-t",
        "--type",
        choices=["field", "moment", "evolution"],
        default="field",
        help="Plot type: field (B-field), moment (density/velocity), or evolution (B^2 vs time)",
    )
    parser.add_argument("-o", "--output", default=None, help="Output image filepath")
    args = parser.parse_args()

    if args.type != "evolution" and args.step is None:
        parser.error("step is required for --type field|moment")

    run = picnix.Run(args.profile)

    if args.type == "field":
        save_path = args.output or f"field_snapshot_{args.step}.png"
        fig, _ = plot_field_snapshot(run, args.step, save=save_path)
    elif args.type == "moment":
        save_path = args.output or f"moment_snapshot_{args.step}.png"
        fig, _ = plot_moment_snapshot(run, args.step, save=save_path)
    else:
        save_path = args.output or "field_evolution.png"
        fig, _ = plot_field_evolution(run, save=save_path)

    print(f"Saved to {save_path}")

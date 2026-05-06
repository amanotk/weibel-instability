"""Summary plots for PIC-NIX simulation data."""

import numpy as np


def plot_b_snapshot(run, step, *, save=None):
    """Plot 2D snapshot of Bx, By, Bz (mean over z) from a picnix Run.

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

    uf = data["uf"]
    bx = uf[..., 3].mean(axis=0)
    by = uf[..., 4].mean(axis=0)
    bz = uf[..., 5].mean(axis=0)

    vmax = max(np.abs(bx).max(), np.abs(by).max(), np.abs(bz).max())

    xc = run.xc / np.sqrt(mime)
    yc = run.yc / np.sqrt(mime)
    X, Y = np.meshgrid(xc, yc, indexing="xy")

    time = run.get_time_at("field", step) / np.sqrt(mime)

    fig = plt.figure(figsize=(14, 4))
    gs = fig.add_gridspec(
        1, 4, width_ratios=[1, 1, 1, 0.05], wspace=0.3,
        left=0.05, right=0.95, bottom=0.18, top=0.85,
    )
    axs = [fig.add_subplot(gs[0, i]) for i in range(3)]
    cbar_ax = fig.add_subplot(gs[0, 3])

    labels = [r"$B_x$", r"$B_y$", r"$B_z$"]
    fields = [bx, by, bz]

    for ax, label, bfield in zip(axs, labels, fields, strict=True):
        ax.pcolormesh(X, Y, bfield, cmap="RdBu_r", vmin=-vmax, vmax=vmax, shading="nearest")
        ax.set_title(label)
        ax.set_xlabel(r"$x / (c/\omega_{\mathrm{pi}})$")
        ax.set_ylabel(r"$y / (c/\omega_{\mathrm{pi}})$")

    sm = plt.cm.ScalarMappable(cmap="RdBu_r", norm=plt.Normalize(vmin=-vmax, vmax=vmax))
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

    parser = argparse.ArgumentParser(description="Plot B-field snapshot from PIC-NIX data")
    parser.add_argument("profile", help="Path to picnix profile (.msg)")
    parser.add_argument("step", type=int, help="Simulation step to plot")
    parser.add_argument("-o", "--output", default=None, help="Output image filepath")
    args = parser.parse_args()

    run = picnix.Run(args.profile)
    save_path = args.output or f"b_snapshot_{args.step}.png"

    fig, _ = plot_b_snapshot(run, args.step, save=save_path)
    print(f"Saved to {save_path}")

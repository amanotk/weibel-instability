import marimo

__generated_with = "0.23.5"
app = marimo.App(width="medium")


@app.cell
def _():
    from pathlib import Path

    import marimo as mo
    import matplotlib
    import matplotlib.pyplot as plt
    import picnix

    from summary import plot_b_snapshot, plot_moment_snapshot

    matplotlib.use("Agg")
    return Path, mo, picnix, plot_b_snapshot, plot_moment_snapshot


@app.cell
def _(Path, mo):
    import os

    repo_root = Path(__file__).parent.parent
    profiles = {}
    for p in sorted(repo_root.rglob("profile.msgpack")):
        profiles[str(p.relative_to(repo_root))] = str(p)
    _info = f"Found {len(profiles)} profile(s)"
    mo.md(_info)
    return (profiles,)


@app.cell
def _(mo, profiles):
    profile = mo.ui.dropdown(
        options=list(profiles.keys()),
        value=list(profiles.keys())[0] if profiles else None,
        label="Profile",
    )
    profile
    return (profile,)


@app.cell
def _(picnix, profile, profiles):
    profile_path = profiles[profile.value] if profile.value else None
    run = picnix.Run(profile_path) if profile_path else None
    run
    return (run,)


@app.cell
def _(mo, run):
    steps = sorted(run.get_step("field")) if run is not None else []
    step = mo.ui.dropdown(
        options=[int(s) for s in steps],
        value=int(steps[-1]) if steps else None,
        label="Step",
    )
    step
    return (step,)


@app.cell
def _(mo, plot_b_snapshot, run, step):
    if run is not None and step.value is not None:
        _fig, _ = plot_b_snapshot(run, step.value)
        _display = _fig
    else:
        _display = mo.md("_Select a profile and step to plot._")
    _display
    return


@app.cell
def _(mo, plot_moment_snapshot, run, step):
    if run is not None and step.value is not None:
        _fig, _ = plot_moment_snapshot(run, step.value)
        _display = _fig
    else:
        _display = mo.md("_Select a profile and step to plot._")
    _display
    return


if __name__ == "__main__":
    app.run()

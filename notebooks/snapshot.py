import marimo

__generated_with = "0.23.5"
app = marimo.App(width="medium")


@app.cell
def _():
    from pathlib import Path

    import marimo as mo
    import matplotlib
    import picnix

    from src.summary import plot_field_snapshot, plot_moment_snapshot

    matplotlib.use("Agg")

    repo_root = Path(__file__).parent.parent
    profiles = {}
    for p in sorted(repo_root.rglob("profile.msgpack")):
        profiles[str(p.relative_to(repo_root))] = str(p)
    return mo, picnix, plot_field_snapshot, plot_moment_snapshot, profiles


@app.cell
def _(mo, profiles):
    profile_widget = mo.ui.dropdown(
        options=list(profiles.keys()) if profiles else [],
        value=list(profiles.keys())[0] if profiles else None,
        label="Profile",
    )
    profile_widget
    return (profile_widget,)


@app.cell
def _(mo, picnix, profile_widget, profiles):
    run = picnix.Run(profiles[profile_widget.value]) if profile_widget.value else None
    step_options = sorted(run.get_step("field")) if run is not None else []
    step_steps = [int(s) for s in step_options]

    step_widget = mo.ui.dropdown(
        options=step_steps,
        value=step_steps[-1] if step_steps else None,
        label="Step",
    )

    plot_button = mo.ui.run_button(label="Update Plot")

    mo.vstack([step_widget, plot_button])
    return plot_button, run, step_steps, step_widget


@app.cell
def _(
    mo,
    plot_button,
    plot_field_snapshot,
    plot_moment_snapshot,
    profiles,
    run,
    step_steps,
    step_widget,
):
    if not profiles:
        display = mo.md("_No profiles found in repository._")
    elif not step_steps:
        display = mo.md("_Selected profile has no field data._")
    elif plot_button.value and step_widget.value is not None:
        fig1, _ = plot_field_snapshot(run, step_widget.value)
        fig2, _ = plot_moment_snapshot(run, step_widget.value)
        display = mo.vstack([fig1, fig2])
    else:
        display = mo.md("_Choose profile, step, and press Update Plot._")
    display
    return


if __name__ == "__main__":
    app.run()

"""MPI-parallel scalar computation over PIC-NIX snapshots.

Usage:
    uv run python src/compute_scalars.py profile.msgpack --kind field-energy
    mpirun -n 8 uv run python src/compute_scalars.py profile.msgpack --kind field-energy
    uv run python src/compute_scalars.py profile.msgpack --kind field-energy,moment-density
"""

import pathlib

import numpy as np


def _get_mpi_comm():
    """Return MPI.COMM_WORLD if running under an MPI launcher, or None for serial."""
    import os

    envvars = [
        "SLURM_PROCID",
        "OMPI_COMM_WORLD_RANK",
        "PMI_RANK",
        "MV2_COMM_WORLD_RANK",
    ]
    if not any(os.environ.get(v) for v in envvars):
        return None
    from mpi4py import MPI

    return MPI.COMM_WORLD


# ---------------------------------------------------------------------------
# Task definitions
# ---------------------------------------------------------------------------

_TASKS = {}


def register_task(kind):
    """Decorator to register a scalar computation task.

    The decorated function must accept (data, config) -> dict[str, float].
    """

    def decorator(func):
        fname = kind.replace("-", "_") + ".npz"
        _TASKS[kind] = (func, fname)
        return func

    return decorator


@register_task("field-energy")
def _compute_field_energy(data, config):
    """Compute volume-averaged B-field energy components."""
    mime = config["mime"]
    alpha = config["alpha"]
    ush = config["ush"]
    Beq = np.sqrt(alpha * (1 - alpha) * ush**2 * mime)

    uf = data["uf"]
    return {
        "bx2": np.mean(uf[..., 3] ** 2) / Beq**2,
        "by2": np.mean(uf[..., 4] ** 2) / Beq**2,
        "bz2": np.mean(uf[..., 5] ** 2) / Beq**2,
    }


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def run_scalar_tasks(run, tasks, prefix="field"):
    """Run tasks across all steps, returning {key: np.array}.

    Works in serial or MPI (auto-detected). Each task reads the same step
    data so picnix caching handles redundant reads within a process.
    """
    comm = _get_mpi_comm()
    if comm is not None:
        return _run_mpi(run, tasks, comm, prefix)
    return _run_serial(run, tasks, prefix)


def _run_serial(run, tasks, prefix):
    steps = run.get_step(prefix)
    config = run.config["parameter"]

    # Collect all output keys
    sample_data = run.read_at(prefix, steps[0])
    all_keys = list(_gather_keys(tasks, sample_data, config))

    out = {k: [] for k in all_keys}
    times = []
    for step in steps:
        data = run.read_at(prefix, step)
        times.append(run.get_time_at(prefix, step))
        for task, _fname in tasks:
            for k, v in task(data, config).items():
                out[k].append(v)

    result = {"steps": np.array(steps), "times": np.array(times)}
    result.update({k: np.array(v) for k, v in out.items()})
    return result


def _run_mpi(run, tasks, comm, prefix):
    rank = comm.Get_rank()
    size = comm.Get_size()
    steps = run.get_step(prefix)
    config = run.config["parameter"]

    # Scatter steps across ranks
    my_steps = steps[rank::size]
    config_copy = dict(config)

    sample_data = run.read_at(prefix, my_steps[0]) if my_steps else None
    gathered_keys = comm.gather(
        set(_gather_keys(tasks, sample_data, config_copy)) if sample_data else set(),
        root=0,
    )
    all_keys = list(set().union(*gathered_keys)) if rank == 0 else None

    # Rank 0 broadcasts keys
    all_keys = comm.bcast(all_keys, root=0)

    # Each rank computes its steps
    local_results = {k: [] for k in all_keys}
    local_times = []
    for step in my_steps:
        data = run.read_at(prefix, step)
        local_times.append(run.get_time_at(prefix, step))
        for task, _fname in tasks:
            for k, v in task(data, config).items():
                local_results[k].append(v)

    # Gather local steps to rank 0
    local_steps_list = comm.gather(my_steps, root=0)
    local_times_list = comm.gather(local_times, root=0)
    local_results_list = comm.gather(local_results, root=0)

    if rank == 0:
        # Reconstruct global results (data is in sorted order by construction)
        global_results = {k: [] for k in all_keys}
        global_steps = []
        global_times = []

        for ls, lt, lr in zip(local_steps_list, local_times_list, local_results_list, strict=True):
            global_steps.extend(ls)
            global_times.extend(lt)
            for k in all_keys:
                global_results[k].extend(lr.get(k, []))

        # Sort by step to ensure correct ordering (in case of uneven scatter)
        order = np.argsort(global_steps)
        result = {
            "steps": np.array(global_steps)[order],
            "times": np.array(global_times)[order],
        }
        result.update({k: np.array(global_results[k])[order] for k in all_keys})
        return result

    return None


def _gather_keys(tasks, sample_data, config):
    keys = set()
    for task, _fname in tasks:
        keys.update(task(sample_data, config).keys())
    return keys


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    import argparse

    import picnix

    parser = argparse.ArgumentParser(description="Compute scalar diagnostics")
    parser.add_argument("profile", help="Path to picnix profile")
    parser.add_argument(
        "--kind",
        required=True,
        help="Comma-separated task kinds (e.g. field-energy,moment-density)",
    )
    parser.add_argument(
        "--prefix",
        default="field",
        help="Diagnostic prefix (default: field)",
    )
    args = parser.parse_args()

    # Resolve task functions
    kinds = [k.strip() for k in args.kind.split(",")]
    for k in kinds:
        if k not in _TASKS:
            available = ", ".join(_TASKS.keys())
            parser.error(f"Unknown task kind '{k}'. Available: {available}")

    task_funcs = [_TASKS[k][0] for k in kinds]
    output_name = _TASKS[kinds[0]][1]  # use first task's filename
    if len(kinds) > 1:
        # Combine: use hyphenated names
        output_name = "_".join(_TASKS[k][1].replace(".npz", "") for k in kinds) + ".npz"

    # Run
    run = picnix.Run(args.profile)
    results = run_scalar_tasks(run, [(f, None) for f in task_funcs], prefix=args.prefix)

    # Write output
    if results is not None:
        data_dir = pathlib.Path(run.format_log_filename()).parent
        output_path = data_dir / output_name
        np.savez(str(output_path), **results)
        print(f"Saved {output_path}")

    comm = _get_mpi_comm()
    if comm is not None:
        comm.Barrier()


if __name__ == "__main__":
    main()

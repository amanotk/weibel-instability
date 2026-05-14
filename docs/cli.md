# CLI Tools Reference

Quick reference for all command-line tools in this repository. All tools use `uv run python` to execute.

---

## `summary.py` — Plot Snapshots

Generate 2D field/moment snapshots or evolution plots from PIC-NIX simulation data.

### Usage

```
uv run python src/summary.py PROFILE [STEP] [OPTIONS]
```

### Arguments

| Argument | Required | Description |
|----------|----------|-------------|
| `PROFILE` | Yes | Path to picnix profile (`profile.msgpack`) |
| `STEP` | No | Simulation step (required for `--type field\|moment`) |

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `-t, --type` | `field` | Plot type: `field` (B-field), `moment` (density/velocity), `evolution` (B² vs time) |
| `-o, --output` | auto-generated | Output image filepath |

### Examples

```bash
# Field snapshot at step 80000
uv run python src/summary.py work/run1m/data/profile.msgpack 80000 -t field -o work/run1m/field.png

# Moment snapshot at step 80000
uv run python src/summary.py work/run1m/data/profile.msgpack 80000 -t moment -o work/run1m/moment.png

# Field evolution plot (reads all field snapshots)
uv run python src/summary.py work/run1m/data/profile.msgpack -t evolution
```

### Evolution plot performance

`--type evolution` reads all field snapshots sequentially, which is slow for many steps. Use `compute_scalars.py` first to pre-compute energies in parallel, then `summary.py` will automatically detect and use the cached results.

```bash
# Pre-compute (parallel via MPI)
mpirun -n 8 uv run python src/compute_scalars.py work/run1m/data/profile.msgpack --kind field-energy

# Plot (instant, reads pre-computed .npz)
uv run python src/summary.py work/run1m/data/profile.msgpack -t evolution
```

---

## `compute_scalars.py` — Scalar Diagnostics

MPI-parallel computation of scalar diagnostics over all snapshots. Each task reads a snapshot, computes scalar values, and writes an `.npz` file.

### Usage

```
uv run python src/compute_scalars.py PROFILE --kind KIND [OPTIONS]
```

### Arguments

| Argument | Required | Description |
|----------|----------|-------------|
| `PROFILE` | Yes | Path to picnix profile (`profile.msgpack`) |

### Options

| Flag | Required | Default | Description |
|------|----------|---------|-------------|
| `--kind` | Yes | — | Task kinds, comma-separated (e.g., `field-energy`) |
| `--prefix` | No | `field` | Diagnostic prefix for data read |

### Available Tasks

| Kind | Output File | Description |
|------|-------------|-------------|
| `field-energy` | `field_energy.npz` | Volume-averaged B-field energy components: `bx2`, `by2`, `bz2` (normalized by `Beq²`) |

### Output Format

Each kind produces an `.npz` file in the run's data directory containing:

```python
import numpy as np
d = np.load("field_energy.npz")
d["steps"]   # array of simulation steps
d["times"]   # array of simulation times
d["bx2"]     # array of Bx²/Beq² values
d["by2"]     # array of By²/Beq² values
d["bz2"]     # array of Bz²/Beq² values
```

### Examples

```bash
# Serial (single process)
uv run python src/compute_scalars.py work/run1m/data/profile.msgpack --kind field-energy

# MPI (8 processes)
mpirun -n 8 uv run python src/compute_scalars.py work/run1m/data/profile.msgpack --kind field-energy
```

### MPI Usage

The script auto-detects MPI when running under a launcher (`srun`, `mpirun`, etc.) by checking environment variables (`SLURM_PROCID`, `OMPI_COMM_WORLD_RANK`, etc.). Steps are distributed across ranks via strided assignment. When run without an MPI launcher, it falls back to serial mode automatically.

### Adding New Tasks

To add a new scalar computation:

```python
@register_task("my-task")
def _compute_my_task(data, config):
    """data: dict from run.read_at(), config: run.config['parameter']"""
    return {"scalar1": value1, "scalar2": value2}
```

The task function receives `data` (from `run.read_at(prefix, step)`) and `config` (the `parameter` section). It returns a `dict[str, float]`. Output filename is `<task-kind>.npz` (hyphens become underscores).

---

## `makemovie.py` — Generate Snapshot Movies

Generate MP4 movies from field/moment snapshots over time. MPI-aware: distributes frame rendering across ranks, rank 0 assembles with ffmpeg.

### Usage

```
uv run python src/makemovie.py PROFILE [OPTIONS]
```

### Arguments

| Argument | Required | Description |
|----------|----------|-------------|
| `PROFILE` | Yes | Path to picnix profile (`.msgpack`) |

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `-t, --type` | `field` | Plot types to include: `field` and/or `moment` |
| `--start` | first diagnostic step | First diagnostic step to include |
| `--end` | last diagnostic step | Last diagnostic step to include |
| `--stride` | `1` | Every N-th diagnostic |
| `--fps` | `10` | Frames per second |
| `-o, --output-dir` | `./movie/` | Output directory |
| `--dry-run` | off | Preview step distribution without generating |
| `--cleanup` | off | Remove frames directory after encoding |

### Examples

```bash
# Generate field movie from all steps
uv run python src/makemovie.py work/run1m/data/profile.msgpack

# MPI: parallel frame generation (8 ranks)
mpirun -n 8 uv run python src/makemovie.py work/run1m/data/profile.msgpack -t field moment

# Subset of steps, custom FPS
uv run python src/makemovie.py work/run1m/data/profile.msgpack --start 10000 --end 50000 --stride 2 --fps 15

# Preview without generating
uv run python src/makemovie.py work/run1m/data/profile.msgpack --dry-run

# Clean up temp frames after encoding
uv run python src/makemovie.py work/run1m/data/profile.msgpack --cleanup
```

---

## `notebooks/snapshot.py` — Interactive Marimo App

Interactive browser-based app for browsing field/moment snapshots with sliders for step selection.

### Usage

```bash
# Run (read-only mode)
uv run marimo run notebooks/snapshot.py -- --profile work/run1m/data/profile.msgpack

# Edit mode
uv run marimo edit notebooks/snapshot.py
```

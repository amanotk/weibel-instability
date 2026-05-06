# Weibel Instability Analysis

This repository is for analyzing Weibel instability PIC simulation data.

## What you can do here

- Generate summary plots from PIC-NIX simulation data
- Run interactive notebooks with marimo
- Track tests and documentation as the project grows

## Directory structure

- `src/`: reusable Python code (`summary.py` for snapshot plotting)
- `notebooks/`: interactive marimo notebooks
- `tests/`: automated tests
- `work/`: large files and scratch outputs (not committed)

## Quick start

1. Install dependencies:

```bash
uv sync
uv pip install -e ${HOME}/pic-nix/python/
```

2. Run snapshot plots from CLI:

```bash
uv run python src/summary.py /path/to/profile.msg STEP -t field
uv run python src/summary.py /path/to/profile.msg STEP -t moment
```

3. Open a notebook:

```bash
uv run marimo edit notebooks/snapshot.py
```

## Marimo notebook workflow

- Run in browser: `uv run marimo run notebooks/<notebook-name>.py`
- Edit notebook: `uv run marimo edit notebooks/<notebook-name>.py`
- Script-mode check: `uv run notebooks/<notebook-name>.py`
- Notebook lint check: `uvx marimo check notebooks/<notebook-name>.py`

## Recommended Git branching strategy

- `main`: stable work
- `develop`: active development
- `feature/*`: short-lived feature branches

### Development on `develop`
- Commit primarily to `develop` for ongoing work
- Merge `develop` into `main` when stable and ready for release (usually with a normal merge commit)

### Feature branches
- Create `feature/*` branches for specific features or experiments
- Merge `feature/*` branches back into `develop` when ready (usually with squash merge)

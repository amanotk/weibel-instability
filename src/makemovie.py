"""Generate a series of snapshot frames and encode to MP4 movie."""

import gc
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

try:
    import matplotlib

    matplotlib.use("Agg")
except ImportError:
    pass


def get_rank_size() -> tuple[int, int]:
    """Detect runtime environment and return (rank, size)."""
    if "SLURM_PROCID" in os.environ and "SLURM_NTASKS" in os.environ:
        return int(os.environ["SLURM_PROCID"]), int(os.environ["SLURM_NTASKS"])
    try:
        from mpi4py import MPI

        comm = MPI.COMM_WORLD
        if comm.size > 1:
            return comm.rank, comm.size
    except (ImportError, OSError):
        pass
    return 0, 1


def distribute_steps(steps: list[int], rank: int, size: int) -> list[int]:
    """Assign every size-th step to this rank (stride-aware round-robin)."""
    return steps[rank::size]


def select_steps(
    run: object,
    start: int | None = None,
    end: int | None = None,
    stride: int = 1,
) -> list[int]:
    """Filter available field steps by range and stride."""
    available = sorted(run.get_step("field"))
    if start is None:
        start = available[0]
    if end is None:
        end = available[-1]
    filtered = [s for s in available if start <= s <= end]
    return filtered[::stride]


def make_frames(
    run: object,
    plot_type: str,
    steps: list[int],
    outdir: Path,
    rank: int,
) -> None:
    """Generate PNG frames for assigned steps using summary plot functions."""
    from matplotlib import pyplot as plt

    from src.summary import plot_field_snapshot, plot_moment_snapshot

    plot_fn = {"field": plot_field_snapshot, "moment": plot_moment_snapshot}[plot_type]
    frames_dir = outdir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    for i, step in enumerate(steps):
        fname = frames_dir / f"{plot_type}_{step:08d}.png"
        fig, _ = plot_fn(run, step, save=str(fname))
        plt.close(fig)
        run.clear_cache()
        gc.collect()
        if rank == 0 or i == 0:
            print(f"[rank {rank}] {plot_type}: {i + 1}/{len(steps)} step={step}")


def encode_movie(outdir: Path, plot_type: str, fps: int) -> None:
    """Run ffmpeg to encode frames into an MP4 movie (rank 0 only)."""
    frames_dir = outdir / "frames"
    pattern = str(frames_dir / f"{plot_type}_*.png")
    output = outdir / f"movie_{plot_type}.mp4"
    cmd = [
        "ffmpeg",
        "-y",
        "-framerate",
        str(fps),
        "-pattern_type",
        "glob",
        "-i",
        pattern,
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        str(output),
    ]
    print(f"Encoding movie_{plot_type}.mp4 ...")
    subprocess.run(cmd, check=True)
    print(f"  -> {output}")


def wait_for_sync(frames_dir: Path, expected: int, timeout: int = 3600) -> None:
    """Block until sentinel files confirm all ranks finished."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        sentinels = list(frames_dir.glob("_done_*"))
        if len(sentinels) >= expected:
            return
        time.sleep(1)
    raise TimeoutError(
        f"sync timed out after {timeout}s; expected {expected} done-sentinels, got {len(sentinels)}"
    )


def print_dry_run(
    profile: str,
    plot_types: list[str],
    selected: dict[str, list[int]],
    rank_steps: dict[int, dict[str, list[int]]],
    outdir: Path,
    fps: int,
    rank: int,
    size: int,
) -> None:
    """Print a summary of what would happen (dry run mode)."""
    print(f"Profile: {profile}")
    for pt in plot_types:
        steps = selected[pt]
        print(f"  {pt}: {len(steps)} frames selected")
        if steps:
            print(f"    range: {steps[0]}..{steps[-1]}")
    if size > 1:
        print(f"  Ranks: {size}")
        for pt in plot_types:
            for r, rst in rank_steps.items():
                n = len(rst[pt])
                sample = ", ".join(str(s) for s in rst[pt][:3])
                tail = ", ..." if len(rst[pt]) > 3 else ""
                print(f"    rank {r} ({pt}): {n} frames [{sample}{tail}]")
    print(f"  Output dir: {outdir}")
    for pt in plot_types:
        if selected[pt]:
            movie = outdir / f"movie_{pt}.mp4"
            print(f"  Movie: {movie} (fps={fps})")
    frames = outdir / "frames"
    print(f"  Frame pattern: {frames}/{pt}_NNNNNNNN.png")


def main() -> None:
    import argparse

    import picnix

    parser = argparse.ArgumentParser(description="Generate snapshot movie from PIC-NIX data")
    parser.add_argument("profile", help="Path to picnix profile (.msg)")
    parser.add_argument(
        "-t",
        "--type",
        nargs="+",
        choices=["field", "moment"],
        default=["field"],
        help="Plot types to include (default: field)",
    )
    parser.add_argument("--start", type=int, default=None, help="First diagnostic step")
    parser.add_argument("--end", type=int, default=None, help="Last diagnostic step")
    parser.add_argument("--stride", type=int, default=1, help="Every N-th diagnostic (default: 1)")
    parser.add_argument("--fps", type=int, default=10, help="Frames per second (default: 10)")
    parser.add_argument("-o", "--output-dir", default="./movie/", help="Output directory")
    parser.add_argument("--dry-run", action="store_true", help="Preview without generating")
    parser.add_argument("--cleanup", action="store_true", help="Remove frames dir after encoding")
    args = parser.parse_args()

    outdir = Path(args.output_dir)
    rank, size = get_rank_size()

    if rank == 0:
        print(f"Running with rank={rank}/{size}")

    run = picnix.Run(args.profile)

    selected: dict[str, list[int]] = {}
    for pt in args.type:
        selected[pt] = select_steps(run, args.start, args.end, args.stride)
        if not selected[pt]:
            if rank == 0:
                print(
                    f"Error: no '{pt}' diagnostic steps found "
                    f"in range [{args.start or 'start'}..{args.end or 'end'}] "
                    f"with stride {args.stride}",
                    file=sys.stderr,
                )
            sys.exit(1)

    if args.dry_run:
        rank_steps_all: dict[int, dict[str, list[int]]] = {}
        for r in range(size):
            rank_steps_all[r] = {}
            for pt in args.type:
                rank_steps_all[r][pt] = distribute_steps(selected[pt], r, size)
        if rank == 0:
            print_dry_run(
                args.profile,
                args.type,
                selected,
                rank_steps_all,
                outdir,
                args.fps,
                rank,
                size,
            )
        return

    if not shutil.which("ffmpeg"):
        print("Error: ffmpeg not found in PATH", file=sys.stderr)
        sys.exit(1)

    outdir.mkdir(parents=True, exist_ok=True)

    for pt in args.type:
        my_steps = distribute_steps(selected[pt], rank, size)
        print(f"[rank {rank}] {pt}: assigned {len(my_steps)} frames")
        if my_steps:
            make_frames(run, pt, my_steps, outdir, rank)

    frames_dir = outdir / "frames"

    if size > 1:
        (frames_dir / f"_done_{rank}").touch()
        if rank == 0:
            wait_for_sync(frames_dir, size)

    if rank == 0:
        for pt in args.type:
            encode_movie(outdir, pt, args.fps)

        if args.cleanup:
            sentinels = list(frames_dir.glob("_done_*"))
            for s in sentinels:
                s.unlink()
            shutil.rmtree(frames_dir, ignore_errors=True)
            print(f"Cleaned up {frames_dir}")

        print("Done.")


if __name__ == "__main__":
    main()

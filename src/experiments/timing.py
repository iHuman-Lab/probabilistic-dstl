"""Evaluation cost of pdSTL: windows at once vs the recurrent register.

One planner iteration evaluates the formula smoothly and back-propagates to
the belief means. This benchmark times exactly that (atoms from Gaussian
beliefs, the formula, the backward pass) for a formula shaped like the lane
merge's phi_1, with every window sized to the trace, in two forms:

- "parallel": `pdstl.operators` (every origin's window reduced at once);
- "recurrent": `pdstl.recurrent` (the register recursion, one step at a time).

Both give the same values (checked on every configuration). Experiment A
varies the trace length at batch 1; experiment B varies the number of
candidate plans scored together at the lane horizon.
"""

import json
import os
from statistics import median
from time import perf_counter

import torch

from models.beliefs import GaussianBeliefTrajectory
from pdstl.operators import Always, Eventually
from pdstl.predicates import AxisInterval
from pdstl.recurrent import RecurrentAlways, RecurrentEventually
from planning.runners import output_path
from utils import cpu_model

METHODS = {
    "parallel": (Always, Eventually),
    "recurrent": (RecurrentAlways, RecurrentEventually),
}
HORIZONS = (31, 61, 121, 241, 481)
BATCHES = (1, 16, 256, 4096)
BATCH_STEPS = 31  # the lane merge's H + 1


def merge_like_formula(steps, always=Always, eventually=Eventually):
    """G[0, n] safe AND F[s0, s1](G[0, d] lane AND G[d, d + D] settled-in-lane).

    The phi_1 shape with windows scaled to a trace of `steps` samples
    (n = steps - 1), so one origin (k = 0) is evaluated, as in planning.

    Args:
        steps: Trace length.
        always: Always operator class.
        eventually: Eventually operator class.

    Returns:
        The formula.
    """
    n = steps - 1
    road = AxisInterval(-1.8, 5.4, dim=1, name="road")
    car = AxisInterval(20.0, 25.0, dim=0) & AxisInterval(2.6, 4.6, dim=1)
    lane = AxisInterval(2.4, 4.8, dim=1, name="lane")
    settled = AxisInterval(-0.3, 0.3, dim=3, name="settled")
    dwell = max(1, n // 6)
    first, last = n // 6, n // 2
    stay = n - last - dwell
    merge = eventually(
        always(lane, [0, dwell])
        & always(lane & settled, [dwell, dwell + stay]),
        [first, last],
    )
    return always(road & ~car, [0, n]) & merge


def random_beliefs(batch, steps, device, generator):
    """Gaussian beliefs of a merge-like path: mean [B, T, 4], covariance.

    Args:
        batch: Number of candidate plans.
        steps: Trace length.
        device: Torch device.
        generator: CPU torch generator.

    Returns:
        (mean, covariance), mean a leaf that requires grad.
    """
    shape = (batch, steps)
    mean = torch.zeros(*shape, 4)
    mean[..., 0] = torch.linspace(0.0, 60.0, steps)
    mean[..., 1] = 3.6 * torch.sigmoid(torch.linspace(-6.0, 6.0, steps))
    mean[..., 1] += 0.3 * torch.randn(*shape, generator=generator)
    mean[..., 2] = 12.0
    mean[..., 3] = 0.2 * torch.randn(*shape, generator=generator)
    variance = 0.01 + 0.05 * torch.rand(*shape, 4, generator=generator)
    covariance = torch.diag_embed(variance)
    return mean.to(device).requires_grad_(True), covariance.to(device)


def _sync(device):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


def iteration_seconds(formula, batch, steps, device, beta, generator):
    """Seconds for one smooth forward and backward pass on fresh beliefs.

    Building the belief trajectory is not timed (the planner's rollout does
    that); the atoms, the formula and the backward pass are.

    Args:
        formula: pdSTL formula.
        batch: Number of candidate plans.
        steps: Trace length.
        device: Torch device.
        beta: Smoothing strength.
        generator: CPU torch generator.

    Returns:
        Elapsed seconds.
    """
    mean, covariance = random_beliefs(batch, steps, device, generator)
    beliefs = GaussianBeliefTrajectory(mean, covariance)
    _sync(device)
    start = perf_counter()
    score = formula(beliefs, beta=beta)[:, 0, 0].sum()
    score.backward()
    _sync(device)
    return perf_counter() - start


def median_ms(run, min_reps=3, max_reps=30, budget=0.5):
    """Median of repeated timings after one warm-up, in milliseconds.

    Args:
        run: Callable returning elapsed seconds.
        min_reps: At least this many timed calls.
        max_reps: At most this many.
        budget: Stop after min_reps once this many seconds are spent.

    Returns:
        (median ms, number of timed calls).
    """
    run()
    times = []
    while len(times) < min_reps or (
        sum(times) < budget and len(times) < max_reps
    ):
        times.append(run())
    return median(times) * 1e3, len(times)


def largest_difference(steps, batch, device, beta, generator):
    """Largest |parallel - recurrent| over the output, smooth and exact.

    Args:
        steps: Trace length.
        batch: Number of candidate plans.
        device: Torch device.
        beta: Smoothing strength.
        generator: CPU torch generator.

    Returns:
        Float.
    """
    mean, covariance = random_beliefs(batch, steps, device, generator)
    worst = 0.0
    with torch.no_grad():
        for smoothing in (None, beta):
            out = [
                merge_like_formula(steps, *METHODS[name])(
                    GaussianBeliefTrajectory(mean, covariance), beta=smoothing
                )
                for name in ("parallel", "recurrent")
            ]
            worst = max(worst, float((out[0] - out[1]).abs().max()))
    return worst


def _row(experiment, device, method, steps, batch, beta, generator):
    formula = merge_like_formula(steps, *METHODS[method])

    def run():
        return iteration_seconds(
            formula, batch, steps, device, beta, generator
        )

    ms, reps = median_ms(run)
    return {
        "experiment": experiment,
        "device": str(device),
        "method": method,
        "steps": steps,
        "batch": batch,
        "ms": ms,
        "plans per s": batch / (ms / 1e3),
        "timed calls": reps,
    }


def benchmark(
    horizons=HORIZONS,
    batches=BATCHES,
    batch_steps=BATCH_STEPS,
    devices=None,
    beta=10.0,
    save=True,
):
    """Run experiments A (trace length) and B (batch) on each device.

    Args:
        horizons: Trace lengths for experiment A (batch 1).
        batches: Batch sizes for experiment B.
        batch_steps: Trace length for experiment B.
        devices: Devices to time (default: CPU, plus CUDA if available).
        beta: Smoothing strength.
        save: Write outputs/timing_benchmark.json and the figure.

    Returns:
        Dict with "rows", "largest difference" and "machine".
    """
    if devices is None:
        devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
    generator = torch.Generator().manual_seed(0)
    rows, difference = [], 0.0
    for device in devices:
        for steps in horizons:
            difference = max(
                difference,
                largest_difference(steps, 2, device, beta, generator),
            )
            for method in METHODS:
                rows.append(
                    _row("A", device, method, steps, 1, beta, generator)
                )
                print(_line(rows[-1]), flush=True)
        for batch in batches:
            for method in METHODS:
                rows.append(
                    _row(
                        "B",
                        device,
                        method,
                        batch_steps,
                        batch,
                        beta,
                        generator,
                    )
                )
                print(_line(rows[-1]), flush=True)
    result = {
        "rows": rows,
        "largest difference": difference,
        "beta": beta,
        "machine": _machine(devices),
    }
    if save:
        output_path("timing_benchmark", ".json").write_text(
            json.dumps(result, indent=2)
        )
        from visualization.figures import plot_timing

        plot_timing(result, save_path=output_path("timing_benchmark", ".pdf"))
    return result


def _line(row):
    return (
        f"{row['experiment']} {row['device']:4s} {row['method']:9s} "
        f"T={row['steps']:4d} B={row['batch']:5d}  {row['ms']:9.2f} ms"
    )


def _machine(devices):
    """CPU model, torch threads and version, allowed CPUs, GPU name."""
    gpu = None
    if any(torch.device(d).type == "cuda" for d in devices):
        gpu = torch.cuda.get_device_name(0)
    return {
        "cpu": cpu_model(),
        "threads": torch.get_num_threads(),
        "allowed cpus": sorted(os.sched_getaffinity(0)),
        "gpu": gpu,
        "torch": torch.__version__,
    }

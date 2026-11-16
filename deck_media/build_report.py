"""Build the human report and complete artifact/numeric inventories."""

import hashlib
import importlib.metadata
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def describe(name):
    if name.startswith("smoothing_window_step"):
        number = int(name.split("step")[1].split(".")[0])
        return [
            "Scores only",
            "Hard minimum and derivative spike",
            "Smooth minimum and derivatives, beta 5",
            "Smooth minimum and derivatives, beta 20",
            "Smooth minimum and derivatives, beta 100",
        ][number - 1]
    if name.startswith("register_step"):
        number = int(name.split("step")[1].split(".")[0])
        return [
            "r3 has left upward",
            "r2 and r1 shifted upward",
            "r0 inserted",
            "Lower column minimum",
            "Upper column minimum",
            "Purple min interval",
            "Purple max interval",
            "Affine recurrence and shift/insertion matrices",
        ][number - 1]
    if name.startswith("reach_avoid_optimizer_step"):
        number = int(name.split("step")[1].split(".")[0])
        return [
            "Initial trajectory",
            "Trajectory after 50 updates",
            "Trajectory after 150 updates",
            "Trajectory after 300 updates",
            "Trajectory after 450 updates",
            "Terminal trajectory after 600 updates",
            "Selected output trajectory",
        ][number - 1]
    if name.startswith("reach_avoid_iterates_"):
        return (
            "Initial parameters and 600 exact hook-recorded iterates, controls, full state paths, loss, beta, selected iteration: "
            + name.removeprefix("reach_avoid_iterates_").removesuffix(".npz")
        )
    descriptions = {
        "smoothing_window.mp4": "Five-stage smoothing time plot with pause holds",
        "smoothing_window_final.png": "Final smoothing frame, identical to step 5",
        "smoothing_values.csv": "Full precision smin, outer bound and all eleven derivatives for each beta",
        "smoothing_values.json": "Teaching inputs, numeric table and repository/autograd comparisons",
        "smoothing_weights.md": "Readable derivative table for all three beta values",
        "predicate_bounds.png": "Non-exact OutsideRectangle intervals and exact Gaussian probability",
        "predicate_bounds.json": "Actual STLCG/high belief from Task E, times, geometry, exact probabilities and repo bounds",
        "register.mp4": "Moving-row shift register and temporal min/max readouts",
        "register_final.png": "Final register frame, identical to step 8",
        "reach_avoid_paths.png": "Selected STLCG and pdSTL paths with pointwise 95% position ellipses at all noise levels",
        "reach_avoid_paths.pdf": "Vector/PDF copy of the reach-avoid figure",
        "reach_avoid_run.json": "Single-run provenance, input source hashes, config, candidates, selected plans, full means and covariances",
        "reach_avoid_run.log": "Console record of the one experiment run",
        "reach_avoid_monte_carlo.csv": "Counts, empirical satisfaction, confidence limits and separate planned/evaluation scores",
        "reach_avoid_monte_carlo.md": "Readable Monte Carlo results table",
        "reach_avoid_optimizer.mp4": "High-noise pdSTL optimizer replay from recorded iterations",
        "reach_avoid_optimizer_final.png": "Final optimizer replay frame, identical to step 7",
        "run_reach_avoid.py": "Run unchanged source experiment with passive iteration recording",
        "render_media.py": "Render all teaching and experiment media",
        "validate_media.py": "Validate numerical data, all saved paths, and fully decoded H.264 videos",
        "reproduce.py": "Archive the pinned source and recreate the run, media and validation",
        "build_report.py": "Rebuild this report and inventories from delivered data",
        "requirements-media.txt": "Additional media encoder dependency",
        "environment.json": "Versions used for this run and rendering",
        "numeric_inventory.json": "Map every on-screen number to its exact input, calculation or run artifact",
        "manifest.json": "Per-file descriptions, byte sizes and SHA-256 digests (self-digest omitted)",
        "INVENTORY.md": "One row for every deliverable file",
        "README.md": "Report, usage, results and material limitations",
        "validation.json": "Completed numerical/media checks and decoded video metadata",
        "Rubik.ttf": "Original Google Fonts Rubik variable font used to derive the static instances",
        "Rubik-Regular.ttf": "Rubik weight 400 static instance for all ordinary text",
        "Rubik-Bold.ttf": "Rubik weight 700 static instance for headings",
        "Rubik-LICENSE.txt": "SIL Open Font License and Rubik copyright notice",
        ".gitignore": "Exclude Python caches and retain the console run log",
    }
    return descriptions[name]


def main():
    smooth = json.loads((HERE / "smoothing_values.json").read_text())
    run = json.loads((HERE / "reach_avoid_run.json").read_text())
    validation = json.loads((HERE / "validation.json").read_text())
    versions = {
        package: importlib.metadata.version(package)
        for package in [
            "torch",
            "numpy",
            "scipy",
            "matplotlib",
            "Pillow",
            "PyYAML",
            "imageio-ffmpeg",
            "fonttools",
        ]
    }
    (HERE / "environment.json").write_text(
        json.dumps(versions, indent=2) + "\n"
    )
    weights = [
        "| time step j | L_j | derivative, beta 5 | derivative, beta 20 | derivative, beta 100 |",
        "|---:|---:|---:|---:|---:|",
    ]
    for j, value in enumerate(smooth["teaching_values"]):
        weights.append(
            f"| {j} | {value:.2f} | "
            + " | ".join(f"{row[f'w{j}']:.10g}" for row in smooth["rows"])
            + " |"
        )
    (HERE / "smoothing_weights.md").write_text(
        "# Smoothing derivatives (teaching values)\n\n"
        + "\n".join(weights)
        + "\n\nValues are derivatives, not event probabilities. Full precision is in the CSV/JSON.\n"
    )
    numeric = {
        "global": {
            "frame_px": [1920, 1080],
            "predicate_frame_px": [2000, 780],
            "dpi": 100,
            "smallest_text_pt": 17,
            "smallest_text_px_at_1920": 17 * 100 / 72,
            "fps": 24,
            "stage_holds_s": {
                "smoothing": 43 / 24,
                "register": 1.5,
                "optimizer": 43 / 24,
            },
        },
        "smoothing": {
            "data": "smoothing_values.json",
            "time_ticks_steps": list(range(11)),
            "score_y_limits": [0, 1.05],
            "score_ticks": [0, 0.2, 0.4, 0.6, 0.8, 1],
            "derivative_y_limits": [0, 1],
            "derivative_ticks": [0, 0.25, 0.5, 0.75, 1],
            "hard_min": 0.4,
            "hard_min_index": 4,
            "runner_up": 0.45,
            "runner_up_index": 5,
            "hard_min_derivatives": [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            "m": 11,
            "beta": [5, 20, 100],
            "decimal_places_shown": 6,
            "outer_rule": "min(L)-log(11)/beta <= smin <= min(L)",
            "displayed_numeric_bounds": [
                {
                    k: r[k]
                    for k in ["beta", "bound_lower", "smin", "bound_upper"]
                }
                for r in smooth["rows"]
            ],
        },
        "predicate": {
            "data": "predicate_bounds.json",
            "source_plan": "reach_avoid_run.json:predictions.stlcg/high",
            "time_range_s": [0, 12],
            "samples": 61,
            "time_step_s": 0.2,
            "time_ticks_s": [0, 2, 4, 6, 8, 10, 12],
            "P_ticks": [0, 0.2, 0.4, 0.6, 0.8, 1],
            "mean_and_covariance": "predicate_bounds.json",
            "q": run["config"]["noise"]["high"],
            "spread_at_10s_cm": 50,
            "rectangle_m": {"x": [4, 5.5], "y": [0, 3]},
            "joint_probability": "1 - p_x * p_y (independent Gaussian axes)",
            "bounds": "[1-min(p_x,p_y), min(1,2-p_x-p_y)]",
        },
        "register": {
            "teaching_values": True,
            "window_steps": [0, 2],
            "memory_shape": [3, 2],
            "r0": [0.7, 0.85],
            "r1": [0.4, 0.6],
            "r2": [0.9, 0.95],
            "r3": [0.8, 0.9],
            "initial_memory": [[0.8, 0.9], [0.9, 0.95], [0.4, 0.6]],
            "final_memory": [[0.9, 0.95], [0.4, 0.6], [0.7, 0.85]],
            "min_output": [0.4, 0.6],
            "max_output": [0.9, 0.95],
            "M": [[0, 1, 0], [0, 0, 1], [0, 0, 0]],
            "e2": [0, 0, 1],
            "e_indexing": "zero based",
            "equation": "H_k = M H_(k+1) + e_2 rhat_k",
        },
        "reach_avoid": {
            "data": "reach_avoid_run.json",
            "Monte_Carlo": "reach_avoid_monte_carlo.csv",
            "all_optimizer_data": "reach_avoid_iterates_*.npz",
            "config": run["config"],
            "axis_limits_m": [-0.4, 11.6],
            "axis_ticks_m": [0, 2, 4, 6, 8, 10],
            "ellipse_confidence": 0.95,
            "ellipse_chi_squared_2d": 5.991464547107979,
            "ellipse_every_steps": 5,
            "ellipse_every_s": 1,
            "full_horizon_steps": 60,
            "formula_windows": run["formula_windows"],
            "full_horizon_s": 12,
            "goal_dwell_steps": 0,
            "reference_spread_time_s": 10,
            "spread_cm": [5, 20, 50],
            "q": run["config"]["noise"],
            "displayed_q_decimal_places": 6,
            "MC_rollouts_per_plan": 5000,
            "MC_seed": 7,
            "MC_CI": 0.95,
            "optimizer_updates_per_solve": 600,
            "solves": 8,
            "optimizer_beta_start": 10,
            "optimizer_beta_end": 200,
            "replay_iteration_indices": [-1, 49, 149, 299, 449, 599, 599],
            "replay_completed_updates": [0, 50, 150, 300, 450, 600, 600],
        },
        "encoded_videos": validation["videos"],
    }
    (HERE / "numeric_inventory.json").write_text(
        json.dumps(numeric, indent=2) + "\n"
    )
    video_rows = "\n".join(
        f"| [{name}]({name}) | {v['duration_s']:.3f} s | {v['bytes'] / 1e6:.3f} MB |"
        for name, v in validation["videos"].items()
    )
    smooth_rows = "\n".join(
        f"| {r['beta']} | {r['smin']:.12f} | {r['bound_lower']:.12f} ≤ smin ≤ 0.400000000000 |"
        for r in smooth["rows"]
    )
    readme = f"""# ECEN 5773 pdSTL deck media

Bennett Dogbey · iHuman Lab · generated 16 November 2026.

Tasks C, A, B and E are complete. Crazyflie is excluded at the user's request. The optional original STLCG paper figure was not copied. Slide-deck editing was outside the requested media scope.

All plots use white backgrounds, Rubik text (minimum 23.6 px at 1920 wide), the requested blue/orange/purple/gray palette and mathtext for mathematical symbols. Animation axes stay fixed. All videos are H.264 MP4 with pause holds and a PNG for every stage plus the final frame. Task A uses the explicitly requested 2000×780 size; the other PNGs and videos are 1920×1080.

| Video | Duration | File size |
|---|---:|---:|
{video_rows}

Use [INVENTORY.md](INVENTORY.md) for **every filename and what it shows**, [manifest.json](manifest.json) for file hashes and [numeric_inventory.json](numeric_inventory.json) for **every number shown on screen** and its source. Each animation's `_final.png` is the fallback still for exported slides.

## Smoothing over a time window (priority C)

![Smooth minimum over time](smoothing_window_step3.png)

The supplied eleven scores and beta values are labeled “teaching values.” The derivative is a line with dots over time, with a 0–1 axis; it is never labeled probability. The unique hard minimum is 0.40 at j=4, with runner-up 0.45 at j=5. Stages: scores; hard minimum/spike; beta 5; beta 20; beta 100.

| beta | smin | Verified numeric bound, m=11 |
|---:|---:|---|
{smooth_rows}

The calculation and autograd derivatives agree with `src/pdstl/operators.py:_reduce_bounds(Minish(), x, beta)` at the pinned main commit to floating-point precision. There is **no mismatch**. The smooth lower minimum is the **unnormalized** negative log-sum-exp; the normalized `Minish` result alone would be different. Stages before a beta is chosen show the numeric hard minimum and symbolic beta-dependent bound.

The complete derivative table is [smoothing_weights.md](smoothing_weights.md); [CSV](smoothing_values.csv) and [JSON](smoothing_values.json) retain full precision.

## Predicate bounds (A)

![Predicate probability intervals](predicate_bounds.png)

The repo has genuinely non-exact rectangular-event bounds. `pdstl.predicates.OutsideRectangle.robustness_trace` evaluates `Negation(And(AxisInterval, AxisInterval))`; `models.beliefs.GaussianBelief.probability` supplies exact Gaussian axis probabilities, and `pdstl.operators._conjunction` combines them using Boole–Fréchet rules.

The belief is **from the same fresh Task E run**: the selected STLCG plan under high evaluation noise, q=0.01224377 (50 cm position standard deviation at 10 s), at all 61 times over 0–12 s. The event is outside the run's floor obstacle `[4,5.5] × [0,3]` m. The linear-Gaussian dynamics preserve independence between the two axes, so the exact outside probability is `1-p_x*p_y`; the repo interval is `[1-min(p_x,p_y), min(1,2-p_x-p_y)]`. Every exact value lies in its interval. The black line is exact; blue/orange ticks mark L/U; gray segments connect bounds. [Complete means, covariances and event data](predicate_bounds.json). This replaces the digitized GIF belief with a reproducible experimental belief and uses its actual 12 s horizon.

## Shift register (B)

The video begins at `H_1=[r3;r2;r1]`. Rows visibly fade/translate between fixed slots, then r0 enters. The min readout is `[0.40,0.60]`; the same memory gives max `[0.90,0.95]`. The shift M is highlighted with row motion, and insertion e2 with the incoming row. Eight stills correspond to the requested eight stages. Every trace value, matrix entry and output is in the numeric inventory.

## Reach–avoid, one fresh run (E)

![Paths and pointwise uncertainty](reach_avoid_paths.png)

**Source:** `iHuman-Lab/probabilistic-dstl`, main commit `{run["source_commit"]}`. A clean `git archive` was used, preserving the existing working branch and edits. Every source/config byte was checked against the commit; SHA-256 values are in [reach_avoid_run.json](reach_avoid_run.json).

**Command:** `PDSTL_DEVICE=cpu python deck_media/run_reach_avoid.py --source /tmp/pdstl-deck-main`

**Runtime:** {run["runtime_s"]:.3f} s, including all eight solves, iteration capture, and six Monte Carlo evaluations. **Seed:** 7. CPU, one Torch thread; Torch {run["torch_version"]}, NumPy {run["numpy_version"]}; other versions in [environment.json](environment.json).

The runner invokes the same optimize/validate stages as the risk-aversion blocks in `src/main.py`, using `src/experiments/risk_aversion.py` and the unchanged YAML. It only adds passive iteration recording. Each of eight candidates has its initial theta plus all 600 post-update parameters from the planner's iteration hook. A wrapper copies the actual unconstrained parameters passed to the rollout; theta is **not reconstructed** from rounded controls. All 601 complete mean paths per solve are saved in the NPZ files.

**Formula mismatch:** the **built formula** is `{run["formula"]}` with 0.2 s steps and a 12 s horizon. Both windows begin at step **1**, and there is **no one-second stay-in-goal requirement**. The experiment module's comments describe windows beginning at 0; the actual `Environment.get_specification` result begins at 1, as confirmed by printing the built specification. The experiment code and all computed results were preserved. Report the slide as reach–avoid, or separately change/re-run the scientific experiment before claiming reach-and-stay results.

q denotes process-noise standard deviation in `Q=q²I`. The parenthesized 5/20/50 cm values are per-axis position standard deviations **after 10 s**, not at the 12 s horizon. The figure shows full-horizon paths and pointwise 95% Gaussian position ellipses (2-D chi-square threshold 5.9914645471) every 1 s for **both methods under the same evaluation noise**. These are individual-time regions. The STLCG planner itself is noise-free.

{(HERE / "reach_avoid_monte_carlo.md").read_text().strip()}

Each cell uses 5,000 rollouts with common random draws and 95% Clopper–Pearson confidence intervals. “Planned score” uses the planner's own model; the additional evaluation score re-scores each selected plan at that row's noise. Scores are **pdSTL robustness values, not trajectory satisfaction probabilities**. Empirical satisfaction is separately estimated from sampled trajectories. At high noise the pdSTL planned score is below its 0.99 target; no guarantee is implied.

The replay shows the selected high-noise pdSTL candidate at initial, 50, 150, 300, 450 and 600 updates, then its selected output. The selected iteration in this solve is 599 (600 completed updates), so the final two trajectories coincide. Both index and update count are displayed. All other candidates' iterates are also supplied.

## Recreate and verify

Install the project's dependencies and `imageio-ffmpeg==0.8.1` (the only extra runtime media dependency), then run:

```bash
PDSTL_DEVICE=cpu python deck_media/reproduce.py
python deck_media/build_report.py
```

The reproduction script archives the pinned commit into a temporary directory, runs the experiment, renders all assets and validates them. Re-running overwrites generated media/data in this folder. To check the supplied files without re-optimizing:

```bash
python deck_media/validate_media.py
```

Validation covers all smoothing bounds/derivatives, non-exact predicate containment, register calculations, every saved theta/control/path, selected-plan identity, Monte Carlo confidence intervals, every PNG size and complete decoding of all three MP4s. It also checks that videos end on their exported final frames. Results: [validation.json](validation.json). The pinned source's focused pdSTL/risk-aversion tests passed: **143 passed** (`PYTHONPATH=/tmp/pdstl-deck-main/src python -m pytest /tmp/pdstl-deck-main/tests/test_risk_aversion.py /tmp/pdstl-deck-main/tests/test_pdstl.py -m 'not slow' -q`).

Rubik comes from [Google Fonts](https://github.com/google/fonts/tree/main/ofl/rubik); weights 400 and 700 were instantiated from the bundled variable font. Redistribution terms: [Rubik-LICENSE.txt](Rubik-LICENSE.txt).
"""
    (HERE / "README.md").write_text(readme)
    names = sorted(
        {p.name for p in HERE.iterdir() if p.is_file()}
        | {"INVENTORY.md", "manifest.json"}
    )
    inventory = [
        "# File inventory",
        "",
        "| File | What it shows or contains |",
        "|---|---|",
    ]
    for name in names:
        inventory.append(f"| [{name}]({name}) | {describe(name)} |")
    (HERE / "INVENTORY.md").write_text("\n".join(inventory) + "\n")
    manifest = []
    for name in names:
        p = HERE / name
        manifest.append(
            {
                "file": name,
                "description": describe(name),
                "bytes": None if name == "manifest.json" else p.stat().st_size,
                "sha256": None
                if name == "manifest.json"
                else hashlib.sha256(p.read_bytes()).hexdigest(),
            }
        )
    (HERE / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()

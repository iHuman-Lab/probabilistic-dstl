"""Run unchanged main-branch experiment with passive iterate recording.

Usage: PDSTL_DEVICE=cpu python deck_media/run_reach_avoid.py --source /tmp/pdstl-deck-main
The source directory must be a clean archive of the commit in SOURCE_COMMIT.
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

SOURCE_COMMIT = "12b3bd7987faab130ebcf73120aeddbf77f17085"
DEST = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    args = parser.parse_args()
    source = args.source.resolve()
    repo = DEST.parent
    # Refuse to attach the advertised commit to different experiment code.
    paths = ["src", "configs"]
    expected = subprocess.check_output(
        ["git", "ls-tree", "-r", "--name-only", SOURCE_COMMIT, *paths],
        cwd=repo,
        text=True,
    ).splitlines()
    hashes = {}
    for name in expected:
        original = subprocess.check_output(
            ["git", "show", f"{SOURCE_COMMIT}:{name}"], cwd=repo
        )
        actual = (source / name).read_bytes()
        if original != actual:
            raise RuntimeError(f"Source differs from {SOURCE_COMMIT}: {name}")
        hashes[name] = hashlib.sha256(actual).hexdigest()
    os.chdir(source)
    os.environ["PDSTL_DEVICE"] = "cpu"
    torch.set_num_threads(1)
    torch.manual_seed(7)
    sys.path.insert(0, str(source / "src"))
    from experiments import risk_aversion as risk

    started = time.perf_counter()
    histories = {}

    def recorded_plan(cfg, method, noise):
        rows = []
        for gap, waypoints in cfg["gaps"].items():
            s = risk.problem(cfg, method, noise)
            guess = risk.initial_guess(s.cfg, s.dyn, waypoints)
            rollout = s.rollout
            latest = {}
            theta, controls, iterations, losses, betas = [], [], [], [], []

            def observer_rollout(parameters, latest=latest, rollout=rollout):
                latest["theta"] = parameters.detach().cpu().clone()
                return rollout(parameters)

            theta.append(
                s.planner._init_controls(guess).detach().cpu().numpy()
            )
            controls.append(guess.detach().cpu().numpy())
            iterations.append(-1)
            losses.append(float("nan"))
            betas.append(float("nan"))

            def on_iteration(
                iteration,
                record,
                theta=theta,
                latest=latest,
                controls=controls,
                iterations=iterations,
                losses=losses,
                betas=betas,
                gap=gap,
            ):
                theta.append(latest["theta"].numpy().copy())
                controls.append(record.controls.detach().cpu().numpy().copy())
                iterations.append(iteration)
                losses.append(record.loss)
                betas.append(record.beta)
                if (iteration + 1) % 100 == 0:
                    print(
                        f"{method}/{noise}/{gap}: {iteration + 1}/600 updates",
                        flush=True,
                    )

            result = s.planner.optimize_window(
                observer_rollout,
                spec=s.spec,
                init_guess=guess,
                on_iteration=on_iteration,
                callback_every=1,
            )
            key = f"{method}_{noise or 'nominal'}_{gap}"
            filename = f"reach_avoid_iterates_{key}.npz"
            arrays = {
                "theta": np.stack(theta),
                "controls": np.stack(controls),
                "iterations": np.asarray(iterations),
                "loss": np.asarray(losses),
                "beta": np.asarray(betas),
                "selected_iteration": result.selected_iteration,
            }
            assert (
                len(iterations)
                == cfg["methods"][method]["planner"]["max_iters"] + 1
            )
            np.testing.assert_allclose(
                np.tanh(arrays["theta"]) * s.dyn.u_max,
                arrays["controls"],
                atol=2e-7,
            )
            # Store a full path for every iterate using the same linear dynamics.
            state = np.broadcast_to(
                s.state[0].cpu().numpy(), (len(theta), 4)
            ).copy()
            means = [state.copy()]
            a, b = s.dyn.A.cpu().numpy(), s.dyn.B.cpu().numpy()
            for k in range(cfg["H"]):
                state = state @ a.T + arrays["controls"][:, k] @ b.T
                means.append(state.copy())
            arrays["mean_trajectories"] = np.stack(means, axis=1)
            np.savez_compressed(DEST / filename, **arrays)
            histories[key] = {
                "file": filename,
                "selected_iteration": result.selected_iteration,
                "planning_time_s": result.planning_time,
            }
            mean = result.rollout.aux["mean_trace"]
            rows.append(
                {
                    "method": method,
                    "noise": noise,
                    "start": gap,
                    "gap": risk.gap_used(cfg, mean[0].cpu().numpy()),
                    "robustness": risk.mean_robustness(s.spec, mean),
                    "score": result.hard_interval[0],
                    "effort": risk.effort(result.controls),
                    "time_s": result.planning_time,
                    "controls": result.controls.detach().cpu(),
                }
            )
            print(
                f"saved {filename}; selected iterate {result.selected_iteration}",
                flush=True,
            )
        return rows

    risk.plan = recorded_plan
    cfg = risk.load()
    candidates, selected, saved_plans, predictions = [], [], {}, {}
    for method in ("stlcg", "pdstl"):
        c, picks = risk.optimize(cfg, method)
        candidates.extend(c)
        selected.extend(risk.validate(cfg, picks))
        for (m, level), pick in picks.items():
            key = f"{m}/{level}"
            saved_plans[key] = pick["controls"].numpy().tolist()
            s = risk.evaluation_problem(cfg, level)
            mean, cov = s.state
            means, covs = [mean], [cov]
            for u in pick["controls"]:
                mean, cov = s.dyn.step(mean, cov, u)
                means.append(mean)
                covs.append(cov)
            predictions[key] = {
                "mean": torch.stack(means).tolist(),
                "covariance": torch.stack(covs).tolist(),
            }
    built_spec = risk.evaluation_problem(cfg, "high").spec
    safe_window = list(built_spec.subformula1.interval)
    goal_window = list(built_spec.subformula2.interval)
    result = {
        "source_commit": SOURCE_COMMIT,
        "source_sha256": hashes,
        "command": "PDSTL_DEVICE=cpu python deck_media/run_reach_avoid.py --source /tmp/pdstl-deck-main",
        "seed": 7,
        "threads": torch.get_num_threads(),
        "device": "cpu",
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "runtime_s": time.perf_counter() - started,
        "config": cfg,
        "candidates": candidates,
        "selected": selected,
        "controls": saved_plans,
        "predictions": predictions,
        "histories": histories,
        "formula": f"G_[{safe_window[0]},{safe_window[1]}] safe AND F_[{goal_window[0]},{goal_window[1]}] goal",
        "formula_built": str(built_spec),
        "formula_windows": {"safe": safe_window, "goal": goal_window},
        "formula_mismatch": "Both windows start at step 1; there is no goal dwell. The deck requests windows beginning at 0 and a 1-second reach-and-stay clause.",
    }
    (DEST / "reach_avoid_run.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(
        f"Completed one run in {result['runtime_s']:.1f} seconds", flush=True
    )


if __name__ == "__main__":
    main()

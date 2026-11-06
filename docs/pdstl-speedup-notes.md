# pdSTL speed-up: what changed, why it is the same method, how to defend it

Branch `lane-merge-clean`, 2026-11-05. Commits `845ffd4`, `694dd93`, `c5793ee`,
`32e21de` (smooth `[a, end]` windows now follow Eq. (15)) and `ef56383`
(figure/table notation), plus `827dbf3` earlier the same day for the lane task.

> Tables below are from the final pinned batch after `32e21de`
> (`outputs/lane_*_batch.json`, 2026-11-05 19:13).

## The short version

Replanning in the lane merge went from about 0.5–0.7 s per 0.2 s step to about
0.17–0.18 s (median, one CPU core). Nothing about **what** is computed changed:
the pdSTL semantics, the Fréchet bounds, the smooth surrogate, the formulas, the
planner and every config are the same. (One deliberate fix, `32e21de`: smooth
`[a, end]` windows now follow Eq. (15) exactly; see below.) What changed is **how** the computer
evaluates them: fewer, larger tensor operations instead of thousands of tiny
ones, and no repeated work. Tests compare the new code against a copy of the
old code and agree to 1e-5 in values **and** gradients.

## Why it was slow

Each replan runs 15 optimizer iterations. Each iteration evaluates the formula
twice (smooth for the gradient, exact for the reported score) on a 31-step
belief trajectory and back-propagates through it. The traces are tiny (31 steps,
a handful of numbers), so the arithmetic is negligible; the cost is the
~5–10 µs PyTorch spends on **every** operation call, forwards and backwards.

The old temporal operators walked the trace one step at a time (the register
recursion, Eq. (12) in the draft): ~360 Python-level steps per evaluation, each
6–8 small operations. That alone was most of the time. A GPU was even slower,
because each tiny operation becomes a kernel launch.

## What changed

### 1. Temporal operators evaluate all windows at once (`src/pdstl/operators.py`)

`Temporal_Operator` (line ~282), used by Always (G) and Eventually (F).

**Bounded window `[a, b]`** (`_window_trace`). The output at origin k is

$$\hat r^{G_{[a,b]}\psi}_k=\Big[\min_{j=k+a}^{k+b}L^\psi_j,\ \min_{j=k+a}^{k+b}U^\psi_j\Big]$$

(Eq. (9)). Old code: build the register $H_s$ step by step (Eq. (12)), and at
each step reduce rows $k+a..k+b$. New code: the register at origin k is just
the slice $k+a..k+b$ of the trace, so all slices are laid side by side
(`unfold`) and reduced in one call. Same min/max, same smooth $s_\beta^\pm$
with the same $\delta_{|I|}=\log|I|/\beta$ shift (Eqs. (15)–(16)).

> Example, G[0,2] over 6 steps: old = walk backwards 6 times keeping the last 3
> values; new = form windows [0..2],[1..3],[2..4],[3..5] at once and take each
> minimum. Same four numbers.

**Window to the end of the plan `[a, end]`** (`_tail_trace`). Used by the lane
formulas, e.g. $\Box_{[d,\mathrm{end}]}(\text{lane}\wedge\text{settled})$. At origin
k the window is $k+a,\dots,T-1$, so it has $n_k=T-k-a$ values. The new code
reduces each origin's window exactly as Eq. (9) and Eqs. (15)–(16) say: the
exact value is the suffix min/max (`cummin`/`cummax` from the trace end), and
the smooth value is $s^\mp_\beta$ over the $n_k$ values with the
$\delta_{n_k}=\log n_k/\beta$ shift, computed for all origins with one
cumulative log-sum-exp from the end.

> History (commit `32e21de`): the old recursive code, and the first vectorized
> version that copied it, computed the **smooth** `[a, end]` value by folding
> pairs from the end of the trace, which is sound but is not Eq. (15). This was
> changed so the code matches the paper. Exact values (the reported scores) were
> the same before and after; the smooth objective changed slightly, so the lane
> batch was rerun.

**Until `ψ1 U[a,b] ψ2`** (`Until._window_trace`, line ~458). Eq. (10): for each
origin k and witness j, the prefix min of ψ1 over $k..j$, conjoined with ψ2 at
j, then the max over witnesses. Old code: two nested Python loops (origins ×
witnesses). New code: for each origin's window, all prefix minima come from one
cumulative min (exact) or one cumulative log-sum-exp (smooth, with each prefix's
own $\log n/\beta$ shift); then one conjunction and one max for all origins at
once. Unbounded Until (`[a, end]`) still uses the old loop; the lane formulas do
not use it.

**Predicate cache** (line ~119). A belief trajectory that carries a
`bounds_cache` (Gaussian trajectories now do) computes each atom's
$[p^\downarrow_k,p^\uparrow_k]$ once. Atom bounds do not depend on β, so the
smooth and exact passes of one iteration share them.

### 2. Rollout: compute control-free parts once (`src/models/rollouts.py`)

`lane_rollout` (line ~162). The ego covariance $P_{k+1}=AP_kA^\top+Q$, the
traffic means and the traffic covariances do not depend on the controls, so
they are now predicted once per planning window (`extend`) instead of in all 18
rollouts of that window. The ego mean still steps through time, because the
longitudinal acceleration limit depends on the speed reached so far (keeps
$v_x$ in bounds). `tanh` and the lateral clamp act on all steps at once.

### 3. Two small identical-value changes

- `interval_probability` (`src/models/beliefs.py`, line ~16):
  $P(\ell\le X\le h)$ is $\Phi(-a)-\Phi(-b)$ above the mean and $\Phi(b)-\Phi(a)$
  otherwise (for tail accuracy). It used to compute both (four CDFs) and pick
  one; now it picks the arguments first (two CDFs). Same number.
- `check_probability_bounds` (`src/pdstl/base.py`, line ~41): one combined test
  that every bound is in [0, 1] and lower ≤ upper; only if that fails does it
  run the detailed checks that build the error message. NaN and ±inf still
  fail (NaN fails every comparison).

### 4. Batch records the machine (`src/experiments/lane.py`)

`_machine` now saves the CPU model, the torch thread count and the CPUs the
process was allowed to run on, so a timing claim can be traced to its setup.

## How we know the answers did not change

- `tests/test_pdstl.py::test_temporal_windows_are_the_window_reductions`
  — 56 cases: G and F; windows `None`, `[0,end]`, `[3,end]`, `[0,0]`, `[0,4]`,
  `[2,7]`, `[5,5]`; β = exact, 0.5, 3, 40; random traces whose smooth lowers
  leave [0, 1]. Every case is compared (values **and gradients**) with a
  direct per-origin reduction written straight from Eqs. (9), (15)–(16).
  Bounded windows and all exact windows are also compared with
  `_LoopTemporal`, a verbatim copy of the old recursive code (Eq. (12)) kept
  in the test file.
- `test_vectorized_until_matches_the_step_by_step_loop` — 16 cases, same idea
  against the old nested loop.
- Full suite: 402 tests pass (and runs in 68 s instead of 229 s).
- In the lane batch, the first planning window of every seed matches the old
  code to ~1e-7.

## Why the closed-loop runs still differ a little

Floating-point addition is not associative: summing in a different order
(one cumulative sum vs. a chain of pairs, one thread vs. twelve) changes the
last bits. A 15-iteration optimizer inside a 38-step closed loop amplifies
those bits, so runs drift apart after 4–34 steps. Effect on results:

| Scene | Old code | Vectorized, old `[a, end]` smoothing | Final (`32e21de`) |
|---|---|---|---|
| Merge | 20/20 success | 20/20 success | 20/20 success |
| Merge + headway | 11 success / 9 violation | 10 / 10 | 11 / 9 |
| Dense | 20/20 yielded | 20/20 yielded | 20/20 yielded |
| One gap | 20/20 success | 20/20 success | 20/20 success |

The middle column differs from the old code by float rounding only (window 0
equal to ~1e-7; 7 headway seeds flipped, both directions). The final column
also has the slightly different smooth objective of `32e21de`. Headway moves
between 9 and 10 violations because that scene holds the gap at the 22 m
limit, so tiny differences decide the outcome; report it as a boundary case.

One visible effect of `32e21de`: in 2 of 20 merge runs (seeds 6 and 8) the
exact score drops to 0 for 2–3 windows right after the lead brakes
(4.8–5.8 s). Both runs still succeed, with clearance ≥ 2.0 m. Before
`32e21de` the lowest merge window was 0.87 (the first window).

## Timing (pinned batch, 20 seeds per scene)

One core of an Intel Core i7-12700K (`taskset -c 10`, `OMP_NUM_THREADS=1`),
replans after the first:

| Scene | Before (ms, median / p95) | After (ms, median / p95 / max) | Over 200 ms |
|---|---|---|---|
| Merge | 467 / 953 | 169 / 183 / 250 | 0.8% |
| Merge + headway | 671 / 1020 | 182 / 193 / 248 | 3.0% |
| Dense | 692 / 819 | 174 / 191 / 251 | 0.7% |
| One gap | 710 / 1134 | 173 / 189 / 209 | 0.3% |

Why pinned: the 12700K has fast (P) and slow (E) cores; unpinned, the scheduler
moves the planner between them and timings swing by ±30 ms. Claim "median and
p95 within the step", not "every step".

## Questions you may be asked

**"Your paper says the evaluation is recurrent / LSTM-style. Is it?"**
The register recursion (Eqs. (11)–(13)) defines the operator: the register at
origin k holds the window $k+a..k+b$. Because each register state is just a
window of the trace, all of them can be formed at once, and the implementation
does that. Outputs are identical (tested against the recursive version).
Honest caveat: it was never an LSTM in the gated sense (no input/forget gates,
no cell state); it is a shift register. Fig. 2's "LSTM unit" should be
relabelled.

**"Is it still linear time?"**
Yes. Both forms do $O(T(b-a+1))$ work per bounded operator, and $O(T)$ for
`[a, end]` windows. The new form simply issues that work in a few calls.

**"Did the speed-up change the guarantees?"**
No. Propositions 1 and 2 hold for the exact values (unchanged) and for the
smooth values (now exactly Eqs. (15)–(16) for every window).
The reported score is still the exact evaluation; the smooth one is still only
the optimization objective.

**"Why are the new results not identical to the old ones?"**
Two reasons. Float rounding amplified by the closed loop (the first window
agrees to 1e-7), and `32e21de`, which made the smooth `[a, end]` windows follow
Eq. (15) and so changed the optimization objective slightly. Exact scores of
a given plan are unchanged by either. Outcome counts agree except at the
headway boundary.

**"Why is CPU faster than GPU?"**
Each window is a tiny problem; the GPU's per-operation launch cost dominates.
The GPU numbers in the archive predate the speed-up and should not be quoted.

**"Is it real time?"**
Median and 95th percentile are within the 0.2 s step on one CPU core; 0.3–3.0%
of steps exceed it (worst 251 ms). The simulation waits for each plan, so a
late step does not change the outcome; on a real car it would mean reusing the
previous plan's next input for one step.

## Suggested wording changes in the draft

- Abstract / intro / Sec. III intro: replace "recurrent, LSTM-style unfolding /
  computation graph" with "a formula-structured computation graph whose
  temporal operators are sliding-window (register) reductions".
- Sec. III-B.2, after Eq. (13): "Since each register state is a window of the
  trace, all $T-b$ states can be formed at once; our implementation evaluates
  them in parallel, which gives the same outputs as running (12) in sequence."
- Fig. 2: "Differentiable LSTM Unit" → "temporal window module" (or similar).

## If you need to undo it

```bash
git revert 32e21de 694dd93 845ffd4   # back to the recursive operators (slower)
```

That also brings back the pairwise smooth fold for `[a, end]` windows.

`c5793ee` (machine info) is independent and can stay. The previous outputs are
in `outputs/archive-2026-11-05-loop-operators/`.

## Also changed earlier the same day (lane task, not the core)

- `827dbf3`: once the ego is in the lane, the merge rule is one Until over the
  possible dwell finishes instead of an Or over each finish. Same exact lower
  bound (the Fréchet And lower is $\max(0,\sum L_i-(n-1))$ for any grouping);
  removed the slow p95 tail.
- `ce7b49c`: simulated traffic starts with zero lateral speed (keeps its lane);
  the planner's belief still allows lateral speed.

## Windows that end at the plan end (`END`)

The paper's grammar has finite windows only, and so does the code: every
trace is finite (one plan, N + 1 = H + 1 steps). The lane formulas need a
window that runs from the dwell finish **to the plan end**, because inside
$\Diamond_I$ the start s varies and the stay window is $s+d,\dots,N$, so its
length depends on s. A fixed $[d,b]$ would either stop early or run past the
plan end (and the late starts would be dropped as incomplete windows).

The code names that bound `END` (`src/pdstl/operators.py`); it prints as
`[a, end]`. At origin k, `G[a, END]` is the minimum over $k+a,\dots,N$: a
finite, nonempty window (when $k+a\le N$) with $|I| = N-k-a+1$. Eq. (9),
Eqs. (15)–(16) and Propositions 1–2 need nothing more. Internally `END` is
`np.inf`, so values are unchanged from before the rename (commit after
`ef56383`).

Suggested paper text (Sec. II-B or III-B.2): "Because each plan is finite, a
window may also end at the end of the plan: $\mathbf G_{[a,\mathrm{end}]}\psi$
at step k takes the minimum of each endpoint over $j=k+a,\dots,N$, and
$\mathbf F_{[a,\mathrm{end}]}$ the maximum. The index set is finite and
nonempty whenever $k+a\le N$, so (9), (15)–(16) and Propositions 1–2 apply
with $|I| = N-k-a+1$." In the lane section write $[d,\mathrm{end}]$ for
$[d,\infty)$, and the yield branch as $\Box_{[4.0,\,6]}\,\mathrm{stop}$
(what the code computes; the same event as
$\Diamond_{[2.4,4.0]}\Box_{[0,\mathrm{end}]}\mathrm{stop}$).

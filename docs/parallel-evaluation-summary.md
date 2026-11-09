# Parallel evaluation of pdSTL temporal operators

## Goal

Make pdSTL fast enough for closed-loop MPC without changing what it computes.
In the lane merge, each 0.2 s control step replans with 15 gradient
iterations, and each iteration evaluates the formula (smooth and exact) and
back-propagates through it. With the recurrent evaluation a replan took
0.47–0.71 s (median), 2–4× too slow for the control step.

## What was done

1. **Temporal operators (always, eventually, until) evaluate every time step
   at once** instead of stepping through the trace one sample at a time.
2. **No repeated work per iteration**: atom probabilities are shared by the
   smooth and exact passes; covariances and traffic predictions (independent
   of the controls) are computed once per planning window.
3. **Checked equal to the recurrent version** (values and gradients), then
   measured: a stand-alone benchmark and the lane-merge closed loop.

The idea of replacing the recurrent evaluation by parallel reductions is the
one STLCG++ [11] introduced for deterministic STL; here it is applied to
pdSTL's interval-valued, smooth-and-sound operators.

## Code changes compared with the recurrent version (`opt-planning`)

All in `src/pdstl/operators.py` unless noted.

### Always / Eventually, window `[a, b]`

**Before (`opt-planning`)**: a shift register runs backwards over the trace,
one step per sample (Eqs. (11)–(13) of the draft).

```python
def _run_cell(self, x, scale):
    outputs = []
    hc = self._initialize_rnn_cell(x)          # register filled with x[0]
    for xs_i in torch.split(x, 1, dim=1):      # one step per sample (T steps)
        o, hc = self._rnn_cell(xs_i, hc, scale)
        outputs.append(o)
    return torch.cat(outputs, dim=1)

def _rnn_cell(self, x, hc, scale=-1):          # Always, bounded [a, b]
    h0, c = hc
    new_h0 = self._apply_shift(h0, x)          # M @ H + e_b z   (Eq. 12)
    window = new_h0[:, : b - a + 1, :]         # rows k+a .. k+b
    output = self.operation(window, scale, dim=1, keepdim=True)  # min
    return output, (new_h0, None)
```

**After**: the register at step k is just the slice `k+a .. k+b` of the
trace, so all slices are formed at once and reduced in one call.

```python
def _window_trace(self, trace, beta):
    a, b = int(self._interval[0]), int(self._interval[1])
    windows = trace[:, a:].unfold(1, b - a + 1, 1).transpose(-1, -2)
    return _reduce_bounds(self.operation, windows, beta, dim=2, keepdim=False)
```

Same min/max (exact) or smooth s±β with the log n/β shift (Eqs. (15)–(16)).
Tensor operations per operator: about 8 per time step before (so ∝ T), a
handful after, independent of T.

### Always / Eventually, window to the end of the plan `[a, end]`

**Before**: a second register case that folded pairs from the end of the
trace, one step per sample.

**After**: one cumulative reduction from the trace end.

```python
suffix = trace[:, a:].flip(1)
exact  = torch.cummin(suffix, dim=1).values.flip(1)          # (cummax for F)
smooth = torch.logcumsumexp(sign * beta * suffix, dim=1).flip(1)  # + log n/β shift
```

This also makes the smooth value follow Eq. (15) exactly (the old pairwise
fold was sound but was not Eq. (15)). The window is finite: it ends at the
last step of the plan, written `END` in code and `[a, end]` in the paper.

### Until `φ U[a,b] ψ`

**Before (`opt-planning`)**: two nested Python loops, over origins t and
witnesses τ, each reducing the prefix of φ again.

```python
for t in range(T):
    for tau in range(t + a, t + b + 1):
        min_phi = self.min_op(phi[:, t:tau, :], ...)   # prefix min, recomputed
        val = ...combine(min_phi, psi[:, tau])
    best = self.max_op(all τ values, ...)
```

**After**: every origin's window at once; all prefix minima of a window from
one cumulative min (exact) or cumulative log-sum-exp (smooth).

```python
phi_w, psi_w = windows(phi), windows(psi)            # [B, origins, b+1, 2]
prefix = torch.cummin(phi_w, dim=2).values           # all prefixes at once
candidates = _conjunction(prefix[:, :, a:], psi_w[:, :, a:], beta)
return _reduce_bounds(self.max_op, candidates, beta, dim=2, keepdim=False)
```

### Supporting changes

| File | Change | Values |
|---|---|---|
| `pdstl/operators.py` (`Predicate`) | a belief trajectory caches each atom's bounds, so the smooth and exact passes share them | same |
| `models/rollouts.py` | covariances and traffic predicted once per window (they do not depend on u) | same |
| `models/beliefs.py` | interval probability computes 2 normal CDFs instead of 4; covariance check runs in chunks cuSOLVER accepts | same |
| `pdstl/base.py` | bound check in one pass when bounds are valid | same |
| `pdstl/recurrent.py` | the recurrent register kept as the reference and the benchmark baseline | — |

## Is it the same method?

- Tests compare the parallel operators with the recurrent register on random
  traces: 56 always/eventually cases and 16 until cases, exact and smooth
  (β = 0.5, 3, 40), values **and gradients**, all within 1e-5.
- Benchmark: the two versions agree to 1.2e-7 on every configuration.
- Propositions 1–2, Eqs. (8)–(10) and (15)–(16) are unchanged.

## Worked example: Always, recurrent vs parallel

$\mathbf G_{[1,2]}$ (a = 1, b = 2) on a 6-step trace, lower endpoint. The
register contents below are printed from the recurrent code.

| k | 0 | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|---|
| $L_k$ | 0.90 | 0.60 | 0.80 | 0.70 | 0.95 | 0.85 |

At origin k the answer is the min of $L_{k+1}, L_{k+2}$, for k = 0 … T−1−b = 0 … 3.

**Recurrent.** The trace is fed backwards, $z_s = L_{5-s}$. The register has
b + 1 = 3 rows and starts filled with $z_0$. Each step shifts the rows up and
writes $z_s$ into the last row; the output is the min of rows 0 … b−a.

| s | origin k | $z_s$ | register (rows 0, 1, 2) | min of rows 0–1 | kept? |
|---|---|---|---|---|---|
| start | | | 0.85, 0.85, 0.85 | | |
| 0 | 5 | 0.85 | 0.85, 0.85, 0.85 | 0.85 | dropped |
| 1 | 4 | 0.95 | 0.85, 0.85, 0.95 | 0.85 | dropped |
| 2 | 3 | 0.70 | 0.85, 0.95, 0.70 | **0.85** | kept |
| 3 | 2 | 0.80 | 0.95, 0.70, 0.80 | **0.70** | kept |
| 4 | 1 | 0.60 | 0.70, 0.80, 0.60 | **0.70** | kept |
| 5 | 0 | 0.90 | 0.80, 0.60, 0.90 | **0.60** | kept |

The first b outputs are dropped because the register still holds copies of
the starting value. From s = 2 every row is a real sample, and at origin k
rows 0 and 1 are $x_{k+2}$ and $x_{k+1}$: the window.

**Parallel.** Read each window straight from the trace (all at once, one
`unfold`) and reduce it once.

| k | window | exact (min) | smooth, β = 10: $-\tfrac1\beta\log\sum_j e^{-\beta x_j}$ |
|---|---|---|---|
| 0 | 0.60, 0.80 | **0.60** | $-\tfrac1{10}\log(e^{-6.0}+e^{-8.0})$ = **0.5873** |
| 1 | 0.80, 0.70 | **0.70** | $-\tfrac1{10}\log(e^{-8.0}+e^{-7.0})$ = **0.6687** |
| 2 | 0.70, 0.95 | **0.70** | $-\tfrac1{10}\log(e^{-7.0}+e^{-9.5})$ = **0.6921** |
| 3 | 0.95, 0.85 | **0.85** | $-\tfrac1{10}\log(e^{-9.5}+e^{-8.5})$ = **0.8187** |

The recurrent version reduces the same two register rows with the same
function, so its kept outputs are identical: exact (0.60, 0.70, 0.70, 0.85),
smooth (0.5873, 0.6687, 0.6921, 0.8187). The smooth lower is below the exact
min by at most δ₂ = ln 2 / β = 0.069 (Eq. (16)).

**Why they agree.** At origin k the register holds $x_{k+b}, \dots, x_{k+a}$
in rows 0 … b−a, and the parallel window holds the same values. Min and
log-sum-exp do not depend on the order of their inputs, so both versions give
the same value up to float rounding (largest difference in the benchmark
1.2e-7). This holds for bounded windows and for Until. For windows to the
end of the plan the exact values also agree; the smooth values differ on
purpose, because the old recurrent fold counted the last value more than
once and was not Eq. (15).

## Results

**Benchmark** (one smooth forward + backward pass, φ₁-shaped formula, batch 1;
CPU = one core of an i7-12700K, GPU = RTX 3090):

| Trace length T | Recurrent, CPU | Parallel, CPU | Recurrent, GPU | Parallel, GPU |
|---|---|---|---|---|
| 31 (lane horizon) | 16.9 ms | 2.1 ms (8×) | 36 ms | 4.2 ms (9×) |
| 121 | 67 ms | 2.3 ms (29×) | 133 ms | 4.2 ms (32×) |
| 481 | 366 ms | 4.3 ms (86×) | 772 ms | 4.3 ms (179×) |

Batching candidate plans (T = 31): the GPU scores 4096 plans in 4.8 ms
(≈850,000 plans/s); one plan takes 4.2 ms.

**Lane-merge closed loop** (20 seeds per scene, one CPU core, replan time):

| Scene | Recurrent median / p95 | Parallel median / p95 / max |
|---|---|---|
| Merge | 467 / 953 ms | 169 / 183 / 250 ms |
| Merge + headway | 671 / 1020 ms | 182 / 193 / 248 ms |
| Merge or yield, dense | 692 / 819 ms | 174 / 191 / 251 ms |
| Merge or yield, gap | 710 / 1134 ms | 173 / 189 / 209 ms |

The recurrent row is the earlier batch (all CPU cores, not pinned); the
parallel row is pinned to one core and also includes the supporting changes
above. Median and 95th percentile are now inside the 0.2 s control step; 0.3–3% of
steps exceed it. Outcomes: merge 20/20, headway 11/20 (9 violations of the
22 m gap, by ≤ 0.19 m, no collisions), dense 20/20 yielded, gap 20/20.

## Decisions for the paper

1. **Framing.** The draft describes the evaluation as "recurrent,
   LSTM-style" (abstract, Fig. 1, Fig. 2, Sec. III). Options: (a) keep the
   register (Eqs. (11)–(13), Fig. 3) as the definition and add that all
   windows are evaluated in parallel, citing STLCG++; (b) describe it as
   sliding-window reductions throughout. Either way, Fig. 2's "LSTM unit"
   (cell and hidden states) does not match pdSTL, old or new.
2. **Efficiency claim.** Include the benchmark figure, or only the
   closed-loop timing in one sentence? The parallel idea is from STLCG++, so
   it supports the paper rather than being a contribution.
3. **Windows to the end of the plan.** Define `[a, end]` in the grammar
   (one sentence) so the lane formulas fit the formal definitions.

## Note

`opt-planning` and this branch also differ in ways that predate this change
(e.g. the smoothing parameter, dropping incomplete windows, until's prefix
and conjunction). The parallel version was checked and timed against the
recurrent version on this branch (`pdstl/recurrent.py`), which has the same
register structure as `opt-planning`.

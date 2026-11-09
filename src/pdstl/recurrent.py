"""The recurrent (register) evaluation of always and eventually.

This is the step-by-step form of the draft's Eqs. (11)-(13): a shift register
of b + 1 interval pairs runs over the trace in reverse time, and each step
reduces the window rows. `pdstl.operators` evaluates the same windows for all
origins at once; this module keeps the recurrent form as the reference the
tests compare against and as the baseline of the timing benchmark.

For bounded windows [a, b], and for every window in exact mode, the outputs
equal `Always` / `Eventually`. For [a, END] in smooth mode the recursion folds
pairs from the trace end, which is sound but is not Eq. (15).
"""

import numpy as np
import torch

from pdstl.operators import (
    END,
    Maxish,
    Minish,
    STL_Formula,
    _checked_interval,
    _reduce_bounds,
)


class RecurrentTemporal(STL_Formula):
    """Window reduction by the register recursion (12), one step at a time.

    Args:
        subformula: Child formula.
        interval: Window [a, b], [a, END] or None (the rest of the trace).
        operation: Minish() (always) or Maxish() (eventually).
    """

    symbol = "rec"

    def __init__(self, subformula, interval=None, operation=None):
        super().__init__()
        self.operation = operation
        self.subformula = subformula
        self.interval = _checked_interval(interval)
        self._interval = [0, END] if self.interval is None else self.interval

        a, b = self._interval
        self._unbounded = b == END
        # [0, inf) is the same window as no interval: the rest of the trace.
        self._suffix = interval is None or (self._unbounded and int(a) == 0)

        if self._suffix:
            self.rnn_dim = 1
        elif self._unbounded:
            self.rnn_dim = int(a)
        else:
            self.rnn_dim = int(b) + 1

        # Shift register: M drops the oldest slot, b writes the newest one.
        self.register_buffer(
            "M", torch.tensor(np.diag(np.ones(self.rnn_dim - 1), k=1)).float()
        )
        b_vec = torch.zeros(self.rnn_dim, 1)
        b_vec[-1] = 1.0
        self.register_buffer("b", b_vec)

    @property
    def lookahead(self):
        """Future steps one evaluation needs (0 for a suffix window)."""
        a, b = self._interval
        if self._suffix:
            return 0
        return int(a) if self._unbounded else int(b)

    def _initialize_rnn_cell(self, x):
        """Register pre-filled with the first reversed value; those outputs are dropped."""
        h0 = (
            x[:, :1, :].expand(-1, self.rnn_dim, -1).clone()
        )  # [B, rnn_dim, 2]
        if self._unbounded and not self._suffix:  # [a, inf), a > 0
            return ((x[:, :1, :], h0), 0.0)
        return (h0, 0.0)

    def _apply_shift(self, h0, x):
        """M @ h0 + b * x, applied to each interval endpoint."""
        batch, rnn_dim, bounds = h0.shape
        M = self.M.to(dtype=h0.dtype, device=h0.device)
        b = self.b.to(dtype=h0.dtype, device=h0.device)
        h0_flat = h0.permute(0, 2, 1).reshape(-1, rnn_dim)  # [B*2, rnn_dim]
        shifted = (
            torch.matmul(h0_flat, M.t())
            .reshape(batch, bounds, rnn_dim)
            .permute(0, 2, 1)
        )
        return shifted + b.view(1, -1, 1) * x.squeeze(1).unsqueeze(1)

    def _rnn_cell(self, x, hc, beta=None):
        h0, _ = hc
        if self._suffix:  # [t, end of trace]
            output = _reduce_bounds(
                self.operation, torch.cat([h0, x], dim=1), beta
            )
            state = (output, None)
        elif self._unbounded:  # [a, inf), a > 0
            d0, h0 = h0
            dh = torch.cat([d0, h0[:, :1, :]], dim=1)
            output = _reduce_bounds(self.operation, dh, beta)
            state = ((output, self._apply_shift(h0, x)), None)
        else:  # [a, b]: slot rnn_dim-1-k holds the value k steps ahead
            a, b = int(self._interval[0]), int(self._interval[1])
            new_h0 = self._apply_shift(h0, x)
            output = _reduce_bounds(
                self.operation, new_h0[:, : b - a + 1, :], beta
            )
            state = (new_h0, None)
        return output, state

    def _run_cell(self, x, beta):
        outputs, hc = [], self._initialize_rnn_cell(x)
        for x_i in torch.split(x, 1, dim=1):
            o, hc = self._rnn_cell(x_i, hc, beta)
            outputs.append(o)
        return torch.cat(outputs, dim=1)

    def robustness_trace(
        self, belief_trajectory, beta=None, keepdim=True, **kwargs
    ):
        trace = self.subformula(
            belief_trajectory, beta=beta, keepdim=keepdim, **kwargs
        )
        if trace.shape[1] - self.lookahead <= 0:
            raise ValueError(
                f"{self}: one evaluation origin needs {self.lookahead + 1} steps, "
                f"child trace has {trace.shape[1]}"
            )
        # Run backwards so each origin sees its future; drop incomplete windows.
        output_reversed = self._run_cell(
            torch.flip(trace, dims=[1]), beta=beta
        )
        return torch.flip(output_reversed[:, self.lookahead :], dims=[1])

    def __str__(self):
        if self.interval is None:
            return f"{self.symbol}({self.subformula})"
        return f"{self.symbol}_{self._interval}({self.subformula})"


class RecurrentAlways(RecurrentTemporal):
    """G[a, b] by the register recursion."""

    def __init__(self, subformula, interval=None):
        super().__init__(subformula, interval, Minish())


class RecurrentEventually(RecurrentTemporal):
    """F[a, b] by the register recursion."""

    def __init__(self, subformula, interval=None):
        super().__init__(subformula, interval, Maxish())

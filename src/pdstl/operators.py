import torch
import numpy as np


class STL_Formula(torch.nn.Module):
    """
    Base class for Probabilistic STL formulas.
    """

    def __init__(self):
        super(STL_Formula, self).__init__()

    def robustness_trace(self, belief_trajectory, scale=-1, keepdim=True, **kwargs):
        """
        Compute robustness trace for belief trajectory.

        Args:
           belief_trajectory: BeliefTrajectory object
           scale: smoothing parameter (scale > 0 for smooth, <= 0 for exact)
           keepdim: keep dimensions

        Returns:
           [B,T,2] probability bounds on robustness trace where
           [..., 0] is lower bound and [..., 1] is upper bound
        """
        raise NotImplementedError("robustness_trace not yet implemented")

    def forward(self, belief_trajectory, **kwargs):
        """Forward pass delegates to robustness_trace"""
        return self.robustness_trace(belief_trajectory, **kwargs)

    def __str__(self):
        raise NotImplementedError("__str__ not yet implemented")

    def __and__(self, other):
        """Overload & operator for And"""
        return And(self, other)

    def __or__(self, other):
        """Overload | operator for Or"""
        return Or(self, other)

    def __invert__(self):
        """Overload ~ operator for Negation"""
        return Negation(self)


class Minish(torch.nn.Module):
    """Compute minimum (exact or smooth) over specified dimension"""

    def forward(self, x, scale, dim=1, keepdim=True):
        """
        The bounds dimension [..., 2] is automatically processed element-wise.
        """
        if scale > 0:
            return -torch.logsumexp(-x * scale, dim=dim, keepdim=keepdim) / scale
        else:
            return x.min(dim=dim, keepdim=keepdim)[0]


class Maxish(torch.nn.Module):
    """Compute maximum (exact or smooth) over specified dimension"""

    def forward(self, x, scale, dim=1, keepdim=True):
        """
        The bounds dimension [..., 2] is automatically processed element-wise.
        """
        if scale > 0:
            return torch.logsumexp(x * scale, dim=dim, keepdim=keepdim) / scale
        else:
            return x.max(dim=dim, keepdim=keepdim)[0]


class GreaterThan(STL_Formula):
    """
    Predicate: x >= threshold
    Returns conservative probability intervals [lower, upper].

    Lower bound: Pessimistic
    Upper bound: Optimistic
    """

    def __init__(self, threshold):
        super(GreaterThan, self).__init__()
        self.threshold = threshold

    def robustness_trace(self, belief_trajectory, **kwargs):
        probs_lower = []
        probs_upper = []

        for t in range(len(belief_trajectory)):
            belief = belief_trajectory[t]

            # LOWER BOUND:

            lower_bound = belief.lower_bound()  # μ - k*σ
            residual_lower = lower_bound - self.threshold
            prob_lower = belief.probability_of(residual_lower)

            # UPPER BOUND: Optimistic assumption

            upper_bound = belief.upper_bound() 
            residual_upper = upper_bound - self.threshold
            prob_upper = belief.probability_of(residual_upper)

            probs_lower.append(prob_lower)
            probs_upper.append(prob_upper)

        # Stack along time dimension
        lower_tensor = torch.cat(probs_lower, dim=1)  # [B, T, D]
        upper_tensor = torch.cat(probs_upper, dim=1)  # [B, T, D]

        # Remove the D dimension (assuming D=1 for scalar signals)
        if lower_tensor.shape[2] == 1:
            lower_tensor = lower_tensor.squeeze(2)  # [B, T]
            upper_tensor = upper_tensor.squeeze(2)  # [B, T]

        # Return as [lower, upper] bounds
        return torch.stack([lower_tensor, upper_tensor], dim=-1)  # [B, T, 2]

    def __str__(self):
        return f"x >= {self.threshold}"


class LessThan(STL_Formula):
    """
    Predicate: x <= threshold
    Returns conservative probability intervals [lower, upper].
    """

    def __init__(self, threshold):
        super(LessThan, self).__init__()
        self.threshold = threshold

    def robustness_trace(self, belief_trajectory, **kwargs):
        probs_lower = []
        probs_upper = []

        for t in range(len(belief_trajectory)):
            belief = belief_trajectory[t]

            # LOWER BOUND: Pessimistic assumption
            upper_bound = belief.upper_bound()  # μ + k*σ
            residual_lower = self.threshold - upper_bound
            prob_lower = belief.probability_of(residual_lower)

            # UPPER BOUND: Optimistic assumption
            lower_bound = belief.lower_bound()  # μ - k*σ
            residual_upper = self.threshold - lower_bound
            prob_upper = belief.probability_of(residual_upper)

            probs_lower.append(prob_lower)
            probs_upper.append(prob_upper)

        # Stack along time dimension
        lower_tensor = torch.cat(probs_lower, dim=1)  # [B, T, D]
        upper_tensor = torch.cat(probs_upper, dim=1)  # [B, T, D]

        if lower_tensor.shape[2] == 1:
            lower_tensor = lower_tensor.squeeze(2)  # [B, T]
            upper_tensor = upper_tensor.squeeze(2)  # [B, T]

        return torch.stack([lower_tensor, upper_tensor], dim=-1)  # [B, T, 2]

    def __str__(self):
        return f"x <= {self.threshold}"


class Negation(STL_Formula):
    """
    Negation: ¬ϕ
    For StoRI: Swaps and complements bounds
    """

    def __init__(self, subformula):
        super(Negation, self).__init__()
        self.subformula = subformula

    def robustness_trace(self, belief_trajectory, scale=-1, keepdim=True, **kwargs):
        trace = self.subformula(
            belief_trajectory, scale=scale, keepdim=keepdim, **kwargs
        )
        # trace: [B,T,2]
        # [lower, upper] -> [1 - upper, 1 - lower]
        lower = 1.0 - trace[..., 1]
        upper = 1.0 - trace[..., 0]
        return torch.stack([lower, upper], dim=-1)

    def __str__(self):
        return f"¬({self.subformula})"


class And(STL_Formula):
    """
    Conjunction: ϕ₁ ∧ ϕ₂
    Uses Frechet bounds element-wise:
      lower = max(l1 + l2 - 1, 0)
      upper = min(u1, u2)
    """

    def __init__(self, subformula1, subformula2):
        super(And, self).__init__()
        self.subformula1 = subformula1
        self.subformula2 = subformula2

    def robustness_trace(self, belief_trajectory, scale=-1, keepdim=True, **kwargs):
        trace1 = self.subformula1(
            belief_trajectory, scale=scale, keepdim=keepdim, **kwargs
        )
        trace2 = self.subformula2(
            belief_trajectory, scale=scale, keepdim=keepdim, **kwargs
        )
        # Both: [B,T,2]
        l1, u1 = trace1[..., 0:1], trace1[..., 1:2]
        l2, u2 = trace2[..., 0:1], trace2[..., 1:2]

        # Product lower bound: P(A ∩ B) ≥ P(A)·P(B) under independence.
        # Valid when sub-formulas constrain independent spatial regions (goal vs obstacles).
        # Tighter than Fréchet (max(l1+l2-1,0)) and gives better gradients for optimisation.
        lower = l1 * l2
        upper = torch.minimum(u1, u2)

        return torch.cat([lower, upper], dim=-1)

    def __str__(self):
        return f"({self.subformula1}) ∧ ({self.subformula2})"


class Or(STL_Formula):
    """
    Disjunction: ϕ₁ ∨ ϕ₂
    Uses Frechet bounds element-wise:
      lower = max(l1, l2)
      upper = min(u1 + u2, 1)
    """

    def __init__(self, subformula1, subformula2):
        super(Or, self).__init__()
        self.subformula1 = subformula1
        self.subformula2 = subformula2

    def robustness_trace(self, belief_trajectory, scale=-1, keepdim=True, **kwargs):
        trace1 = self.subformula1(
            belief_trajectory, scale=scale, keepdim=keepdim, **kwargs
        )
        trace2 = self.subformula2(
            belief_trajectory, scale=scale, keepdim=keepdim, **kwargs
        )
        # Both: [B,T,2]
        l1, u1 = trace1[..., 0:1], trace1[..., 1:2]
        l2, u2 = trace2[..., 0:1], trace2[..., 1:2]

        lower = torch.maximum(l1, l2)
        upper = torch.minimum(u1 + u2, torch.ones_like(u1))

        return torch.cat([lower, upper], dim=-1)

    def __str__(self):
        return f"({self.subformula1}) ∨ ({self.subformula2})"


class Implies(STL_Formula):
    """
    Implication: ϕ₁ ⇒ ϕ₂
    Defined as: ¬ϕ₁ ∨ ϕ₂
    """

    def __init__(self, subformula1, subformula2):
        super(Implies, self).__init__()
        self.subformula1 = subformula1
        self.subformula2 = subformula2
        self.equivalent = Or(Negation(subformula1), subformula2)

    def robustness_trace(self, belief_trajectory, scale=-1, keepdim=True, **kwargs):
        return self.equivalent(
            belief_trajectory, scale=scale, keepdim=keepdim, **kwargs
        )

    def __str__(self):
        return f"({self.subformula1}) ⇒ ({self.subformula2})"


class Temporal_Operator(STL_Formula):
    """
    Base class for temporal operators.
    """

    def __init__(self, subformula, interval=None):
        super(Temporal_Operator, self).__init__()
        self.subformula = subformula
        self.interval = interval
        self._interval = [0, np.inf] if self.interval is None else self.interval

        # RNN memory length in time
        if not self.interval:
            self.rnn_dim = 1
        else:
            # approximate memory length; for bounded interval use window length
            a, b = self._interval
            if np.isinf(b):
                self.rnn_dim = int(max(1, a))
            else:
                self.rnn_dim = int(b + 1)

        # Operation set by subclass (Minish or Maxish)
        self.operation = None

        # Shift matrices (for sliding window)
        self.M = (
            torch.tensor(np.diag(np.ones(self.rnn_dim - 1), k=1))
            .requires_grad_(False)
            .float()
        )
        self.b = torch.zeros(self.rnn_dim).unsqueeze(-1).requires_grad_(False).float()
        self.b[-1] = 1.0

    def _initialize_rnn_cell(self, x):
        """
        Initialize hidden state.
        x: [B,T,2]
        Returns: h0 with same shape structure
        """
        if x.is_cuda:
            self.M = self.M.cuda()
            self.b = self.b.cuda()

        # Padding with first value - automatically handles bounds dimension
        h0 = x[:, :1, :].expand(-1, self.rnn_dim, -1).clone()  # [B,rnn_dim,2]
        count = 0.0

        # Special case for [a, inf)
        if (self._interval[1] == np.inf) and (self._interval[0] > 0):
            d0 = x[:, :1, :]
            return ((d0, h0), count)

        return (h0, count)

    def _apply_shift(self, h0, x):
        """
        Apply M @ h0 + b * x
        h0: [B,rnn_dim,2]
        x: [B,1,2]
        """
        batch, rnn_dim, bounds = h0.shape

        # Treat (batch, bounds) as batch dimension for matmul
        h0_reshaped = h0.permute(0, 2, 1)  # [B,2,rnn_dim]

        h0_flat = h0_reshaped.reshape(-1, rnn_dim)  # [B*2,rnn_dim]

        # Shift
        shifted_flat = torch.matmul(h0_flat, self.M.t())  # [B*2,rnn_dim]
        shifted = shifted_flat.reshape(batch, bounds, rnn_dim)
        shifted = shifted.permute(0, 2, 1)  # [B,rnn_dim,2]

        # Add new value into last position
        b_broadcast = self.b.view(1, -1, 1)  # [1,rnn_dim,1]
        x_broadcast = x.squeeze(1).unsqueeze(1)  # [B,1,2]

        return shifted + b_broadcast * x_broadcast

    def _rnn_cell(self, x, hc, scale=-1, **kwargs):
        """Must be implemented by subclass"""
        raise NotImplementedError

    def _run_cell(self, x, scale):
        """Run RNN through entire trace"""
        outputs = []
        hc = self._initialize_rnn_cell(x)
        xs = torch.split(x, 1, dim=1)  # list of [B,1,2]

        for xs_i in xs:
            o, hc = self._rnn_cell(xs_i, hc, scale)
            outputs.append(o)

        return torch.cat(outputs, dim=1)  # [B,T,2]

    def robustness_trace(self, belief_trajectory, scale=-1, keepdim=True, **kwargs):
        trace = self.subformula(
            belief_trajectory, scale=scale, keepdim=keepdim, **kwargs
        )
        # FORWARD-LOOKING
        trace_reversed = torch.flip(trace, dims=[1])
        output_reversed = self._run_cell(trace_reversed, scale=scale)
        return torch.flip(output_reversed, dims=[1])


class Always(Temporal_Operator):
    """
    □_I ϕ: Always operator
    Computes min over time interval.
    """

    def __init__(self, subformula, interval=None):
        super(Always, self).__init__(subformula=subformula, interval=interval)
        self.operation = Minish()
        self.oper = "min"

    def _rnn_cell(self, x, hc, scale=-1, **kwargs):
        """
        Compute running minimum.
        """
        h0, c = hc

        if self.operation is None:
            raise Exception("Operation not initialized")

        # CASE 1: Global Always (no interval)
        if self.interval is None:
            # h0: [B,rnn_dim,2], x: [B,1,2]
            input_ = torch.cat([h0, x], dim=1)  # [B,rnn_dim+1,2]
            output = self.operation(input_, scale, dim=1, keepdim=True)  # [B,1,2]
            state = (output, None)

        # CASE 2: Unbounded future [a, inf)
        elif (self._interval[1] == np.inf) and (self._interval[0] > 0):
            d0, h0 = h0  # unpack tuple state
            dh = torch.cat([d0, h0[:, :1, :]], dim=1)  # [B,2,2]
            output = self.operation(dh, scale, dim=1, keepdim=True)
            new_h0 = self._apply_shift(h0, x)
            state = ((output, new_h0), None)

        # CASE 3: Bounded interval [a,b]
        else:
            a, b = int(self._interval[0]), int(self._interval[1])
            new_h0 = self._apply_shift(h0, x)
            window = new_h0[:, : b - a + 1, :]
            output = self.operation(window, scale, dim=1, keepdim=True)
            state = (new_h0, None)

        return output, state

    def __str__(self):
        if self.interval is None:
            return f"□({self.subformula})"
        return f"□_{self._interval}({self.subformula})"


class Eventually(Temporal_Operator):
    """
    Eventually operator: ♢_I ϕ
    Computes max over time interval.
    The bounds dimension is processed automatically.
    """

    def __init__(self, subformula, interval=None):
        super(Eventually, self).__init__(subformula=subformula, interval=interval)
        self.operation = Maxish()
        self.oper = "max"

    def _rnn_cell(self, x, hc, scale=-1, **kwargs):
        """
        Compute running maximum.
        """
        h0, c = hc

        if self.operation is None:
            raise Exception("Operation not initialized")

        # Case 1: Global Eventually
        if self.interval is None:
            input_ = torch.cat([h0, x], dim=1)
            output = self.operation(input_, scale, dim=1, keepdim=True)
            state = (output, None)

        # Case 2: Unbounded future [a, inf)
        elif (self._interval[1] == np.inf) and (self._interval[0] > 0):
            d0, h0 = h0
            dh = torch.cat([d0, h0[:, :1, :]], dim=1)
            output = self.operation(dh, scale, dim=1, keepdim=True)
            new_h0 = self._apply_shift(h0, x)
            state = ((output, new_h0), None)

        # Case 3: Bounded interval [a,b]
        else:
            a, b = int(self._interval[0]), int(self._interval[1])
            new_h0 = self._apply_shift(h0, x)
            window = new_h0[:, : b - a + 1, :]
            output = self.operation(window, scale, dim=1, keepdim=True)
            state = (new_h0, None)

        return output, state

    def __str__(self):
        if self.interval is None:
            return f"♢({self.subformula})"
        return f"♢_{self._interval}({self.subformula})"


class Until(STL_Formula):
    """
    ϕ U_I ψ : Until operator
    """

    def __init__(self, left, right, interval=None):
        super(Until, self).__init__()
        self.left = left
        self.right = right
        self.interval = [0, np.inf] if interval is None else interval
        self._interval = self.interval

        self.min_op = Minish()
        self.max_op = Maxish()

    def robustness_trace(self, belief_trajectory, scale=-1, keepdim=True, **kwargs):
        # ϕ and ψ traces: [B,T,2]
        phi = self.left(belief_trajectory, scale=scale, keepdim=True, **kwargs)
        psi = self.right(belief_trajectory, scale=scale, keepdim=True, **kwargs)

        B, T, _ = phi.shape
        a, b = self._interval
        a = int(a)
        if np.isinf(b):
            b = T - 1
        else:
            b = int(b)

        device = phi.device
        dtype = phi.dtype

        results = []  # list of [B,2], one per t

        for t in range(T):
            start = t + a
            end = min(t + b, T - 1)

            # If the interval is empty for this t, probability of satisfaction is 0
            if start > end:
                results.append(torch.zeros(B, 2, device=device, dtype=dtype))
                continue

            tau_vals = []  # candidate values for each τ

            for tau in range(start, end + 1):
                # min_{k ∈ [t, τ)} ϕ(k)
                if tau == t:
                    # Empty prefix: ϕ vacuously holds with probability 1
                    min_phi = torch.ones_like(phi[:, 0, :])  # [B,2]
                else:
                    phi_segment = phi[:, t:tau, :]  # [B,τ-t,2]
                    # min over time dim=1
                    min_phi = self.min_op(
                        phi_segment, scale, dim=1, keepdim=False
                    )  # [B,2]

                # ψ(τ)
                psi_tau = psi[:, tau, :]  # [B,2]

                # min(ψ(τ), min_prefix_ϕ)
                pair = torch.stack([min_phi, psi_tau], dim=1)  # [B,2,2]
                val_tau = self.min_op(pair, scale, dim=1, keepdim=False)  # [B,2]

                tau_vals.append(val_tau.unsqueeze(1))  # [B,1,2]

            # max over τ ∈ [t+a, t+b]
            tau_tensor = torch.cat(tau_vals, dim=1)  # [B,τ_count,2]
            best = self.max_op(tau_tensor, scale, dim=1, keepdim=False)  # [B,2]

            results.append(best)

        out = torch.stack(results, dim=1)  # [B,T,2]
        return out

    def __str__(self):
        return f"({self.left}) U_{self._interval} ({self.right})"

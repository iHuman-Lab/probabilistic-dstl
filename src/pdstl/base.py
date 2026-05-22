from abc import ABC, abstractmethod


class Belief(ABC):
    """
    Abstract base class for any belief representation.
    Users must implement:
        - value() -> representative range of state (tensor)
        - prob_from_residual(residual) -> probability (tensor)
    """

    @abstractmethod
    def value(self):
        """
        Return a representative range of state x(t)
        """
        raise NotImplementedError

    @abstractmethod
    def probability_of(self, residual):
        """
        Return P(residual >= 0)
        """
        raise NotImplementedError


class BeliefTrajectory:
    """
    Offline belief trajectory with list of beliefs
    """

    def __init__(self, beliefs):
        self.beliefs = beliefs

    def __getitem__(self, t):
        return self.beliefs[t]

    def __len__(self):
        return len(self.beliefs)

    def suffix(self, t):
        return BeliefTrajectory(self.beliefs[t:])


class OnlineBeliefTrajectory:
    """
    Online/streaming belief trajectory
    Allows incremental updates as new beliefs arrive
    """

    def __init__(self):
        self.beliefs = []

    def append(self, belief):
        self.beliefs.append(belief)

    def __getitem__(self, t):
        return self.beliefs[t]

    def suffix(self, t):
        return OnlineBeliefTrajectory.from_list(self.beliefs[t:])

    @classmethod
    def from_list(cls, lst):
        obj = cls()
        obj.beliefs = lst
        return obj

    def __len__(self):
        return len(self.beliefs)

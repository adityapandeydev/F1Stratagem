"""The forecast object every model produces, built from joint simulation draws.

Because every quantity comes from the same draws, the forecast is consistent by
construction: each entrant's position distribution sums to 1, each position is
filled exactly once per draw, outcome probabilities sum to 1 and win <= podium
<= top 10. `check()` asserts these properties.
"""

from dataclasses import dataclass, field

import numpy as np

from .rules import points_table
from .targets import OUTCOMES

INTERVAL = (0.1, 0.9)  # central 80% interval


@dataclass
class Forecast:
    event_key: str
    session_type: str
    cutoff_label: str
    cutoff_time: str
    driver_numbers: list
    driver_ids: list
    position_probs: np.ndarray            # entrants x positions
    outcome_probs: np.ndarray | None      # entrants x OUTCOMES (races and sprints)
    expected_points: np.ndarray | None
    model: str
    draws: int
    seed: int
    metadata: dict = field(default_factory=dict)

    @classmethod
    def from_draws(cls, *, positions, outcomes=None, season, session_type, **fields):
        """`positions`: draws x entrants array of finishing positions (1-based).
        `outcomes`: draws x entrants array of indices into OUTCOMES, or None."""
        draws, entrants = positions.shape
        position_probs = np.zeros((entrants, entrants))
        for column in range(entrants):
            position_probs[column] = np.bincount(positions[:, column] - 1, minlength=entrants) / draws
        outcome_probs = expected_points = None
        if outcomes is not None:
            outcome_probs = np.stack([(outcomes == k).mean(axis=0) for k in range(len(OUTCOMES))], axis=1)
            table = np.zeros(entrants + 1)
            scoring = points_table(season, session_type)
            table[1:len(scoring) + 1] = scoring[:entrants]
            classified = outcomes <= OUTCOMES.index("outside_points")
            expected_points = (table[positions] * classified).mean(axis=0)
        forecast = cls(position_probs=position_probs, outcome_probs=outcome_probs,
                       expected_points=expected_points, draws=draws, session_type=session_type, **fields)
        forecast.check()
        return forecast

    # Derived quantities -------------------------------------------------------------
    def top(self, k):
        return self.position_probs[:, :k].sum(axis=1)

    @property
    def expected_position(self):
        return self.position_probs @ np.arange(1, self.position_probs.shape[1] + 1)

    def interval(self, low=INTERVAL[0], high=INTERVAL[1]):
        """Central interval of positions per entrant, e.g. (3, 7)."""
        cdf = np.cumsum(self.position_probs, axis=1)
        lower = (cdf >= low - 1e-12).argmax(axis=1) + 1
        upper = (cdf >= high - 1e-12).argmax(axis=1) + 1
        return lower, upper

    def check(self, tolerance=1e-9):
        p = self.position_probs
        if not (np.all(p >= 0) and np.all(np.isfinite(p))):
            raise ValueError("position probabilities must be finite and non-negative")
        if not np.allclose(p.sum(axis=1), 1, atol=tolerance):
            raise ValueError("each entrant's position distribution must sum to 1")
        if not np.allclose(p.sum(axis=0), 1, atol=tolerance):
            raise ValueError("each position must be filled exactly once per draw")
        if np.any(self.top(1) > self.top(3) + tolerance) or np.any(self.top(3) > self.top(10) + tolerance):
            raise ValueError("win <= podium <= top 10 violated")
        if self.outcome_probs is not None:
            if not np.allclose(self.outcome_probs.sum(axis=1), 1, atol=tolerance):
                raise ValueError("outcome probabilities must sum to 1")
            win = self.outcome_probs[:, OUTCOMES.index("win")]
            if np.any(win > self.top(1) + tolerance):
                raise ValueError("win probability exceeds P1 probability")

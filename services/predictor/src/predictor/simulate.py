"""Ranking simulation under the Plackett-Luce model.

Each entrant has a log-strength; a finishing order is drawn by picking the winner
with probability proportional to strength, then second from the rest, and so on.
Adding independent Gumbel noise to log-strengths and sorting gives exact
Plackett-Luce draws.
"""

import numpy as np


def sample_orders(log_strength, draws, rng):
    """Draws x entrants array of positions (1 = first)."""
    log_strength = np.asarray(log_strength, dtype=float)
    keys = log_strength + rng.gumbel(size=(draws, log_strength.size))
    order = np.argsort(-keys, axis=1)                     # entrant index per position
    positions = np.empty_like(order)
    rows = np.arange(draws)[:, None]
    positions[rows, order] = np.arange(1, log_strength.size + 1)
    return positions


def log_likelihood(log_strength, positions):
    """Plackett-Luce log-likelihood of one observed order.

    `positions[i]` is entrant i's finishing position (1-based, all distinct)."""
    log_strength = np.asarray(log_strength, dtype=float)
    order = np.argsort(positions)
    ordered = log_strength[order]
    # log-sum-exp of the remaining entrants at each step, from the back
    tail = np.logaddexp.accumulate(ordered[::-1])[::-1]
    return float(np.sum(ordered - tail))

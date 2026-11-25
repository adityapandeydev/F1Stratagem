import numpy as np
from types import SimpleNamespace

from predictor.baselines import RetirementModel
from predictor.retirements import fit_grid_beta, grid_factor


def test_grid_factor_keeps_the_field_rate():
    f = grid_factor(np.arange(1, 21), 1.2)
    assert f.shape == (1, 20) and np.isclose(f.mean(), 1.0) and f[0, 0] < f[0, -1]
    draws = grid_factor(np.array([np.arange(1, 21), np.arange(20, 0, -1)]), 1.2)
    assert np.allclose(draws.mean(axis=1), 1.0) and np.isclose(draws[1, 0], f[0, -1])
    assert np.allclose(grid_factor(np.arange(1, 21), 0.0), 1.0)


def test_fit_recovers_the_direction_of_a_gradient():
    rng = np.random.default_rng(0)
    races = []
    for _ in range(300):
        slot = rng.permutation(20) + 1
        p = 0.15 * grid_factor(slot, 1.0)[0]
        races.append((np.full(20, 0.15), slot[None, :], rng.random(20) < p))
    beta = fit_grid_beta(races)
    assert 0.6 < beta < 1.4
    flat = [(b, s, rng.random(20) < 0.15) for b, s, _ in races]
    assert abs(fit_grid_beta(flat)) < 0.4
    assert fit_grid_beta([]) == 0.0


def test_zero_effect_reproduces_the_draws_without_a_grid():
    model = RetirementModel()
    model.retire_rate = 0.2
    example = SimpleNamespace(season=2022, session_type="R", driver_numbers=list(range(20)))
    positions = np.tile(np.arange(1, 21), (500, 1))
    a = model.apply(positions, example, np.random.default_rng(3))
    b = model.apply(positions, example, np.random.default_rng(3), grid=np.arange(1, 21))
    assert all(np.array_equal(x, y) for x, y in zip(a, b))
    model.grid_beta = 1.5
    c = model.apply(positions, example, np.random.default_rng(3), grid=np.arange(1, 21))
    retired = c[1] >= 3          # outcome indices of retirements come after the finishing ones
    assert retired[:, :5].mean() < retired[:, -5:].mean()

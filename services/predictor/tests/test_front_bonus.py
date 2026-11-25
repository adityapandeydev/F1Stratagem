import numpy as np

from predictor.race import RaceModel


def _model():
    return RaceModel(pace=None, name="race:test")


def test_zero_bonus_reproduces_the_order():
    rng = np.random.default_rng(0)
    race = rng.normal(size=(400, 20))
    grid = np.arange(1, 21, dtype=float)[None, :]
    m = _model()
    a = m.positions(race, grid, 0.075, 0.6, 7)
    b = m.positions(race, grid, 0.075, 0.6, 7, front=0.0)
    c = m.order(race, grid, {"kappa": 0.075, "sigma": 0.6, "front": 0.0}, {"difficulty": 1.0}, 7)
    assert np.array_equal(a, b) and np.array_equal(a, c)


def test_bonus_moves_only_the_front_three_up():
    rng = np.random.default_rng(1)
    race = rng.normal(size=(2000, 20))
    grid = np.arange(1, 21, dtype=float)[None, :]
    m = _model()
    a = m.positions(race, grid, 0.05, 0.6, 3)
    b = m.positions(race, grid, 0.05, 0.6, 3, front=0.5)
    assert (b[:, :3].mean(axis=0) < a[:, :3].mean(axis=0)).all()
    # cars behind keep their order among themselves
    assert np.array_equal(np.argsort(a[:, 3:], axis=1), np.argsort(b[:, 3:], axis=1))

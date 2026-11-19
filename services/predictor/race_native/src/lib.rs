//! Native lap-by-lap race simulation for the race model (`predictor.race_sim`).
//!
//! `simulate` runs whole races: for every draw, the start (grid spacing plus start
//! noise), then every lap: order the cars, add each car's lap time, then front to
//! back let a car within `battle_gap` of the car ahead pass with the battle model's
//! probability, else hold it at least `min_follow` behind. At most one place gained
//! per car per lap.
//!
//! Neutralisations (optional): before each green lap a safety car, virtual safety
//! car or red flag starts with that lap's hazard. Its kind and length in laps are
//! drawn from the observed mix and lengths. A safety car or red flag closes the
//! field up to `sc_gap` seconds per car in the running order; a virtual safety car
//! keeps the gaps. Nobody passes and pace does not act while it lasts. A red flag
//! ends the race with probability `red_end`. With every hazard zero no random
//! number is drawn for them, so races are exactly those without neutralisations.
//!
//! Strategy (optional, `pit_loss` > 0): a simulated race has a planned number of
//! stops (the circuit's typical count, rounded at random once per race). With
//! `deviate` (three chances per car: one more stop when the plan is one stop, one
//! fewer and one more when it is two or more) each car may take another count,
//! never fewer than one stop; without it every car keeps the plan. A car's stops
//! fall at evenly spread laps moved by noise of its own; its tyres lose `wear`
//! seconds per lap of age, which also counts in the
//! pace difference that decides passes (fresh tyres pass more easily). A stop costs
//! `pit_loss` plus a slow-stop residual drawn from observed ones and resets the
//! tyres. When a safety car or virtual safety car comes out, cars with a stop due
//! within `window` laps stop at once for `sc_ratio` of the loss, before the field
//! closes up; so do cars whose stop comes due within `window` laps on a later lap
//! of the same neutralisation (behind a safety car the field closes up again). A
//! red flag gives everyone new tyres. Without strategy no random
//! number is drawn for it. With team wear, each team's wear deviation from the
//! field is drawn once per simulated race from its mean and spread, shared by
//! teammates, so tyre outcomes differ between simulated races.
//!
//! Random numbers are drawn here: each draw has its own PCG64 generator, its state
//! spread from the forecast's seed and the draw index, so a forecast is
//! reproducible from its seed and does not depend on the number of threads. Draws
//! run in parallel (rayon).

use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rand::{Rng, SeedableRng};
use rand_distr::StandardNormal;
use rand_pcg::Pcg64;
use rayon::prelude::*;

/// P(the car behind is ahead after the lap): logistic on [1, gap, ln gap, delta,
/// delta * gap] (gap floored at 0.01 s, delta clipped to +-3 s) plus the circuit
/// term and the shift.
#[inline]
fn probability(gap: f64, delta: f64, coef: &[f64; 6], term: f64, shift: f64) -> f64 {
    let gap = gap.max(0.01);
    let delta = delta.clamp(-3.0, 3.0);
    let z = coef[0] + coef[1] * gap + coef[2] * gap.ln() + coef[3] * delta + coef[4] * (delta * gap)
        + coef[5] * term
        + shift;
    1.0 / (1.0 + (-z).exp())
}

fn coefficients(coef: &[f64]) -> PyResult<[f64; 6]> {
    coef.try_into().map_err(|_| PyValueError::new_err("the battle model has 6 coefficients"))
}

/// One lap of one draw. `time`: each car's race time so far (updated in place);
/// `seconds`, `noise`: per car (by entrant index); `luck`: per position in the
/// running order.
#[allow(clippy::too_many_arguments)]
fn lap_of_draw(
    time: &mut [f64],
    seconds: &[f64],
    noise: &[f64],
    luck: &[f64],
    coef: &[f64; 6],
    term: f64,
    shift: f64,
    scale: f64,
    battle_gap: f64,
    min_follow: f64,
    order: &mut Vec<usize>,
    before: &mut Vec<f64>,
    after: &mut Vec<f64>,
    own: &mut Vec<f64>,
) {
    let n = time.len();
    order.clear();
    order.extend(0..n);
    order.sort_by(|&a, &b| time[a].partial_cmp(&time[b]).unwrap()); // stable, like NumPy's
    before.clear();
    after.clear();
    own.clear();
    for &car in order.iter() {
        before.push(time[car]);
        after.push(time[car] + (seconds[car] + scale * noise[car]));
        own.push(seconds[car]);
    }
    for k in 1..n {
        let gap = before[k] - before[k - 1];
        let delta = own[k] - own[k - 1];
        let chance = if gap < battle_gap { probability(gap, delta, coef, term, shift) } else { 0.0 };
        let passed = luck[k] < chance; // by position, like the NumPy version
        let (t_ahead, t_behind) = (after[k - 1], after[k]);
        let (first, second) = if passed {
            let mut first = t_ahead.min(t_behind);
            if k >= 2 {
                first = first.max(after[k - 2] + min_follow); // the passer stays behind the car two ahead
            }
            (first, t_ahead.max(first + min_follow))
        } else {
            (t_ahead, t_behind.max(t_ahead + min_follow))
        };
        after[k - 1] = first;
        after[k] = second;
        if passed {
            order.swap(k - 1, k);
            before.swap(k - 1, k);
            own.swap(k - 1, k);
        }
    }
    for (position, &car) in order.iter().enumerate() {
        time[car] = after[position];
    }
}

/// Neutralisations of one race: per-lap hazard of a new one, the kind mix
/// (safety car, virtual safety car, red flag) and observed lengths of each kind.
struct Neutral {
    hazard: Vec<f64>,
    kinds: [f64; 3],
    lengths: [Vec<usize>; 3],
    sc_gap: f64,
    red_end: f64,
}

/// Pit stops and tyre wear of one race.
struct Strategy {
    wear: f64,
    pit_loss: f64,
    sc_ratio: f64,
    stops_mean: f64,
    stint_noise: f64,
    window: usize,
    residuals: Vec<f64>,
    deviate: Vec<f64>,
    /// Team wear (optional): each car's team (`team[car]`); per team, the mean and
    /// spread of its wear deviation from the field (s per lap of age), drawn once
    /// per simulated race and shared by teammates. Empty: every car wears at `wear`.
    team: Vec<usize>,
    team_mean: Vec<f64>,
    team_sd: Vec<f64>,
}

impl Strategy {
    fn enabled(&self) -> bool {
        self.pit_loss > 0.0
    }

    /// Planned stops of one simulated race: the circuit's typical count rounded at
    /// random, shared by the field (rounding per car would split it far more often
    /// than real strategies do; cars leave the plan through `deviate`).
    fn count(&self, rng: &mut Pcg64) -> usize {
        let whole = self.stops_mean.floor();
        (whole as usize + usize::from(rng.gen::<f64>() < self.stops_mean - whole)).max(1)
    }

    /// Stops of `car` in a race planned for `count` (no random number without `deviate`).
    fn own(&self, car: usize, count: usize, rng: &mut Pcg64) -> usize {
        if self.deviate.is_empty() {
            return count;
        }
        let chance = &self.deviate[3 * car..3 * car + 3];
        let u = rng.gen::<f64>();
        if count == 1 {
            1 + usize::from(u < chance[0])
        } else if u < chance[1] {
            count - 1
        } else if u < chance[1] + chance[2] {
            count + 1
        } else {
            count
        }
    }

    /// Planned stop laps (1-based, ascending) of one car making `count` stops.
    fn plan(&self, laps: usize, count: usize, rng: &mut Pcg64) -> Vec<usize> {
        let last = laps.saturating_sub(1).max(2);
        let mut plan: Vec<usize> = (1..=count)
            .map(|j| {
                let at = laps as f64 * j as f64 / (count + 1) as f64
                    + self.stint_noise * rng.sample::<f64, _>(StandardNormal);
                (at.round().max(2.0) as usize).min(last)
            })
            .collect();
        plan.sort_unstable();
        plan
    }

    /// Each car's wear in one simulated race (no random number without team wear).
    fn car_wear(&self, n: usize, rng: &mut Pcg64) -> Vec<f64> {
        if self.team.is_empty() {
            return vec![self.wear; n];
        }
        let draws: Vec<f64> = self
            .team_mean
            .iter()
            .zip(&self.team_sd)
            .map(|(&mean, &sd)| mean + sd * rng.sample::<f64, _>(StandardNormal))
            .collect();
        self.team.iter().map(|&t| (self.wear + draws[t]).max(0.0)).collect()
    }

    fn cost(&self, rng: &mut Pcg64) -> f64 {
        let extra = if self.residuals.is_empty() {
            0.0
        } else {
            self.residuals[rng.gen_range(0..self.residuals.len())]
        };
        (self.pit_loss + extra).max(0.0)
    }
}

const SC: usize = 0;
const RED: usize = 2;

/// Settings of one simulated race, shared by all its draws.
struct Race {
    coef: [f64; 6],
    term: f64,
    shift: f64,
    laps: usize,
    start_noise: f64,
    start_spacing: f64,
    scale: f64,
    battle_gap: f64,
    min_follow: f64,
    neutral: Neutral,
    strategy: Strategy,
}

/// Closes the field up behind the leader, `gap` seconds per place, order kept.
fn close_up(time: &mut [f64], order: &mut Vec<usize>, gap: f64) {
    order.clear();
    order.extend(0..time.len());
    order.sort_by(|&a, &b| time[a].partial_cmp(&time[b]).unwrap());
    let leader = time[order[0]];
    for (place, &car) in order.iter().enumerate() {
        time[car] = leader + gap * place as f64;
    }
}

/// One draw: the race from the grid to the flag; writes finishing positions (1 =
/// first) into `positions`.
fn race_of_draw(race: &Race, seconds: &[f64], grid: &[f64], rng: &mut Pcg64, positions: &mut [i64]) {
    let time = race_times(race, seconds, grid, rng);
    let mut order: Vec<usize> = (0..time.len()).collect();
    order.sort_by(|&a, &b| time[a].partial_cmp(&time[b]).unwrap());
    for (position, &car) in order.iter().enumerate() {
        positions[car] = position as i64 + 1;
    }
}

/// Each car's race time of one draw (s against the start).
fn race_times(race: &Race, seconds: &[f64], grid: &[f64], rng: &mut Pcg64) -> Vec<f64> {
    let n = seconds.len();
    let mut time: Vec<f64> = grid
        .iter()
        .map(|&slot| race.start_spacing * (slot - 1.0) + race.start_noise * rng.sample::<f64, _>(StandardNormal))
        .collect();
    let (mut noise, mut luck) = (vec![0.0; n], vec![0.0; n]);
    let (mut order, mut before, mut after, mut own) =
        (Vec::with_capacity(n), Vec::with_capacity(n), Vec::with_capacity(n), Vec::with_capacity(n));
    let strategy = &race.strategy;
    let pits = strategy.enabled();
    let plans: Vec<Vec<usize>> = if pits {
        let count = strategy.count(rng);
        (0..n)
            .map(|car| {
                let own = strategy.own(car, count, rng);
                strategy.plan(race.laps, own, rng)
            })
            .collect()
    } else {
        vec![]
    };
    let mut next = vec![0usize; n];
    let mut age = vec![0.0f64; n];
    let wear = if pits { strategy.car_wear(n, rng) } else { vec![] };
    let mut effective = seconds.to_vec();
    let mut neutral_left = 0usize;
    let mut neutral_kind = SC;
    for lap in 0..race.laps {
        let lap1 = lap + 1;
        let mut started = false;
        if neutral_left == 0 && race.neutral.hazard[lap] > 0.0 && rng.gen::<f64>() < race.neutral.hazard[lap] {
            let pick = rng.gen::<f64>();
            let kind = if pick < race.neutral.kinds[0] {
                SC
            } else if pick < race.neutral.kinds[0] + race.neutral.kinds[1] {
                1
            } else {
                RED
            };
            neutral_kind = kind;
            started = true;
            let lengths = &race.neutral.lengths[kind];
            neutral_left = if lengths.is_empty() { 1 } else { lengths[rng.gen_range(0..lengths.len())].max(1) };
            if kind == RED && rng.gen::<f64>() < race.neutral.red_end {
                break; // the race is not restarted: classified as it stands
            }
            if pits {
                for car in 0..n {
                    let due = next[car] < plans[car].len() && plans[car][next[car]] <= lap1 + strategy.window;
                    if kind == RED {
                        age[car] = 0.0; // tyres may be changed under the red flag
                        if due {
                            next[car] += 1;
                        }
                    } else if due {
                        time[car] += strategy.sc_ratio * strategy.cost(rng);
                        age[car] = 0.0;
                        next[car] += 1;
                    }
                }
            }
            if kind != 1 {
                close_up(&mut time, &mut order, race.neutral.sc_gap);
            }
        }
        if neutral_left > 0 {
            // a stop coming due later in a long safety car or virtual safety car is
            // taken then, still at the neutralised loss; behind a safety car the car
            // rejoins the queue
            if pits && !started && neutral_kind != RED {
                let mut stopped = false;
                for car in 0..n {
                    if next[car] < plans[car].len() && plans[car][next[car]] <= lap1 + strategy.window {
                        time[car] += strategy.sc_ratio * strategy.cost(rng);
                        age[car] = 0.0;
                        next[car] += 1;
                        stopped = true;
                    }
                }
                if stopped && neutral_kind == SC {
                    close_up(&mut time, &mut order, race.neutral.sc_gap);
                }
            }
            neutral_left -= 1; // neutralised: nobody passes, pace does not act
            continue;
        }
        for value in noise.iter_mut() {
            *value = rng.sample(StandardNormal);
        }
        for value in luck.iter_mut() {
            *value = rng.gen::<f64>();
        }
        if pits {
            for car in 0..n {
                effective[car] = seconds[car] + wear[car] * age[car];
            }
        }
        lap_of_draw(
            &mut time,
            &effective,
            &noise,
            &luck,
            &race.coef,
            race.term,
            race.shift,
            race.scale,
            race.battle_gap,
            race.min_follow,
            &mut order,
            &mut before,
            &mut after,
            &mut own,
        );
        if pits {
            for car in 0..n {
                if next[car] < plans[car].len() && plans[car][next[car]] <= lap1 {
                    time[car] += strategy.cost(rng); // in and out of the pits: the field goes by
                    age[car] = 0.0;
                    next[car] += 1;
                } else {
                    age[car] += 1.0;
                }
            }
        }
    }
    time
}

/// Finishing positions (draws x entrants, 1 = first) of `laps` laps. `seconds`:
/// each draw's race pace in s per lap against the field's reference; `grid`: grid
/// slots per draw; `scale`: s of lap noise per unit normal; `seed`: the forecast's.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn simulate<'py>(
    py: Python<'py>,
    seconds: PyReadonlyArray2<f64>,
    grid: PyReadonlyArray2<f64>,
    coef: PyReadonlyArray1<f64>,
    term: f64,
    shift: f64,
    laps: usize,
    start_noise: f64,
    start_spacing: f64,
    scale: f64,
    battle_gap: f64,
    min_follow: f64,
    seed: u64,
    hazard: PyReadonlyArray1<f64>,
    kinds: PyReadonlyArray1<f64>,
    sc_lengths: PyReadonlyArray1<i64>,
    vsc_lengths: PyReadonlyArray1<i64>,
    red_lengths: PyReadonlyArray1<i64>,
    sc_gap: f64,
    red_end: f64,
    wear: f64,
    pit_loss: f64,
    sc_ratio: f64,
    stops_mean: f64,
    stint_noise: f64,
    window: usize,
    residuals: PyReadonlyArray1<f64>,
    deviate: PyReadonlyArray1<f64>,
    team: PyReadonlyArray1<i64>,
    team_mean: PyReadonlyArray1<f64>,
    team_sd: PyReadonlyArray1<f64>,
) -> PyResult<Bound<'py, PyArray2<i64>>> {
    let shape = seconds.as_array().shape().to_vec();
    if grid.as_array().shape() != shape.as_slice() {
        return Err(PyValueError::new_err("seconds and grid must have one shape"));
    }
    let hazard = hazard.as_slice()?.to_vec();
    if hazard.len() != laps {
        return Err(PyValueError::new_err("one hazard per lap"));
    }
    let kinds: [f64; 3] =
        kinds.as_slice()?.try_into().map_err(|_| PyValueError::new_err("three kind shares: SC, VSC, red"))?;
    let lengths = |a: &PyReadonlyArray1<i64>| -> PyResult<Vec<usize>> {
        Ok(a.as_slice()?.iter().map(|&x| x.max(1) as usize).collect())
    };
    let neutral = Neutral {
        hazard,
        kinds,
        lengths: [lengths(&sc_lengths)?, lengths(&vsc_lengths)?, lengths(&red_lengths)?],
        sc_gap,
        red_end,
    };
    let (draws, n) = (shape[0], shape[1]);
    let deviate = deviate.as_slice()?.to_vec();
    if !deviate.is_empty() && deviate.len() != 3 * n {
        return Err(PyValueError::new_err("three deviation chances per entrant"));
    }
    let team: Vec<usize> = team.as_slice()?.iter().map(|&t| t.max(0) as usize).collect();
    let (team_mean, team_sd) = (team_mean.as_slice()?.to_vec(), team_sd.as_slice()?.to_vec());
    if !team.is_empty()
        && (team.len() != n || team_mean.len() != team_sd.len() || team.iter().any(|&t| t >= team_mean.len()))
    {
        return Err(PyValueError::new_err("one team per entrant, a mean and spread per team"));
    }
    let race = Race {
        coef: coefficients(coef.as_slice()?)?,
        term,
        shift,
        laps,
        start_noise,
        start_spacing,
        scale,
        battle_gap,
        min_follow,
        neutral,
        strategy: Strategy {
            wear,
            pit_loss,
            sc_ratio,
            stops_mean,
            stint_noise,
            window,
            residuals: residuals.as_slice()?.to_vec(),
            deviate,
            team,
            team_mean,
            team_sd,
        },
    };
    let (seconds, grid) = (seconds.as_slice()?, grid.as_slice()?);
    let mut positions = vec![0i64; draws * n];
    py.allow_threads(|| {
        positions.par_chunks_mut(n).enumerate().for_each(|(d, out)| {
            // one generator per draw, its state spread from (seed, draw) (no correlated streams)
            let mut rng = Pcg64::seed_from_u64(seed ^ (d as u64 + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15));
            let cars = d * n..(d + 1) * n;
            race_of_draw(&race, &seconds[cars.clone()], &grid[cars], &mut rng, out);
        });
    });
    numpy::ndarray::Array2::from_shape_vec((draws, n), positions)
        .map(|a| a.into_pyarray_bound(py))
        .map_err(|e| PyValueError::new_err(e.to_string()))
}

/// The battle model's pass probability for each (gap, delta) pair.
#[pyfunction]
fn pass_probability<'py>(
    py: Python<'py>,
    gap: PyReadonlyArray1<f64>,
    delta: PyReadonlyArray1<f64>,
    coef: PyReadonlyArray1<f64>,
    term: f64,
    shift: f64,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let coef = coefficients(coef.as_slice()?)?;
    let (gap, delta) = (gap.as_slice()?, delta.as_slice()?);
    if gap.len() != delta.len() {
        return Err(PyValueError::new_err("gap and delta must have one length"));
    }
    let out: Vec<f64> = gap.iter().zip(delta).map(|(&g, &d)| probability(g, d, &coef, term, shift)).collect();
    Ok(out.into_pyarray_bound(py))
}

/// Stop counts (draws x entrants) the simulator gives: the shared plan rounded from
/// `stops_mean`, then each car's own count from `deviate` (three chances per car),
/// by the rules `simulate` uses (for checking the distribution).
#[pyfunction]
fn stop_counts<'py>(
    py: Python<'py>,
    stops_mean: f64,
    deviate: PyReadonlyArray1<f64>,
    entrants: usize,
    draws: usize,
    seed: u64,
) -> PyResult<Bound<'py, PyArray2<i64>>> {
    let deviate = deviate.as_slice()?.to_vec();
    if !deviate.is_empty() && deviate.len() != 3 * entrants {
        return Err(PyValueError::new_err("three deviation chances per entrant"));
    }
    let strategy = Strategy {
        wear: 0.0,
        pit_loss: 1.0,
        sc_ratio: 1.0,
        stops_mean,
        stint_noise: 0.0,
        window: 0,
        residuals: vec![],
        deviate,
        team: vec![],
        team_mean: vec![],
        team_sd: vec![],
    };
    let mut counts = vec![0i64; draws * entrants];
    for (d, row) in counts.chunks_mut(entrants.max(1)).enumerate() {
        let mut rng = Pcg64::seed_from_u64(seed ^ (d as u64 + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15));
        let count = strategy.count(&mut rng);
        for (car, out) in row.iter_mut().enumerate() {
            *out = strategy.own(car, count, &mut rng) as i64;
        }
    }
    numpy::ndarray::Array2::from_shape_vec((draws, entrants), counts)
        .map(|a| a.into_pyarray_bound(py))
        .map_err(|e| PyValueError::new_err(e.to_string()))
}

#[pymodule]
fn predictor_race_native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(simulate, m)?)?;
    m.add_function(wrap_pyfunction!(stop_counts, m)?)?;
    m.add_function(wrap_pyfunction!(pass_probability, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// One car, no lap time or noise: its race time is what its stops cost.
    fn lone_car(hazard_lap: usize, length: usize, laps: usize) -> f64 {
        let mut hazard = vec![0.0; laps];
        hazard[hazard_lap - 1] = 1.0;
        let race = Race {
            coef: coefficients(&[-0.8, -1.9, -0.3, -1.6, 0.2, 0.7]).unwrap(),
            term: 0.0,
            shift: 0.0,
            laps,
            start_noise: 0.0,
            start_spacing: 0.25,
            scale: 0.0,
            battle_gap: 2.5,
            min_follow: 0.35,
            neutral: Neutral {
                hazard,
                kinds: [1.0, 0.0, 0.0],
                lengths: [vec![length], vec![1], vec![1]],
                sc_gap: 0.52,
                red_end: 0.0,
            },
            strategy: Strategy {
                wear: 0.0,
                pit_loss: 22.0,
                sc_ratio: 0.5,
                stops_mean: 1.0,
                stint_noise: 0.0,
                window: 6,
                residuals: vec![],
                deviate: vec![],
                team: vec![],
                team_mean: vec![],
                team_sd: vec![],
            },
        };
        race_times(&race, &[0.0], &[1.0], &mut Pcg64::seed_from_u64(7))[0]
    }

    #[test]
    fn team_wear_is_shared_by_teammates_and_absent_without_teams() {
        let mut strategy = Strategy {
            wear: 0.05,
            pit_loss: 22.0,
            sc_ratio: 0.8,
            stops_mean: 1.0,
            stint_noise: 0.0,
            window: 6,
            residuals: vec![],
            deviate: vec![],
            team: vec![],
            team_mean: vec![],
            team_sd: vec![],
        };
        let mut rng = Pcg64::seed_from_u64(1);
        assert_eq!(strategy.car_wear(3, &mut rng), vec![0.05; 3]);
        strategy.team = vec![0, 1, 0];
        strategy.team_mean = vec![0.01, -0.02];
        strategy.team_sd = vec![0.0, 0.0];
        let wear = strategy.car_wear(3, &mut rng);
        assert!((wear[0] - 0.06).abs() < 1e-12 && (wear[1] - 0.03).abs() < 1e-12 && wear[0] == wear[2]);
        strategy.team_sd = vec![0.01, 0.01];
        let drawn = strategy.car_wear(3, &mut rng);
        assert_eq!(drawn[0], drawn[2]);
    }

    #[test]
    fn a_stop_coming_due_late_in_a_long_safety_car_is_taken_under_it() {
        // 20 laps, one stop planned on lap 10; safety car from lap 3 through lap 12:
        // the stop is outside the 6-lap window when the safety car comes out but
        // comes due on lap 4, so it costs the neutralised share, not a green stop
        assert!((lone_car(3, 10, 20) - 0.5 * 22.0).abs() < 1e-9);
        // a one-lap safety car (lap 3) ends before the stop is due: a green stop
        assert!((lone_car(3, 1, 20) - 22.0).abs() < 1e-9);
        // two laps (3-4): on lap 4 the stop is within the window, so it is taken then
        assert!((lone_car(3, 2, 20) - 0.5 * 22.0).abs() < 1e-9);
    }
}

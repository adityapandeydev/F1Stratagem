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
    let n = seconds.len();
    let mut time: Vec<f64> = grid
        .iter()
        .map(|&slot| race.start_spacing * (slot - 1.0) + race.start_noise * rng.sample::<f64, _>(StandardNormal))
        .collect();
    let (mut noise, mut luck) = (vec![0.0; n], vec![0.0; n]);
    let (mut order, mut before, mut after, mut own) =
        (Vec::with_capacity(n), Vec::with_capacity(n), Vec::with_capacity(n), Vec::with_capacity(n));
    let mut neutral_left = 0usize;
    for lap in 0..race.laps {
        if neutral_left == 0 && race.neutral.hazard[lap] > 0.0 && rng.gen::<f64>() < race.neutral.hazard[lap] {
            let pick = rng.gen::<f64>();
            let kind = if pick < race.neutral.kinds[0] {
                SC
            } else if pick < race.neutral.kinds[0] + race.neutral.kinds[1] {
                1
            } else {
                RED
            };
            let lengths = &race.neutral.lengths[kind];
            neutral_left = if lengths.is_empty() { 1 } else { lengths[rng.gen_range(0..lengths.len())].max(1) };
            if kind == RED && rng.gen::<f64>() < race.neutral.red_end {
                break; // the race is not restarted: classified as it stands
            }
            if kind != 1 {
                close_up(&mut time, &mut order, race.neutral.sc_gap);
            }
        }
        if neutral_left > 0 {
            neutral_left -= 1; // neutralised: nobody passes, pace does not act
            continue;
        }
        for value in noise.iter_mut() {
            *value = rng.sample(StandardNormal);
        }
        for value in luck.iter_mut() {
            *value = rng.gen::<f64>();
        }
        lap_of_draw(
            &mut time,
            seconds,
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
    }
    order.clear();
    order.extend(0..n);
    order.sort_by(|&a, &b| time[a].partial_cmp(&time[b]).unwrap());
    for (position, &car) in order.iter().enumerate() {
        positions[car] = position as i64 + 1;
    }
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

#[pymodule]
fn predictor_race_native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(simulate, m)?)?;
    m.add_function(wrap_pyfunction!(pass_probability, m)?)?;
    Ok(())
}

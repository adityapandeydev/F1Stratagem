//! Native lap-by-lap race simulation for the race model (`predictor.race_sim`).
//!
//! Python draws the random numbers (so a forecast is reproducible from its seed and
//! the same as the reference NumPy implementation) and calls `race_lap` once per
//! lap; this module runs the lap for every draw: order the cars, add each car's lap
//! time, then front to back let a car within `battle_gap` of the car ahead pass with
//! the battle model's probability, else hold it at least `min_follow` behind. At
//! most one place gained per car per lap. Draws run in parallel (rayon); each
//! draw's arithmetic is sequential, so results do not depend on the thread count.

use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1, PyReadonlyArray2, PyReadwriteArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
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
/// `seconds`, `noise`: per car (by entrant index); `luck`: per position.
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

/// Runs one lap for every draw, updating `time` (draws x entrants) in place.
/// `noise`: standard normal lap noise, scaled by `scale` s; `luck`: uniform draws
/// deciding passes (indexed by position in the running order).
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn race_lap(
    mut time: PyReadwriteArray2<f64>,
    seconds: PyReadonlyArray2<f64>,
    noise: PyReadonlyArray2<f64>,
    luck: PyReadonlyArray2<f64>,
    coef: PyReadonlyArray1<f64>,
    term: f64,
    shift: f64,
    scale: f64,
    battle_gap: f64,
    min_follow: f64,
) -> PyResult<()> {
    let coef = coefficients(coef.as_slice()?)?;
    let shape = time.as_array().shape().to_vec();
    for other in [seconds.as_array().shape(), noise.as_array().shape(), luck.as_array().shape()] {
        if other != shape.as_slice() {
            return Err(PyValueError::new_err("time, seconds, noise and luck must have one shape"));
        }
    }
    let n = shape[1];
    let time = time.as_slice_mut()?;
    let (seconds, noise, luck) = (seconds.as_slice()?, noise.as_slice()?, luck.as_slice()?);
    // draws are independent: run them in parallel (the result does not depend on it)
    time.par_chunks_mut(n).enumerate().for_each_init(
        || (Vec::with_capacity(n), Vec::with_capacity(n), Vec::with_capacity(n), Vec::with_capacity(n)),
        |(order, before, after, own), (d, row)| {
            let cars = d * n..(d + 1) * n;
            lap_of_draw(
                row,
                &seconds[cars.clone()],
                &noise[cars.clone()],
                &luck[cars],
                &coef,
                term,
                shift,
                scale,
                battle_gap,
                min_follow,
                order,
                before,
                after,
                own,
            );
        },
    );
    Ok(())
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
    m.add_function(wrap_pyfunction!(race_lap, m)?)?;
    m.add_function(wrap_pyfunction!(pass_probability, m)?)?;
    Ok(())
}

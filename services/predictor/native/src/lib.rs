//! Native Kalman filter for the pace model.
//!
//! Replays a program recorded by `predictor.program.Recorder` (which states are
//! added, parked, reset and observed, in order) for one set of settings and
//! returns the log-likelihood (the quantity the settings fit maximises) and the
//! final mean and covariance (what a forecast reads). Design rows are used
//! sparsely (each driver row touches a handful of states), so an update costs
//! O(m * n^2) for a group of m drivers and n states.
//!
//! Two tracks: 0 is the filter; 1 is a copy taken by SNAPSHOT at an event start,
//! for the qualifying objective's before-the-weekend horizon.

use numpy::{IntoPyArray, PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray1};
use pyo3::exceptions::{PyArithmeticError, PyValueError};
use pyo3::prelude::*;

const ADVANCE: i64 = 0;
const ADD: i64 = 1;
const PARK: i64 = 2;
const RESET: i64 = 3;
const UPDATE: i64 = 4;
const SNAPSHOT: i64 = 5;
const TRACKS: usize = 2;
const KINDS: usize = 8; // car, driver, offset, weekend, affinity, form, skill, season_skill

pub struct Settings<'a> {
    pub priors: &'a [f64],
    pub drifts: &'a [f64],
    pub winters: &'a [f64],
    pub multiplied: &'a [f64],
    pub regulation_multiplier: f64,
    pub noise: &'a [f64],
    pub estimate_scale: f64, // multiplies a group's estimation covariance (laps fitted together)
}

pub struct Program<'a> {
    pub ops: &'a [i64], // rows of 5: code, track, a, b, c
    pub floats: &'a [f64],
    pub ints: &'a [i64],
    pub group_rows: &'a [i64], // rows of 4: first row, drivers, noise type, covariance offset in floats or -1
    pub row_starts: &'a [i64],
    pub entries: &'a [i64],
    pub weights: &'a [f64], // one per entry: how much of that state the driver's pace reads
    pub values: &'a [f64],
    pub slots: usize,
}

#[derive(Clone)]
pub struct State {
    pub n: usize,
    cap: usize,
    kind: Vec<usize>,
    pub mean: Vec<f64>,
    cov: Vec<f64>, // cap x cap, row-major; the leading n x n block is live
}

impl State {
    fn new() -> Self {
        State { n: 0, cap: 0, kind: Vec::new(), mean: Vec::new(), cov: Vec::new() }
    }

    /// The live n x n covariance, row-major.
    pub fn covariance(&self) -> Vec<f64> {
        let mut out = Vec::with_capacity(self.n * self.n);
        for i in 0..self.n {
            out.extend_from_slice(&self.cov[i * self.cap..i * self.cap + self.n]);
        }
        out
    }

    fn grow(&mut self) {
        let cap = (self.cap * 2).max(16);
        let mut cov = vec![0.0; cap * cap];
        for i in 0..self.n {
            cov[i * cap..i * cap + self.n].copy_from_slice(&self.cov[i * self.cap..i * self.cap + self.n]);
        }
        self.cov = cov;
        self.cap = cap;
    }

    fn add(&mut self, kind: usize, mean: f64, variance: f64) {
        if self.n == self.cap {
            self.grow();
        }
        let (n, cap) = (self.n, self.cap);
        for i in 0..=n {
            self.cov[i * cap + n] = 0.0;
            self.cov[n * cap + i] = 0.0;
        }
        self.cov[n * cap + n] = variance;
        self.mean.push(mean);
        self.kind.push(kind);
        self.n += 1;
    }

    fn remove(&mut self, removed: &[usize]) {
        let keep: Vec<usize> = (0..self.n).filter(|i| !removed.contains(i)).collect();
        let cap = self.cap;
        for (a, &i) in keep.iter().enumerate() {
            for (b, &j) in keep.iter().enumerate() {
                self.cov[a * cap + b] = self.cov[i * cap + j];
            }
        }
        self.mean = keep.iter().map(|&i| self.mean[i]).collect();
        self.kind = keep.iter().map(|&i| self.kind[i]).collect();
        self.n = keep.len();
    }
}

fn winter(s: &Settings, kind: usize, ordinary: i64, regulation: i64) -> f64 {
    let base = s.winters[kind];
    let multiplied = if s.multiplied[kind] > 0.0 { base * s.regulation_multiplier } else { base };
    ordinary as f64 * base + regulation as f64 * multiplied
}

/// In-place Cholesky of a k x k row-major matrix; false if not positive definite.
fn cholesky(a: &mut [f64], k: usize) -> bool {
    for j in 0..k {
        let mut d = a[j * k + j];
        for p in 0..j {
            d -= a[j * k + p] * a[j * k + p];
        }
        if !(d > 0.0) {
            return false;
        }
        let d = d.sqrt();
        a[j * k + j] = d;
        for i in j + 1..k {
            let mut v = a[i * k + j];
            for p in 0..j {
                v -= a[i * k + p] * a[j * k + p];
            }
            a[i * k + j] = v / d;
        }
        for p in j + 1..k {
            a[j * k + p] = 0.0;
        }
    }
    true
}

/// Solves L L^T x = b in place for a Cholesky factor L.
fn cholesky_solve(l: &[f64], k: usize, b: &mut [f64]) {
    for i in 0..k {
        let mut v = b[i];
        for p in 0..i {
            v -= l[i * k + p] * b[p];
        }
        b[i] = v / l[i * k + i];
    }
    for i in (0..k).rev() {
        let mut v = b[i];
        for p in i + 1..k {
            v -= l[p * k + i] * b[p];
        }
        b[i] = v / l[i * k + i];
    }
}

/// Why a replay stopped.
#[derive(Debug)]
pub enum Failure {
    /// The settings make an innovation covariance numerically not positive definite
    /// (e.g. noise so small that the covariance update loses its precision).
    Numerical(String),
    /// The program itself is malformed.
    Invalid(String),
}

impl From<Failure> for PyErr {
    fn from(failure: Failure) -> PyErr {
        match failure {
            Failure::Numerical(message) => PyArithmeticError::new_err(message),
            Failure::Invalid(message) => PyValueError::new_err(message),
        }
    }
}

/// Replays the program: counted log-likelihood per track, and track 0's final state.
pub fn run(p: &Program, s: &Settings) -> Result<([f64; TRACKS], State), Failure> {
    if p.weights.len() != p.entries.len() {
        return Err(Failure::Invalid(format!("{} weights for {} entries", p.weights.len(), p.entries.len())));
    }
    for (name, values) in [("priors", s.priors), ("drifts", s.drifts), ("winters", s.winters), ("multiplied", s.multiplied)] {
        if values.len() != KINDS {
            return Err(Failure::Invalid(format!("{name} has {} values for {KINDS} kinds", values.len())));
        }
    }
    let mut states = [State::new(), State::new()];
    let mut parkeds = [vec![(0.0f64, 0.0f64); p.slots], vec![(0.0f64, 0.0f64); p.slots]];
    let mut likelihoods = [0.0; TRACKS];
    let log_two_pi = (2.0 * std::f64::consts::PI).ln();
    // scratch buffers reused across updates
    let (mut hp, mut sm, mut innovation): (Vec<f64>, Vec<f64>, Vec<f64>) = (Vec::new(), Vec::new(), Vec::new());

    for op in p.ops.chunks_exact(5) {
        let (code, track, a, b, c) = (op[0], op[1] as usize, op[2], op[3], op[4]);
        if track >= TRACKS {
            return Err(Failure::Invalid(format!("unknown track {track}")));
        }
        if code == SNAPSHOT {
            states[1] = states[0].clone();
            parkeds[1] = parkeds[0].clone();
            continue;
        }
        let st = &mut states[track];
        let parked = &mut parkeds[track];
        match code {
            ADVANCE => {
                let steps = p.floats[a as usize];
                let cap = st.cap;
                for i in 0..st.n {
                    let k = st.kind[i];
                    st.cov[i * cap + i] += s.drifts[k] * steps + winter(s, k, b, c);
                }
            }
            ADD => {
                let kind = a as usize;
                if kind >= KINDS {
                    return Err(Failure::Invalid(format!("unknown kind {kind}")));
                }
                if b >= 0 {
                    let (mean, mut variance) = parked[b as usize];
                    let at = c as usize;
                    let steps_index = p.ints[at];
                    if steps_index >= 0 {
                        let steps = p.floats[steps_index as usize];
                        variance += s.drifts[kind] * steps + winter(s, kind, p.ints[at + 1], p.ints[at + 2]);
                    }
                    st.add(kind, mean, variance);
                } else {
                    st.add(kind, 0.0, s.priors[kind] * s.priors[kind]);
                }
            }
            PARK => {
                let pairs = &p.ints[a as usize..(a + 2 * b) as usize];
                let mut removed = Vec::with_capacity(b as usize);
                for pair in pairs.chunks_exact(2) {
                    let (i, slot) = (pair[0] as usize, pair[1] as usize);
                    parked[slot] = (st.mean[i], st.cov[i * st.cap + i]);
                    removed.push(i);
                }
                st.remove(&removed);
            }
            RESET => {
                // c = the kind whose prior the states return to (weekend effects, season form)
                if c < 0 || c as usize >= KINDS {
                    return Err(Failure::Invalid(format!("unknown kind {c} in RESET")));
                }
                let cap = st.cap;
                let prior = s.priors[c as usize] * s.priors[c as usize];
                for &i in &p.ints[a as usize..(a + b) as usize] {
                    let i = i as usize;
                    st.mean[i] = 0.0;
                    for j in 0..st.n {
                        st.cov[i * cap + j] = 0.0;
                        st.cov[j * cap + i] = 0.0;
                    }
                    st.cov[i * cap + i] = prior;
                }
            }
            UPDATE => {
                let g = &p.group_rows[4 * a as usize..4 * a as usize + 4];
                let (first, drivers, noise_type) = (g[0] as usize, g[1] as usize, g[2] as usize);
                let estimate = g[3];
                let m = drivers - 1;
                let (n, cap) = (st.n, st.cap);
                let row = |r: usize| -> std::iter::Zip<std::slice::Iter<'_, i64>, std::slice::Iter<'_, f64>> {
                    let span = p.row_starts[first + r] as usize..p.row_starts[first + r + 1] as usize;
                    p.entries[span.clone()].iter().zip(p.weights[span].iter())
                };
                let base = || row(0);
                // HP = h @ cov, one row per driver after the first (h_i = d_{i+1} - d_0)
                hp.clear();
                hp.resize(m * n, 0.0);
                let mut base_row = vec![0.0; n];
                for (&j, &w) in base() {
                    let j = j as usize;
                    for q in 0..n {
                        base_row[q] += w * st.cov[j * cap + q];
                    }
                }
                innovation.clear();
                let base_mean: f64 = base().map(|(&j, &w)| w * st.mean[j as usize]).sum();
                for i in 0..m {
                    let out = &mut hp[i * n..(i + 1) * n];
                    for (&j, &w) in row(i + 1) {
                        let j = j as usize;
                        for q in 0..n {
                            out[q] += w * st.cov[j * cap + q];
                        }
                    }
                    for q in 0..n {
                        out[q] -= base_row[q];
                    }
                    let predicted: f64 = row(i + 1).map(|(&j, &w)| w * st.mean[j as usize]).sum::<f64>() - base_mean;
                    innovation.push((p.values[first + i + 1] - p.values[first]) - predicted);
                }
                // S = HP @ h^T + R
                let variance = s.noise[noise_type] * s.noise[noise_type];
                sm.clear();
                sm.resize(m * m, 0.0);
                for i in 0..m {
                    let hp_row = &hp[i * n..(i + 1) * n];
                    let base_part: f64 = base().map(|(&j, &w)| w * hp_row[j as usize]).sum();
                    for k in 0..m {
                        let part: f64 = row(k + 1).map(|(&j, &w)| w * hp_row[j as usize]).sum();
                        sm[i * m + k] = part - base_part + variance * (if i == k { 2.0 } else { 1.0 });
                    }
                }
                if estimate >= 0 && s.estimate_scale > 0.0 {
                    // estimation covariance C of the group's values (drivers fitted together):
                    // cov(v_{i+1} - v_0, v_{k+1} - v_0) = C[i+1,k+1] - C[i+1,0] - C[0,k+1] + C[0,0]
                    let d = drivers;
                    let c = &p.floats[estimate as usize..estimate as usize + d * d];
                    for i in 0..m {
                        for k in 0..m {
                            sm[i * m + k] += s.estimate_scale
                                * (c[(i + 1) * d + k + 1] - c[(i + 1) * d] - c[k + 1] + c[0]);
                        }
                    }
                }
                for i in 0..m {
                    for k in 0..i {
                        let v = 0.5 * (sm[i * m + k] + sm[k * m + i]);
                        sm[i * m + k] = v;
                        sm[k * m + i] = v;
                    }
                }
                if !cholesky(&mut sm, m) {
                    return Err(Failure::Numerical("innovation covariance is not positive definite".into()));
                }
                // log-likelihood of the innovation
                let mut solved = innovation.clone();
                for i in 0..m {
                    let mut v = solved[i];
                    for q in 0..i {
                        v -= sm[i * m + q] * solved[q];
                    }
                    solved[i] = v / sm[i * m + i];
                }
                let quadratic: f64 = solved.iter().map(|v| v * v).sum();
                let log_det: f64 = (0..m).map(|i| sm[i * m + i].ln()).sum();
                if b != 0 {
                    likelihoods[track] += -0.5 * quadratic - log_det - 0.5 * m as f64 * log_two_pi;
                }
                // W = S^-1 HP (column by column), mean += HP^T S^-1 innovation, cov -= HP^T W
                let mut weights = innovation.clone();
                cholesky_solve(&sm, m, &mut weights);
                // transposed (state-major) copies so the n x n loop reads contiguous memory
                let mut hp_t = vec![0.0; n * m];
                let mut w_t = vec![0.0; n * m];
                let mut column = vec![0.0; m];
                for q in 0..n {
                    for i in 0..m {
                        column[i] = hp[i * n + q];
                        hp_t[q * m + i] = column[i];
                    }
                    cholesky_solve(&sm, m, &mut column);
                    w_t[q * m..(q + 1) * m].copy_from_slice(&column);
                }
                for q in 0..n {
                    let h = &hp_t[q * m..(q + 1) * m];
                    st.mean[q] += h.iter().zip(&weights).map(|(a, b)| a * b).sum::<f64>();
                }
                for r in 0..n {
                    let h = &hp_t[r * m..(r + 1) * m];
                    for q in r..n {
                        let w = &w_t[q * m..(q + 1) * m];
                        let v: f64 = h.iter().zip(w).map(|(a, b)| a * b).sum();
                        let symmetric = 0.5 * ((st.cov[r * cap + q] - v) + (st.cov[q * cap + r] - v));
                        st.cov[r * cap + q] = symmetric;
                        st.cov[q * cap + r] = symmetric;
                    }
                }
            }
            _ => return Err(Failure::Invalid(format!("unknown opcode {code}"))),
        }
    }
    let [first, _] = states;
    Ok((likelihoods, first))
}

/// Borrows the numpy inputs as a program and settings, and replays it without the GIL.
#[allow(clippy::too_many_arguments)]
fn replay(
    py: Python<'_>,
    ops: &PyReadonlyArray1<i64>,
    floats: &PyReadonlyArray1<f64>,
    ints: &PyReadonlyArray1<i64>,
    group_rows: &PyReadonlyArray1<i64>,
    row_starts: &PyReadonlyArray1<i64>,
    entries: &PyReadonlyArray1<i64>,
    weights: &PyReadonlyArray1<f64>,
    values: &PyReadonlyArray1<f64>,
    slots: usize,
    priors: &PyReadonlyArray1<f64>,
    drifts: &PyReadonlyArray1<f64>,
    winters: &PyReadonlyArray1<f64>,
    multiplied: &PyReadonlyArray1<f64>,
    regulation_multiplier: f64,
    noise: &PyReadonlyArray1<f64>,
    estimate_scale: f64,
) -> PyResult<([f64; TRACKS], State)> {
    let program = Program {
        ops: ops.as_slice()?,
        floats: floats.as_slice()?,
        ints: ints.as_slice()?,
        group_rows: group_rows.as_slice()?,
        row_starts: row_starts.as_slice()?,
        entries: entries.as_slice()?,
        weights: weights.as_slice()?,
        values: values.as_slice()?,
        slots,
    };
    let settings = Settings {
        priors: priors.as_slice()?,
        drifts: drifts.as_slice()?,
        winters: winters.as_slice()?,
        multiplied: multiplied.as_slice()?,
        regulation_multiplier,
        noise: noise.as_slice()?,
        estimate_scale,
    };
    py.allow_threads(|| run(&program, &settings)).map_err(PyErr::from)
}

/// Counted log-likelihood of a recorded program under one set of settings.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn log_likelihood(
    py: Python<'_>,
    ops: PyReadonlyArray1<i64>,
    floats: PyReadonlyArray1<f64>,
    ints: PyReadonlyArray1<i64>,
    group_rows: PyReadonlyArray1<i64>,
    row_starts: PyReadonlyArray1<i64>,
    entries: PyReadonlyArray1<i64>,
    weights: PyReadonlyArray1<f64>,
    values: PyReadonlyArray1<f64>,
    slots: usize,
    priors: PyReadonlyArray1<f64>,
    drifts: PyReadonlyArray1<f64>,
    winters: PyReadonlyArray1<f64>,
    multiplied: PyReadonlyArray1<f64>,
    regulation_multiplier: f64,
    noise: PyReadonlyArray1<f64>,
    estimate_scale: f64,
) -> PyResult<f64> {
    let (likelihoods, _) = replay(py, &ops, &floats, &ints, &group_rows, &row_starts, &entries, &weights, &values, slots,
                                  &priors, &drifts, &winters, &multiplied, regulation_multiplier, &noise, estimate_scale)?;
    Ok(likelihoods.iter().sum())
}

/// Counted log-likelihood per track, and track 0's final mean and covariance.
#[pyfunction]
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn filter_state<'py>(
    py: Python<'py>,
    ops: PyReadonlyArray1<i64>,
    floats: PyReadonlyArray1<f64>,
    ints: PyReadonlyArray1<i64>,
    group_rows: PyReadonlyArray1<i64>,
    row_starts: PyReadonlyArray1<i64>,
    entries: PyReadonlyArray1<i64>,
    weights: PyReadonlyArray1<f64>,
    values: PyReadonlyArray1<f64>,
    slots: usize,
    priors: PyReadonlyArray1<f64>,
    drifts: PyReadonlyArray1<f64>,
    winters: PyReadonlyArray1<f64>,
    multiplied: PyReadonlyArray1<f64>,
    regulation_multiplier: f64,
    noise: PyReadonlyArray1<f64>,
    estimate_scale: f64,
) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>, Bound<'py, PyArray2<f64>>)> {
    let (likelihoods, state) = replay(py, &ops, &floats, &ints, &group_rows, &row_starts, &entries, &weights, &values, slots,
                                      &priors, &drifts, &winters, &multiplied, regulation_multiplier, &noise, estimate_scale)?;
    let n = state.n;
    let cov = state.covariance().into_pyarray_bound(py).reshape([n, n])?;
    Ok((likelihoods.to_vec().into_pyarray_bound(py), state.mean.clone().into_pyarray_bound(py), cov))
}

#[pymodule]
fn predictor_native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(log_likelihood, m)?)?;
    m.add_function(wrap_pyfunction!(filter_state, m)?)?;
    Ok(())
}

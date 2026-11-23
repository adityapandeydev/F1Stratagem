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
//! number is drawn for it. With a pit hazard there are no plans: at every lap's
//! end (and on each neutralised lap, at the neutralised loss) each car pits with
//! the hazard of what it sees: tyre age, the race's progress, places it would lose
//! and the clean air where it would rejoin, the gaps around it, a rival who pitted
//! on the lap before, its pace against the car it would rejoin behind and the
//! circuit's passing. A car without a stop is called in 3 laps from the end. With team wear, each team's wear deviation from the
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
    follow: &[f64],
    follow_u: &[f64],
    held_gap: &mut [f64],
    held_ahead: &mut [usize],
    attack_u: &[f64],
    release: f64,
    penalty: &[f64; 3],
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
    if penalty.iter().any(|&p| p != 0.0) {
        // following penalty: a car starting the lap within 1, 2 or 3 s of the car
        // ahead loses that band's time on it
        for k in 1..n {
            let gap = before[k] - before[k - 1];
            let band = if gap < 1.0 { 0 } else if gap < 2.0 { 1 } else if gap < 3.0 { 2 } else { 3 };
            if band < 3 {
                after[k] += penalty[band];
            }
        }
    }
    for k in 1..n {
        let gap = before[k] - before[k - 1];
        let delta = own[k] - own[k - 1];
        let chance = if gap < battle_gap { probability(gap, delta, coef, term, shift) } else { 0.0 };
        let passed = luck[k] < chance; // by position, like the NumPy version
        let (t_ahead, t_behind) = (after[k - 1], after[k]);
        let behind_car = order[k];
        let (first, second) = if passed {
            let mut first = t_ahead.min(t_behind);
            if k >= 2 {
                first = first.max(after[k - 2] + min_follow); // the passer stays behind the car two ahead
            }
            if !follow.is_empty() {
                held_gap[behind_car] = f64::NAN;
            }
            (first, t_ahead.max(first + min_follow))
        } else if follow.is_empty() {
            (t_ahead, t_behind.max(t_ahead + min_follow))
        } else {
            // held up: the follower settles at a gap of its own, drawn when the hold
            // begins and kept while it would otherwise run closer to the same car;
            // released when it falls back on its own, passes, or another car is ahead
            let ahead_car = order[k - 1];
            let holding = !held_gap[behind_car].is_nan() && held_ahead[behind_car] == ahead_car;
            if holding && attack_u[behind_car] < release {
                // commits to an attack: leaves its gap and closes in (the battle model
                // decides the pass from the next lap's gap)
                held_gap[behind_car] = f64::NAN;
                (t_ahead, t_behind.max(t_ahead + min_follow))
            } else if holding && t_behind < t_ahead + held_gap[behind_car] {
                (t_ahead, t_ahead + held_gap[behind_car])
            } else if !holding && t_behind < t_ahead + min_follow {
                held_gap[behind_car] = quantile(follow, follow_u[behind_car]).max(min_follow);
                held_ahead[behind_car] = ahead_car;
                (t_ahead, t_ahead + held_gap[behind_car])
            } else {
                held_gap[behind_car] = f64::NAN;
                (t_ahead, t_behind)
            }
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

/// The value at `u` (0-1) of a distribution given by equally spaced quantiles.
fn quantile(table: &[f64], u: f64) -> f64 {
    let x = u.clamp(0.0, 1.0) * (table.len() - 1) as f64;
    let i = (x.floor() as usize).min(table.len() - 2);
    table[i] + (x - i as f64) * (table[i + 1] - table[i])
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
    /// Pit hazard (optional): intercept then one coefficient per feature (the order
    /// of `predictor.race_sim.PIT_FEATURES`); `passing`: the circuit's centred log
    /// pass rate; `lap_seconds`: a typical lap. Empty: stops follow the plans.
    pit_coef: Vec<f64>,
    passing: f64,
    lap_seconds: f64,
}

/// The random numbers of one draw. `Seq`: drawn in sequence from the draw's
/// generator. `Keyed` (forecasts and settings fits): each number a hash of (seed,
/// purpose, lap, car), so variants of one race (other settings, pit timing, no
/// traffic) share the same lap noise, battles, neutralisations and starts: moving a
/// stop does not shift which random number later becomes lap noise or a pass.
enum Rand<'a> {
    Seq(&'a mut Pcg64),
    Keyed(u64),
}

const P_START: u64 = 1;
const P_NEUTRAL: u64 = 2;
const P_KIND: u64 = 3;
const P_LENGTH: u64 = 4;
const P_RED: u64 = 5;
const P_COUNT: u64 = 6;
const P_OWN: u64 = 7;
const P_PLAN: u64 = 8;
const P_TEAM: u64 = 9;
const P_COST: u64 = 10;
const P_DECIDE: u64 = 11;
const P_NOISE: u64 = 12;
const P_LUCK: u64 = 13;
const P_FOLLOW: u64 = 14;
const P_ATTACK: u64 = 15;
const P_ROLE: u64 = 16;

/// Grid bands of the starting tyre table: slots 1-3, 4-10, 11-15, 16 on (and the pit lane).
fn role_band(slot: f64) -> usize {
    if slot >= 1.0 && slot <= 3.0 { 0 } else if slot >= 1.0 && slot <= 10.0 { 1 } else if slot >= 1.0 && slot <= 15.0 { 2 } else { 3 }
}

fn splitmix(mut z: u64) -> u64 {
    z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

impl Rand<'_> {
    fn key(seed: u64, purpose: u64, lap: usize, car: usize, part: u64) -> f64 {
        let h = splitmix(splitmix(splitmix(splitmix(seed ^ purpose) ^ lap as u64) ^ car as u64) ^ part);
        ((h >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    }

    fn uniform(&mut self, purpose: u64, lap: usize, car: usize) -> f64 {
        match self {
            Rand::Seq(rng) => rng.gen::<f64>(),
            Rand::Keyed(seed) => Self::key(*seed, purpose, lap, car, 0),
        }
    }

    fn normal(&mut self, purpose: u64, lap: usize, car: usize) -> f64 {
        match self {
            Rand::Seq(rng) => rng.sample(StandardNormal),
            Rand::Keyed(seed) => {
                let (u, v) = (Self::key(*seed, purpose, lap, car, 0), Self::key(*seed, purpose, lap, car, 1));
                (-2.0 * u.ln()).sqrt() * (std::f64::consts::TAU * v).cos()
            }
        }
    }

    fn index(&mut self, purpose: u64, lap: usize, car: usize, len: usize) -> usize {
        match self {
            Rand::Seq(rng) => rng.gen_range(0..len),
            Rand::Keyed(seed) => ((Self::key(*seed, purpose, lap, car, 0) * len as f64) as usize).min(len - 1),
        }
    }
}

const AGE_EDGES: [f64; 7] = [5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 40.0];
const PIT_FEATURES: usize = 27;
const PIT_FIRST_LAP: usize = 2;
const PIT_LAST_LAPS: usize = 3;
const END_LAPS: f64 = 8.0;
const OLD_TYRES: f64 = 10.0;
const COVER_GAP: f64 = 3.0;
const GAP_CAP: f64 = 20.0;

impl Strategy {
    fn enabled(&self) -> bool {
        self.pit_loss > 0.0
    }

    /// Planned stops of one simulated race: the circuit's typical count rounded at
    /// random, shared by the field (rounding per car would split it far more often
    /// than real strategies do; cars leave the plan through `deviate`).
    fn count(&self, rng: &mut Rand) -> usize {
        let whole = self.stops_mean.floor();
        (whole as usize + usize::from(rng.uniform(P_COUNT, 0, 0) < self.stops_mean - whole)).max(1)
    }

    /// Stops of `car` in a race planned for `count` (no random number without `deviate`).
    fn own(&self, car: usize, count: usize, rng: &mut Rand) -> usize {
        if self.deviate.is_empty() {
            return count;
        }
        let chance = &self.deviate[3 * car..3 * car + 3];
        let u = rng.uniform(P_OWN, 0, car);
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

    /// Planned stop laps (1-based, ascending) of one car making `count` stops: at the
    /// learned share of the race of each stop (`windows`, see `Race::windows`), else at
    /// evenly spread stints, with the stint noise.
    fn plan(&self, car: usize, laps: usize, count: usize, windows: &[f64], shift: Option<&[f64]>, role: usize,
            rng: &mut Rand) -> Vec<usize> {
        let last = laps.saturating_sub(1).max(2);
        let offset = count * count.saturating_sub(1) / 2; // counts 1, 2, 3: offsets 0, 1, 3
        let learned = count >= 1 && offset + count <= windows.len();
        let mut plan: Vec<usize> = (1..=count)
            .map(|j| {
                let mut share = if learned { windows[offset + j - 1] } else { j as f64 / (count + 1) as f64 };
                if let (true, Some(s)) = (learned, shift) {
                    share += s[3 * (offset + j - 1) + role];
                }
                let at = laps as f64 * share + self.stint_noise * rng.normal(P_PLAN, j, car);
                (at.round().max(2.0) as usize).min(last)
            })
            .collect();
        plan.sort_unstable();
        plan
    }

    /// Each car's wear in one simulated race (no random number without team wear).
    fn car_wear(&self, n: usize, rng: &mut Rand) -> Vec<f64> {
        if self.team.is_empty() {
            return vec![self.wear; n];
        }
        let draws: Vec<f64> = self
            .team_mean
            .iter()
            .zip(&self.team_sd)
            .enumerate()
            .map(|(t, (&mean, &sd))| mean + sd * rng.normal(P_TEAM, 0, t))
            .collect();
        self.team.iter().map(|&t| (self.wear + draws[t]).max(0.0)).collect()
    }

    fn hazard(&self) -> bool {
        !self.pit_coef.is_empty()
    }

    /// The pit hazard's features of one car at a lap's end (as `race_sim.pit_features`).
    #[allow(clippy::too_many_arguments)]
    fn features(&self, age: f64, stops: usize, lap: usize, laps: usize, neutral: bool, position: usize, lost: f64,
                clean_air: f64, gap_ahead: f64, gap_behind: f64, cover_behind: bool, cover_ahead: bool,
                pace_adv: f64, wear: f64) -> [f64; PIT_FEATURES] {
        let mut x = [0.0; PIT_FEATURES];
        for (k, &edge) in AGE_EDGES.iter().enumerate() {
            let above_next = AGE_EDGES.get(k + 1).map_or(false, |&next| age > next);
            x[k] = f64::from(u8::from(age > edge && !above_next));
        }
        let frac = lap as f64 / laps as f64;
        let adv = pace_adv.clamp(-3.0, 3.0);
        let flag = |b: bool| f64::from(u8::from(b));
        let cap = |g: f64| g.min(GAP_CAP).ln_1p();
        let rest = [flag(stops == 0), frac, frac * frac, flag(laps as f64 - lap as f64 <= END_LAPS), flag(neutral),
                    flag(neutral && age >= OLD_TYRES), position as f64 / 20.0, lost / 10.0, cap(clean_air), cap(gap_ahead),
                    cap(gap_behind), flag(cover_behind), flag(cover_ahead), adv, self.passing, adv * self.passing,
                    lost / 10.0 * self.passing];
        x[7..24].copy_from_slice(&rest);
        let wear = wear.clamp(0.0, 0.3);
        x[24] = wear * age;
        x[25] = x[24] * x[24];
        x[26] = wear;
        x
    }

    /// The hazard's features of every car deciding at the start of `lap` (1-based)
    /// whether to pit at its end, from the field as it crossed the line at the end
    /// of the lap before (`time`, before any of this lap's stops): `age`: tyre age
    /// on this lap (laps on the set including this one, as FastF1's TyreLife),
    /// `neutral`: this lap is neutralised, `loss`: the stop's expected loss on it,
    /// `pitted`: who pitted at the end of the lap before.
    #[allow(clippy::too_many_arguments)]
    fn state_features(&self, lap: usize, laps: usize, neutral: bool, loss: f64, time: &[f64], pace: &[f64], age: &[f64],
                      wear: &[f64], stops: &[usize], pitted: &[bool]) -> Vec<[f64; PIT_FEATURES]> {
        let n = time.len();
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by(|&a, &b| time[a].partial_cmp(&time[b]).unwrap());
        let mut out = vec![[0.0; PIT_FEATURES]; n];
        for (i, &car) in order.iter().enumerate() {
            let (t, rejoin) = (time[car], time[car] + loss);
            let (mut lost, mut before, mut after, mut target) = (0.0, f64::NEG_INFINITY, f64::INFINITY, None);
            for &other in order.iter() {
                if other == car {
                    continue;
                }
                let u = time[other];
                if u > t && u <= rejoin {
                    lost += 1.0;
                    target = Some(other); // the last car to pass: the one it rejoins behind
                }
                if u <= rejoin {
                    before = before.max(u);
                } else {
                    after = after.min(u);
                }
            }
            let clean_air = (if before.is_finite() { rejoin - before } else { GAP_CAP })
                .min(if after.is_finite() { after - rejoin } else { GAP_CAP });
            let gap_ahead = if i > 0 { t - time[order[i - 1]] } else { GAP_CAP };
            let gap_behind = if i + 1 < n { time[order[i + 1]] - t } else { GAP_CAP };
            let cover_behind = i + 1 < n && pitted[order[i + 1]] && gap_behind < COVER_GAP;
            let cover_ahead = i > 0 && pitted[order[i - 1]] && gap_ahead < COVER_GAP;
            let pace_adv = target.map_or(0.0, |o| (pace[o] - pace[car]) / self.lap_seconds * 100.0);
            // the car's wear in % of a lap per lap of age (the simulator's is in s)
            out[car] = self.features(age[car], stops[car], lap, laps, neutral, i + 1, lost, clean_air, gap_ahead,
                                     gap_behind, cover_behind, cover_ahead, pace_adv, wear[car] / self.lap_seconds * 100.0);
        }
        out
    }

    /// Which cars pit at the end of `lap`, decided at its start (`state_features`):
    /// each by its hazard, a car without a stop called in PIT_LAST_LAPS from the end.
    #[allow(clippy::too_many_arguments)]
    fn decide(&self, lap: usize, laps: usize, neutral: bool, loss: f64, time: &[f64], pace: &[f64], age: &[f64],
              wear: &[f64], stops: &[usize], pitted: &[bool], rng: &mut Rand) -> Vec<bool> {
        let n = time.len();
        let mut out = vec![false; n];
        if lap < PIT_FIRST_LAP || lap + PIT_LAST_LAPS > laps {
            return out;
        }
        let x = self.state_features(lap, laps, neutral, loss, time, pace, age, wear, stops, pitted);
        for car in 0..n {
            if stops[car] == 0 && lap + PIT_LAST_LAPS == laps {
                out[car] = true; // a dry race needs a stop
                continue;
            }
            let z = self.pit_coef[0] + x[car].iter().zip(&self.pit_coef[1..]).map(|(a, b)| a * b).sum::<f64>();
            out[car] = rng.uniform(P_DECIDE, lap, car) < 1.0 / (1.0 + (-z).exp());
        }
        out
    }

    fn cost(&self, lap: usize, car: usize, rng: &mut Rand) -> f64 {
        let extra = if self.residuals.is_empty() {
            0.0
        } else {
            self.residuals[rng.index(P_COST, lap, car, self.residuals.len())]
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
    /// following gaps of a held-up car (equally spaced quantiles, s); empty: held at min_follow
    follow: Vec<f64>,
    /// per-lap chance that a held car leaves its gap to attack
    follow_release: f64,
    /// lap-time penalty (s) of a car starting the lap within 1, 2 or 3 s of the car ahead
    follow_penalty: [f64; 3],
    /// fresh-tyre peak (s faster) on the 1st, 2nd, ... green lap of a set fitted at a
    /// stop (not the starting set), on top of the wear line; empty: none
    fresh_peak: Vec<f64>,
    /// learned pit windows: the share of the race at which each stop of a car making
    /// 1, 2, 3 stops is planned, flat ([1-stop], [2-stop: 1st, 2nd], [3-stop: ...]);
    /// empty (or a count beyond it): evenly spread stints
    windows: Vec<f64>,
    /// starting tyre roles (optional): per grid band (`role_band`) the chances of
    /// starting on the soft, medium and hard role (4 x 3, flat), and per learned window
    /// (the layout of `windows`) each role's shift of that stop (share of the race; 3
    /// per window, flat). Empty: no roles, every car on the plain windows.
    role_chances: Vec<f64>,
    role_shift: Vec<f64>,
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
fn race_of_draw(race: &Race, seconds: &[f64], grid: &[f64], rng: &mut Rand, positions: &mut [i64], stops: &mut [i64]) {
    let (time, made) = race_times(race, seconds, grid, rng, &Diag::default(), None);
    for (out, &k) in stops.iter_mut().zip(&made) {
        *out = k as i64;
    }
    rank(&time, positions);
}

fn rank(time: &[f64], positions: &mut [i64]) {
    let mut order: Vec<usize> = (0..time.len()).collect();
    order.sort_by(|&a, &b| time[a].partial_cmp(&time[b]).unwrap());
    for (position, &car) in order.iter().enumerate() {
        positions[car] = position as i64 + 1;
    }
}

/// Diagnostic variants of a simulated race (none for forecasts): `fixed`: each
/// car's stop count for planned stops (instead of drawing it); `free`: no traffic
/// (each car runs its own lap, nobody is held up or passes).
#[derive(Default)]
struct Diag<'a> {
    fixed: Option<&'a [i64]>,
    free: bool,
    /// each car's planned stop laps (1-based), instead of evenly spread ones
    laps_of: Option<&'a [Vec<usize>]>,
}

/// What happened in one draw, lap by lap (`lap * n + car`): race time at the lap's
/// end, tyre age (laps on the set), the stop made at its end (0 none, 1 green, 2
/// under a neutralisation, 3 called in to make the race's stop, 4 a new set under a
/// red flag) and the time lost being held up (s, against the lap the car would
/// have run alone). `run`: laps run (a red flag can end the race early).
struct Trace {
    n: usize,
    time: Vec<f64>,
    tyre: Vec<f64>,
    event: Vec<i64>,
    held: Vec<f64>,
    run: usize,
}

impl Trace {
    fn new(laps: usize, n: usize) -> Self {
        Trace { n, time: vec![0.0; laps * n], tyre: vec![0.0; laps * n], event: vec![0; laps * n],
                held: vec![0.0; laps * n], run: 0 }
    }

    fn record(&mut self, lap: usize, time: &[f64], tyre: &[f64], event: &[i64], held: &[f64]) {
        let at = lap * self.n;
        self.time[at..at + self.n].copy_from_slice(time);
        self.tyre[at..at + self.n].copy_from_slice(tyre);
        self.event[at..at + self.n].copy_from_slice(event);
        self.held[at..at + self.n].copy_from_slice(held);
        self.run = lap + 1;
    }
}

/// Each car's race time of one draw (s against the start) and its pit stops.
fn race_times(race: &Race, seconds: &[f64], grid: &[f64], rng: &mut Rand, diag: &Diag,
              mut trace: Option<&mut Trace>) -> (Vec<f64>, Vec<usize>) {
    let n = seconds.len();
    let mut time: Vec<f64> = grid
        .iter()
        .enumerate()
        .map(|(car, &slot)| race.start_spacing * (slot - 1.0) + race.start_noise * rng.normal(P_START, 0, car))
        .collect();
    let (mut noise, mut luck) = (vec![0.0; n], vec![0.0; n]);
    let (mut order, mut before, mut after, mut own) =
        (Vec::with_capacity(n), Vec::with_capacity(n), Vec::with_capacity(n), Vec::with_capacity(n));
    let strategy = &race.strategy;
    let pits = strategy.enabled();
    let hazard = pits && strategy.hazard();
    let mut stops = vec![0usize; n];
    let mut pitted = vec![false; n];
    let mut boxed = vec![false; n];
    let mut tyre_laps = vec![0.0f64; n]; // laps run on the set, neutralised ones too (the hazard's clock)
    let (mut event, mut held, mut intended) = (vec![0i64; n], vec![0.0f64; n], vec![0.0f64; n]);
    let plans: Vec<Vec<usize>> = if hazard {
        vec![vec![]; n]
    } else if pits && diag.laps_of.is_some() {
        diag.laps_of.unwrap().to_vec()
    } else if pits {
        match diag.fixed {
            Some(fixed) => (0..n)
                .map(|car| strategy.plan(car, race.laps, fixed[car].max(0) as usize, &race.windows, None, 0, rng))
                .collect(),
            None => {
                let count = strategy.count(rng);
                (0..n)
                    .map(|car| {
                        let own = strategy.own(car, count, rng);
                        let (shift, role) = if race.role_chances.len() == 12 && race.role_shift.len() == 3 * race.windows.len() {
                            // the car's starting tyre role, drawn from its grid band's chances
                            let chance = &race.role_chances[3 * role_band(grid[car])..3 * role_band(grid[car]) + 3];
                            let u = rng.uniform(P_ROLE, 0, car);
                            (Some(race.role_shift.as_slice()), if u < chance[0] { 0 } else if u < chance[0] + chance[1] { 1 } else { 2 })
                        } else {
                            (None, 0)
                        };
                        strategy.plan(car, race.laps, own, &race.windows, shift, role, rng)
                    })
                    .collect()
            }
        }
    } else {
        vec![]
    };
    let mut next = vec![0usize; n];
    let mut age = vec![0.0f64; n];
    // on a set fitted at a pit stop (the fresh-tyre peak applies). The peak's clock is
    // `tyre_laps` (laps on the set, neutralised ones too), as `race_sim.fresh_peak` counts
    // laps into a stint by lap number: the lap after the stop is the out-lap (tyre_laps 0
    // during it; its time is in the stop's loss), peak[k] is on the lap with tyre_laps k + 1
    let mut fresh_set = vec![false; n];
    let wear = if pits { strategy.car_wear(n, rng) } else { vec![] };
    let mut effective = seconds.to_vec();
    let mut neutral_left = 0usize;
    let mut neutral_kind = SC;
    let mut held_gap = vec![f64::NAN; n];
    let mut held_ahead = vec![usize::MAX; n];
    let mut follow_u = vec![0.0f64; if race.follow.is_empty() { 0 } else { n }];
    let mut attack_u = vec![0.0f64; if race.follow.is_empty() { 0 } else { n }];
    for lap in 0..race.laps {
        let lap1 = lap + 1;
        let mut started = false;
        event.iter_mut().for_each(|e| *e = 0);
        held.iter_mut().for_each(|h| *h = 0.0);
        if neutral_left == 0 && race.neutral.hazard[lap] > 0.0 && rng.uniform(P_NEUTRAL, lap, 0) < race.neutral.hazard[lap] {
            let pick = rng.uniform(P_KIND, lap, 0);
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
            neutral_left = if lengths.is_empty() { 1 } else { lengths[rng.index(P_LENGTH, lap, 0, lengths.len())].max(1) };
            if kind == RED && rng.uniform(P_RED, lap, 0) < race.neutral.red_end {
                break; // the race is not restarted: classified as it stands
            }
            if kind == RED && pits {
                // tyres may be changed under the red flag: a new set, no pit visit
                tyre_laps.iter_mut().for_each(|a| *a = 0.0);
                event.iter_mut().for_each(|e| *e = 4);
            }
            if hazard {
                if kind == RED {
                    age.iter_mut().for_each(|a| *a = 0.0);
                    // a set changed under the red flag has no out-lap: not a fresh-peak set (as fitted)
                    fresh_set.iter_mut().for_each(|f| *f = false);
                    stops.iter_mut().for_each(|k| *k += 1);
                } else {
                    let on_lap: Vec<f64> = tyre_laps.iter().map(|a| a + 1.0).collect();
                    boxed = strategy.decide(lap1, race.laps, true, strategy.sc_ratio * strategy.pit_loss, &time, seconds,
                                            &on_lap, &wear, &stops, &pitted, rng);
                    // stopped as the neutralisation begins, before the field closes up (as planned stops)
                    for car in 0..n {
                        if boxed[car] {
                            if stops[car] == 0 && lap1 + PIT_LAST_LAPS == race.laps {
                                event[car] = 3;
                            }
                            time[car] += strategy.sc_ratio * strategy.cost(lap1, car, rng);
                            age[car] = 0.0;
                        fresh_set[car] = true;
                            tyre_laps[car] = 0.0;
                            stops[car] += 1;
                            if event[car] == 0 {
                                event[car] = 2;
                            }
                        }
                    }
                    boxed = vec![false; n];
                }
            } else if pits {
                for car in 0..n {
                    let due = next[car] < plans[car].len() && plans[car][next[car]] <= lap1 + strategy.window;
                    if kind == RED {
                        age[car] = 0.0; // tyres may be changed under the red flag
                        fresh_set[car] = false; // no out-lap: not a fresh-peak set (as fitted)
                        if due {
                            next[car] += 1;
                        }
                    } else if due {
                        time[car] += strategy.sc_ratio * strategy.cost(lap1, car, rng);
                        age[car] = 0.0;
                        fresh_set[car] = true;
                        tyre_laps[car] = 0.0;
                        next[car] += 1;
                        event[car] = 2;
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
            if hazard {
                if !started && neutral_kind != RED {
                    let on_lap: Vec<f64> = tyre_laps.iter().map(|a| a + 1.0).collect();
                    boxed = strategy.decide(lap1, race.laps, true, strategy.sc_ratio * strategy.pit_loss, &time, seconds,
                                            &on_lap, &wear, &stops, &pitted, rng);
                    for car in 0..n {
                        if boxed[car] {
                            if stops[car] == 0 && lap1 + PIT_LAST_LAPS == race.laps {
                                event[car] = 3;
                            }
                            time[car] += strategy.sc_ratio * strategy.cost(lap1, car, rng);
                            age[car] = 0.0;
                        fresh_set[car] = true;
                            tyre_laps[car] = 0.0;
                            stops[car] += 1;
                            if event[car] == 0 {
                                event[car] = 2;
                            }
                        }
                    }
                    if boxed.iter().any(|&b| b) && neutral_kind == SC {
                        close_up(&mut time, &mut order, race.neutral.sc_gap);
                    }
                    boxed = vec![false; n];
                }
                // a car that stopped on this lap starts its new set (its next lap is the set's first)
                for car in 0..n {
                    if event[car] != 2 && event[car] != 3 {
                        tyre_laps[car] += 1.0;
                    }
                }
                pitted = event.iter().map(|&e| e == 2 || e == 3).collect();
            } else {
                if pits && !started && neutral_kind != RED {
                    let mut stopped = false;
                    for car in 0..n {
                        if next[car] < plans[car].len() && plans[car][next[car]] <= lap1 + strategy.window {
                            time[car] += strategy.sc_ratio * strategy.cost(lap1, car, rng);
                            age[car] = 0.0;
                        fresh_set[car] = true;
                            tyre_laps[car] = 0.0;
                            next[car] += 1;
                            stopped = true;
                            event[car] = 2;
                        }
                    }
                    if stopped && neutral_kind == SC {
                        close_up(&mut time, &mut order, race.neutral.sc_gap);
                    }
                }
                for car in 0..n {
                    if event[car] != 2 {
                        tyre_laps[car] += 1.0;
                    }
                }
            }
            neutral_left -= 1; // neutralised: nobody passes, pace does not act
            if let Some(t) = trace.as_deref_mut() {
                t.record(lap, &time, &tyre_laps, &event, &held);
            }
            continue;
        }
        if hazard {
            let on_lap: Vec<f64> = tyre_laps.iter().map(|a| a + 1.0).collect();
            boxed = strategy.decide(lap1, race.laps, false, strategy.pit_loss, &time, seconds, &on_lap, &wear, &stops,
                                    &pitted, rng);
            for car in 0..n {
                if boxed[car] && stops[car] == 0 && lap1 + PIT_LAST_LAPS == race.laps {
                    event[car] = 3;
                }
            }
        }
        for (car, value) in noise.iter_mut().enumerate() {
            *value = rng.normal(P_NOISE, lap, car);
        }
        for (k, value) in luck.iter_mut().enumerate() {
            *value = rng.uniform(P_LUCK, lap, k);
        }
        for (car, value) in follow_u.iter_mut().enumerate() {
            *value = rng.uniform(P_FOLLOW, lap, car);
        }
        for (car, value) in attack_u.iter_mut().enumerate() {
            *value = rng.uniform(P_ATTACK, lap, car);
        }
        if pits {
            for car in 0..n {
                effective[car] = seconds[car] + wear[car] * age[car];
                if fresh_set[car] && tyre_laps[car] >= 1.0 {
                    if let Some(&peak) = race.fresh_peak.get(tyre_laps[car] as usize - 1) {
                        effective[car] -= peak;
                    }
                }
            }
        }
        for car in 0..n {
            intended[car] = time[car] + effective[car] + race.scale * noise[car];
        }
        if diag.free {
            time.copy_from_slice(&intended);
        } else {
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
                &race.follow,
                &follow_u,
                &mut held_gap,
                &mut held_ahead,
                &attack_u,
                race.follow_release,
                &race.follow_penalty,
            );
        }
        for car in 0..n {
            held[car] = time[car] - intended[car];
        }
        if hazard {
            for car in 0..n {
                if boxed[car] {
                    time[car] += strategy.cost(lap1, car, rng);
                    age[car] = 0.0;
                        fresh_set[car] = true;
                    tyre_laps[car] = 0.0;
                    stops[car] += 1;
                    if event[car] == 0 {
                        event[car] = 1;
                    }
                } else {
                    age[car] += 1.0;
                    tyre_laps[car] += 1.0;
                }
            }
            pitted = std::mem::replace(&mut boxed, vec![false; n]);
        } else if pits {
            for car in 0..n {
                if next[car] < plans[car].len() && plans[car][next[car]] <= lap1 {
                    time[car] += strategy.cost(lap1, car, rng); // in and out of the pits: the field goes by
                    age[car] = 0.0;
                        fresh_set[car] = true;
                    tyre_laps[car] = 0.0;
                    next[car] += 1;
                    event[car] = 1;
                } else {
                    age[car] += 1.0;
                    tyre_laps[car] += 1.0;
                }
            }
        }
        if let Some(t) = trace.as_deref_mut() {
            t.record(lap, &time, &tyre_laps, &event, &held);
        }
    }
    (time, if hazard { stops } else { next })
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
    pit_coef: PyReadonlyArray1<f64>,
    passing: f64,
    lap_seconds: f64,
    with_stops: bool,
    trace: bool,
    keyed: bool,
    free_traffic: bool,
    fixed_counts: PyReadonlyArray2<i64>,
    stop_laps: PyReadonlyArray2<i64>,
    follow: PyReadonlyArray1<f64>,
    follow_release: f64,
    follow_penalty: PyReadonlyArray1<f64>,
    fresh_peak: PyReadonlyArray1<f64>,
    windows: PyReadonlyArray1<f64>,
    role_chances: PyReadonlyArray1<f64>,
    role_shift: PyReadonlyArray1<f64>,
) -> PyResult<PyObject> {
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
    let pit_coef = pit_coef.as_slice()?.to_vec();
    if !pit_coef.is_empty() && pit_coef.len() != PIT_FEATURES + 1 {
        return Err(PyValueError::new_err("the pit hazard needs an intercept and one coefficient per feature"));
    }
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
        follow: {
            let table = follow.as_slice()?.to_vec();
            if table.len() == 1 {
                return Err(PyValueError::new_err("following gaps: at least two quantiles"));
            }
            table
        },
        follow_release,
        follow_penalty: {
            let p = follow_penalty.as_slice()?;
            if p.is_empty() { [0.0; 3] } else {
                p.try_into().map_err(|_| PyValueError::new_err("three following penalties: <1 s, 1-2 s, 2-3 s"))?
            }
        },
        fresh_peak: fresh_peak.as_slice()?.to_vec(),
        windows: windows.as_slice()?.to_vec(),
        role_chances: role_chances.as_slice()?.to_vec(),
        role_shift: role_shift.as_slice()?.to_vec(),
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
            pit_coef,
            passing,
            lap_seconds,
        },
    };
    let (seconds, grid) = (seconds.as_slice()?, grid.as_slice()?);
    let fixed = fixed_counts.as_slice()?;
    if !fixed.is_empty() && fixed.len() != draws * n {
        return Err(PyValueError::new_err("fixed stop counts: one per draw and entrant"));
    }
    let laps_of: Vec<Vec<usize>> = stop_laps.as_array().rows().into_iter()
        .map(|row| row.iter().filter(|&&l| l > 0).map(|&l| l as usize).collect()).collect();
    if !laps_of.is_empty() && laps_of.len() != n {
        return Err(PyValueError::new_err("stop laps: one row per entrant"));
    }
    if trace || free_traffic || !fixed.is_empty() || !laps_of.is_empty() {
        // diagnostics: fixed per-car stop counts, no traffic, the lap-by-lap trace
        // (optionally keyed random numbers)
        let runs: Vec<(Vec<i64>, Vec<i64>, Trace)> = py.allow_threads(|| {
            (0..draws)
                .into_par_iter()
                .map(|d| {
                    let draw_seed = seed ^ (d as u64 + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15);
                    let mut generator = Pcg64::seed_from_u64(draw_seed);
                    let mut rng = if keyed { Rand::Keyed(draw_seed) } else { Rand::Seq(&mut generator) };
                    let cars = d * n..(d + 1) * n;
                    let diag = Diag { fixed: if fixed.is_empty() { None } else { Some(&fixed[cars.clone()]) }, free: free_traffic,
                                      laps_of: if laps_of.is_empty() { None } else { Some(&laps_of) } };
                    let mut record = Trace::new(laps, n);
                    let (time, made) = race_times(&race, &seconds[cars.clone()], &grid[cars], &mut rng, &diag, Some(&mut record));
                    let mut positions = vec![0i64; n];
                    rank(&time, &mut positions);
                    (positions, made.iter().map(|&k| k as i64).collect(), record)
                })
                .collect()
        });
        let flat = |f: &dyn Fn(&(Vec<i64>, Vec<i64>, Trace)) -> Vec<i64>| runs.iter().flat_map(f).collect::<Vec<i64>>();
        let array2 = |v: Vec<i64>| numpy::ndarray::Array2::from_shape_vec((draws, n), v)
            .map(|a| a.into_pyarray_bound(py).into_py(py)).map_err(|e| PyValueError::new_err(e.to_string()));
        let array3f = |v: Vec<f64>| numpy::ndarray::Array3::from_shape_vec((draws, laps, n), v)
            .map(|a| a.into_pyarray_bound(py).into_py(py)).map_err(|e| PyValueError::new_err(e.to_string()));
        let array3i = |v: Vec<i64>| numpy::ndarray::Array3::from_shape_vec((draws, laps, n), v)
            .map(|a| a.into_pyarray_bound(py).into_py(py)).map_err(|e| PyValueError::new_err(e.to_string()));
        let out = pyo3::types::PyDict::new_bound(py);
        out.set_item("positions", array2(flat(&|r| r.0.clone()))?)?;
        out.set_item("stops", array2(flat(&|r| r.1.clone()))?)?;
        out.set_item("time", array3f(runs.iter().flat_map(|r| r.2.time.clone()).collect())?)?;
        out.set_item("tyre", array3f(runs.iter().flat_map(|r| r.2.tyre.clone()).collect())?)?;
        out.set_item("event", array3i(runs.iter().flat_map(|r| r.2.event.clone()).collect())?)?;
        out.set_item("held", array3f(runs.iter().flat_map(|r| r.2.held.clone()).collect())?)?;
        out.set_item("run", runs.iter().map(|r| r.2.run as i64).collect::<Vec<i64>>())?;
        return Ok(out.into_py(py));
    }
    let mut positions = vec![0i64; draws * n];
    let mut stops = vec![0i64; draws * n];
    py.allow_threads(|| {
        positions.par_chunks_mut(n).zip(stops.par_chunks_mut(n)).enumerate().for_each(|(d, (out, made))| {
            // one stream per draw, spread from (seed, draw) (no correlated streams). Keyed:
            // each number fixed by (draw, purpose, lap, car), so settings that move a
            // stop leave every other random event of the draw as it was
            let draw_seed = seed ^ (d as u64 + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15);
            let mut generator = Pcg64::seed_from_u64(draw_seed);
            let mut rng = if keyed { Rand::Keyed(draw_seed) } else { Rand::Seq(&mut generator) };
            let cars = d * n..(d + 1) * n;
            race_of_draw(&race, &seconds[cars.clone()], &grid[cars], &mut rng, out, made);
        });
    });
    let array = |v: Vec<i64>| {
        numpy::ndarray::Array2::from_shape_vec((draws, n), v)
            .map(|a| a.into_pyarray_bound(py))
            .map_err(|e| PyValueError::new_err(e.to_string()))
    };
    let positions = array(positions)?;
    Ok(if with_stops { (positions, array(stops)?).into_py(py) } else { positions.into_py(py) })
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
        pit_coef: vec![],
        passing: 0.0,
        lap_seconds: 90.0,
    };
    let mut counts = vec![0i64; draws * entrants];
    for (d, row) in counts.chunks_mut(entrants.max(1)).enumerate() {
        let mut generator = Pcg64::seed_from_u64(seed ^ (d as u64 + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15));
        let mut rng = Rand::Seq(&mut generator);
        let count = strategy.count(&mut rng);
        for (car, out) in row.iter_mut().enumerate() {
            *out = strategy.own(car, count, &mut rng) as i64;
        }
    }
    numpy::ndarray::Array2::from_shape_vec((draws, entrants), counts)
        .map(|a| a.into_pyarray_bound(py))
        .map_err(|e| PyValueError::new_err(e.to_string()))
}

/// The pit hazard's features of one car-lap as the simulator computes them (for
/// checking them against `race_sim.pit_features`).
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn pit_features(age: f64, stops: usize, lap: usize, laps: usize, neutral: bool, position: usize, lost: f64,
                clean_air: f64, gap_ahead: f64, gap_behind: f64, cover_behind: bool, cover_ahead: bool, pace_adv: f64,
                passing: f64, wear: f64) -> Vec<f64> {
    let strategy = Strategy {
        wear: 0.0,
        pit_loss: 1.0,
        sc_ratio: 1.0,
        stops_mean: 1.0,
        stint_noise: 0.0,
        window: 0,
        residuals: vec![],
        deviate: vec![],
        team: vec![],
        team_mean: vec![],
        team_sd: vec![],
        pit_coef: vec![],
        passing,
        lap_seconds: 90.0,
    };
    strategy
        .features(age, stops, lap, laps, neutral, position, lost, clean_air, gap_ahead, gap_behind, cover_behind,
                  cover_ahead, pace_adv, wear)
        .to_vec()
}

/// The pit hazard's features of every car from a race state, through the code the
/// simulator decides with (`Strategy::state_features`): `time`: line-crossing times
/// at the end of the lap before, `age`: tyre age on the deciding lap, `wear`: s per
/// lap of age; one row per car (for checking against `race_sim.pit_decisions`).
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn pit_state_features(lap: usize, laps: usize, neutral: bool, loss: f64, time: Vec<f64>, pace: Vec<f64>, age: Vec<f64>,
                      wear: Vec<f64>, stops: Vec<usize>, pitted: Vec<bool>, passing: f64, lap_seconds: f64)
                      -> Vec<Vec<f64>> {
    let strategy = Strategy {
        wear: 0.0, pit_loss: loss, sc_ratio: 1.0, stops_mean: 1.0, stint_noise: 0.0, window: 0, residuals: vec![],
        deviate: vec![], team: vec![], team_mean: vec![], team_sd: vec![], pit_coef: vec![], passing, lap_seconds,
    };
    strategy
        .state_features(lap, laps, neutral, loss, &time, &pace, &age, &wear, &stops, &pitted)
        .into_iter()
        .map(|row| row.to_vec())
        .collect()
}

#[pymodule]
fn predictor_race_native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(simulate, m)?)?;
    m.add_function(wrap_pyfunction!(stop_counts, m)?)?;
    m.add_function(wrap_pyfunction!(pit_features, m)?)?;
    m.add_function(wrap_pyfunction!(pit_state_features, m)?)?;
    m.add_function(wrap_pyfunction!(pass_probability, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// One car, one planned stop, no neutralisation: its race time is the stop's cost
    /// less the fresh-tyre peak on the set fitted at the stop (never the starting set).
    fn fresh_car(peak: Vec<f64>) -> f64 {
        let laps = 20;
        let race = Race {
            coef: coefficients(&[-0.8, -1.9, -0.3, -1.6, 0.2, 0.7]).unwrap(),
            term: 0.0, shift: 0.0, laps, start_noise: 0.0, start_spacing: 0.25, scale: 0.0, battle_gap: 2.5,
            min_follow: 0.35, follow: vec![], follow_release: 0.0, follow_penalty: [0.0; 3], fresh_peak: peak, windows: vec![], role_chances: vec![], role_shift: vec![],
            neutral: Neutral { hazard: vec![0.0; laps], kinds: [1.0, 0.0, 0.0], lengths: [vec![1], vec![1], vec![1]],
                               sc_gap: 0.52, red_end: 0.0 },
            strategy: Strategy { wear: 0.0, pit_loss: 22.0, sc_ratio: 0.5, stops_mean: 1.0, stint_noise: 0.0, window: 6,
                                 residuals: vec![], deviate: vec![], team: vec![], team_mean: vec![], team_sd: vec![],
                                 pit_coef: vec![], passing: 0.0, lap_seconds: 90.0 },
        };
        race_times(&race, &[0.0], &[1.0], &mut Rand::Seq(&mut Pcg64::seed_from_u64(7)), &Diag::default(), None).0[0]
    }

    /// Per-lap times of `fresh_car`'s race (lap k's time = trace time at k - at k-1).
    fn fresh_laps(peak: Vec<f64>, hazard_lap: Option<usize>) -> Vec<f64> {
        let laps = 20;
        let mut hazard = vec![0.0; laps];
        if let Some(l) = hazard_lap {
            hazard[l - 1] = 1.0;
        }
        let race = Race {
            coef: coefficients(&[-0.8, -1.9, -0.3, -1.6, 0.2, 0.7]).unwrap(),
            term: 0.0, shift: 0.0, laps, start_noise: 0.0, start_spacing: 0.25, scale: 0.0, battle_gap: 2.5,
            min_follow: 0.35, follow: vec![], follow_release: 0.0, follow_penalty: [0.0; 3], fresh_peak: peak, windows: vec![], role_chances: vec![], role_shift: vec![],
            neutral: Neutral { hazard, kinds: [0.0, 1.0, 0.0], lengths: [vec![1], vec![2], vec![1]],
                               sc_gap: 0.52, red_end: 0.0 },
            strategy: Strategy { wear: 0.0, pit_loss: 22.0, sc_ratio: 1.0, stops_mean: 1.0, stint_noise: 0.0, window: 0,
                                 residuals: vec![], deviate: vec![], team: vec![], team_mean: vec![], team_sd: vec![],
                                 pit_coef: vec![], passing: 0.0, lap_seconds: 90.0 },
        };
        let mut trace = Trace::new(laps, 1);
        race_times(&race, &[0.0], &[1.0], &mut Rand::Seq(&mut Pcg64::seed_from_u64(7)), &Diag::default(),
                   Some(&mut trace));
        (0..laps).map(|k| trace.time[k] - if k > 0 { trace.time[k - 1] } else { 0.0 }).collect()
    }

    #[test]
    fn fresh_tyre_peak_starts_on_the_first_flying_lap_after_the_out_lap() {
        // one stop planned at the end of lap 10 (20 laps): lap 10 carries the stop, lap 11
        // is the out-lap, the peak is on laps 12, 13, 14 (as fitted: laps 2-4 into the stint)
        let (base, peak) = (fresh_laps(vec![], None), fresh_laps(vec![0.5, 0.3, 0.1], None));
        let gain: Vec<f64> = base.iter().zip(&peak).map(|(b, p)| b - p).collect();
        for (k, g) in gain.iter().enumerate() {
            let want = match k + 1 { 12 => 0.5, 13 => 0.3, 14 => 0.1, _ => 0.0 };
            assert!((g - want).abs() < 1e-9, "lap {}: gain {g}, want {want}", k + 1);
        }
        // a virtual safety car on laps 12-13 advances the set's clock (as lap numbers do in
        // the fit): only lap 14's 0.1 s is left on a green lap
        let (base, peak) = (fresh_laps(vec![], Some(12)), fresh_laps(vec![0.5, 0.3, 0.1], Some(12)));
        let total: f64 = base.iter().zip(&peak).map(|(b, p)| b - p).sum();
        assert!((total - 0.1).abs() < 1e-9, "{total}");
    }

    #[test]
    fn learned_windows_place_the_stops() {
        let strategy = Strategy { wear: 0.0, pit_loss: 22.0, sc_ratio: 0.5, stops_mean: 1.0, stint_noise: 0.0, window: 6,
                                  residuals: vec![], deviate: vec![], team: vec![], team_mean: vec![], team_sd: vec![],
                                  pit_coef: vec![], passing: 0.0, lap_seconds: 90.0 };
        let mut g = Pcg64::seed_from_u64(1);
        let windows = [0.42, 0.26, 0.63];
        assert_eq!(strategy.plan(0, 50, 1, &windows, None, 0, &mut Rand::Seq(&mut g)), vec![21]);
        assert_eq!(strategy.plan(0, 50, 2, &windows, None, 0, &mut Rand::Seq(&mut g)), vec![13, 32]);
        // three stops: not learned here, evenly spread
        assert_eq!(strategy.plan(0, 50, 3, &windows, None, 0, &mut Rand::Seq(&mut g)), vec![13, 25, 38]);
        assert_eq!(strategy.plan(0, 50, 1, &[], None, 0, &mut Rand::Seq(&mut g)), vec![25]);
        // starting tyre roles: the hard role's shift moves a one-stop car's stop from 21 to 29
        let shift = [-0.03, 0.0, 0.16, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        assert_eq!(strategy.plan(0, 50, 1, &windows, Some(&shift), 2, &mut Rand::Seq(&mut g)), vec![29]);
        assert_eq!(strategy.plan(0, 50, 1, &windows, Some(&shift), 0, &mut Rand::Seq(&mut g)), vec![20]);
    }

    #[test]
    fn fresh_tyre_peak_applies_only_after_a_stop() {
        let base = fresh_car(vec![]);
        assert!((base - 22.0).abs() < 1e-9, "{base}");
        // the first two green laps on the new set are 0.5 s and 0.3 s faster; the starting set gets nothing
        assert!((fresh_car(vec![0.5, 0.3]) - (22.0 - 0.8)).abs() < 1e-9);
    }

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
            follow: vec![],
            follow_release: 0.0,
            follow_penalty: [0.0; 3], fresh_peak: vec![], windows: vec![], role_chances: vec![], role_shift: vec![],
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
                pit_coef: vec![],
                passing: 0.0,
                lap_seconds: 90.0,
            },
        };
        race_times(&race, &[0.0], &[1.0], &mut Rand::Seq(&mut Pcg64::seed_from_u64(7)), &Diag::default(), None).0[0]
    }

    #[test]
    fn a_car_the_hazard_never_calls_in_still_stops_once() {
        let mut hazard = vec![0.0; PIT_FEATURES + 1];
        hazard[0] = -50.0;
        let race = Race {
            coef: coefficients(&[-0.8, -1.9, -0.3, -1.6, 0.2, 0.7]).unwrap(),
            term: 0.0, shift: 0.0, laps: 20, start_noise: 0.0, start_spacing: 0.25, scale: 0.0, battle_gap: 2.5,
            min_follow: 0.35,
            follow: vec![],
            follow_release: 0.0,
            follow_penalty: [0.0; 3], fresh_peak: vec![], windows: vec![], role_chances: vec![], role_shift: vec![],
            neutral: Neutral { hazard: vec![0.0; 20], kinds: [1.0, 0.0, 0.0], lengths: [vec![1], vec![1], vec![1]],
                               sc_gap: 0.52, red_end: 0.0 },
            strategy: Strategy { wear: 0.0, pit_loss: 22.0, sc_ratio: 0.5, stops_mean: 1.0, stint_noise: 0.0, window: 6,
                                 residuals: vec![], deviate: vec![], team: vec![], team_mean: vec![], team_sd: vec![],
                                 pit_coef: hazard.clone(), passing: 0.0, lap_seconds: 90.0 },
        };
        assert!((race_times(&race, &[0.0], &[1.0], &mut Rand::Seq(&mut Pcg64::seed_from_u64(3)), &Diag::default(), None).0[0] - 22.0).abs() < 1e-9);
        let mut eager = race;
        eager.strategy.pit_coef[0] = 50.0; // pits at every lap's end from lap 2 to 3 laps from the end
        let stops = (PIT_FIRST_LAP..=20 - PIT_LAST_LAPS).count() as f64;
        assert!((race_times(&eager, &[0.0], &[1.0], &mut Rand::Seq(&mut Pcg64::seed_from_u64(3)), &Diag::default(), None).0[0] - 22.0 * stops).abs() < 1e-9);
    }

    #[test]
    fn a_hazard_on_tyre_age_stops_on_the_laps_training_would_label() {
        // stop whenever the tyres are older than five laps on the lap (TyreLife > 5):
        // laps 6, 12 and 18 of 22 (a stop's next lap is the set's first)
        let mut hazard = vec![0.0; PIT_FEATURES + 1];
        hazard[0] = -50.0;
        for k in 0..7 {
            hazard[1 + k] = 100.0;
        }
        let race = Race {
            coef: coefficients(&[-0.8, -1.9, -0.3, -1.6, 0.2, 0.7]).unwrap(),
            term: 0.0, shift: 0.0, laps: 22, start_noise: 0.0, start_spacing: 0.25, scale: 0.0, battle_gap: 2.5,
            min_follow: 0.35,
            follow: vec![],
            follow_release: 0.0,
            follow_penalty: [0.0; 3], fresh_peak: vec![], windows: vec![], role_chances: vec![], role_shift: vec![],
            neutral: Neutral { hazard: vec![0.0; 22], kinds: [1.0, 0.0, 0.0], lengths: [vec![1], vec![1], vec![1]],
                               sc_gap: 0.52, red_end: 0.0 },
            strategy: Strategy { wear: 0.0, pit_loss: 22.0, sc_ratio: 0.5, stops_mean: 1.0, stint_noise: 0.0, window: 6,
                                 residuals: vec![], deviate: vec![], team: vec![], team_mean: vec![], team_sd: vec![],
                                 pit_coef: hazard, passing: 0.0, lap_seconds: 90.0 },
        };
        let (time, stops) = race_times(&race, &[0.0], &[1.0], &mut Rand::Seq(&mut Pcg64::seed_from_u64(3)), &Diag::default(), None);
        assert_eq!(stops[0], 3);
        assert!((time[0] - 66.0).abs() < 1e-9);
        // a safety car on laps 3-5 still ages the tyres: the first stop stays on lap 6
        let mut neutral = race;
        neutral.neutral.hazard[2] = 1.0;
        neutral.neutral.lengths = [vec![3], vec![1], vec![1]];
        let (_, stops) = race_times(&neutral, &[0.0], &[1.0], &mut Rand::Seq(&mut Pcg64::seed_from_u64(3)), &Diag::default(), None);
        assert_eq!(stops[0], 3);
    }

    /// Two cars, the leader 50 s ahead; a safety car from lap 5. Only the leader
    /// stops as it comes out: `learned` by a hazard calling in the leader on a
    /// neutralised lap, else by its plan (one stop, due within the window). The
    /// order at the end of lap 5.
    fn safety_car_stop(learned: bool) -> Vec<i64> {
        let mut neutral_hazard = vec![0.0; 20];
        neutral_hazard[4] = 1.0;
        let mut hazard = vec![0.0; PIT_FEATURES + 1];
        hazard[0] = -50.0;
        hazard[1 + 11] = 100.0; // neutralised lap
        hazard[1 + 13] = -700.0; // position / 20: the leader stops (z = +15), the other car not (z = -20)
        let race = Race {
            coef: coefficients(&[-50.0, 0.0, 0.0, 0.0, 0.0, 0.0]).unwrap(),
            term: 0.0, shift: 0.0, laps: 20, start_noise: 0.0, start_spacing: 0.25, scale: 0.0, battle_gap: 2.5,
            min_follow: 0.35,
            follow: vec![],
            follow_release: 0.0,
            follow_penalty: [0.0; 3], fresh_peak: vec![], windows: vec![], role_chances: vec![], role_shift: vec![],
            neutral: Neutral { hazard: neutral_hazard, kinds: [1.0, 0.0, 0.0], lengths: [vec![3], vec![1], vec![1]],
                               sc_gap: 0.52, red_end: 0.0 },
            strategy: Strategy { wear: 0.0, pit_loss: 22.0, sc_ratio: 0.8, stops_mean: 1.0, stint_noise: 0.0, window: 6,
                                 residuals: vec![], deviate: vec![], team: vec![], team_mean: vec![], team_sd: vec![],
                                 pit_coef: if learned { hazard } else { vec![] }, passing: 0.0, lap_seconds: 90.0 },
        };
        let grid = [1.0, 1.0 + 50.0 / 0.25]; // the second car starts 50 s behind
        let fixed = [1i64, 0];
        let diag = Diag { fixed: if learned { None } else { Some(&fixed) }, free: false, laps_of: None };
        let mut trace = Trace::new(20, 2);
        race_times(&race, &[0.0, 0.0], &grid, &mut Rand::Seq(&mut Pcg64::seed_from_u64(1)), &diag, Some(&mut trace));
        assert!(trace.event[4 * 2] == 2 && trace.event[4 * 2 + 1] == 0, "only the leader stops, on lap 5");
        let mut positions = vec![0i64; 2];
        rank(&trace.time[4 * 2..4 * 2 + 2], &mut positions);
        positions
    }

    #[test]
    fn a_stop_as_the_safety_car_comes_out_is_charged_alike_for_planned_and_learned_stops() {
        // charged before the field closes up: 50 - 17.6 s ahead, the leader keeps the lead
        assert_eq!(safety_car_stop(false), vec![1, 2]);
        assert_eq!(safety_car_stop(true), vec![1, 2]);
    }

    #[test]
    fn a_held_up_car_keeps_its_gap_lap_after_lap() {
        // a faster follower that cannot pass, no lap noise, a 2 s following gap
        let race = Race {
            coef: coefficients(&[-50.0, 0.0, 0.0, 0.0, 0.0, 0.0]).unwrap(),
            term: 0.0, shift: 0.0, laps: 20, start_noise: 0.0, start_spacing: 0.25, scale: 0.0, battle_gap: 2.5,
            min_follow: 0.35,
            follow: vec![2.0, 2.0],
            follow_release: 0.0,
            follow_penalty: [0.0; 3], fresh_peak: vec![], windows: vec![], role_chances: vec![], role_shift: vec![],
            neutral: Neutral { hazard: vec![0.0; 20], kinds: [1.0, 0.0, 0.0], lengths: [vec![1], vec![1], vec![1]],
                               sc_gap: 0.52, red_end: 0.0 },
            strategy: Strategy { wear: 0.0, pit_loss: 0.0, sc_ratio: 1.0, stops_mean: 1.0, stint_noise: 0.0, window: 0,
                                 residuals: vec![], deviate: vec![], team: vec![], team_mean: vec![], team_sd: vec![],
                                 pit_coef: vec![], passing: 0.0, lap_seconds: 90.0 },
        };
        let mut trace = Trace::new(20, 2);
        race_times(&race, &[0.0, -0.5], &[1.0, 2.0], &mut Rand::Seq(&mut Pcg64::seed_from_u64(1)), &Diag::default(),
                   Some(&mut trace));
        let gaps: Vec<f64> = (0..20).map(|lap| trace.time[lap * 2 + 1] - trace.time[lap * 2]).collect();
        // it starts 0.25 s behind, is held from lap 1 at 2 s and stays there
        assert!(gaps.iter().all(|g| (g - 2.0).abs() < 1e-9), "{gaps:?}");
    }

    /// A follower 3 s a lap faster, passing certain inside the battle range, a 3 s
    /// following gap (beyond the battle range), starting 3 s behind: the finishing order.
    fn attacker(release: f64) -> Vec<i64> {
        let race = Race {
            coef: coefficients(&[50.0, 0.0, 0.0, 0.0, 0.0, 0.0]).unwrap(),
            term: 0.0, shift: 0.0, laps: 10, start_noise: 0.0, start_spacing: 0.25, scale: 0.0, battle_gap: 2.5,
            min_follow: 0.35, follow: vec![3.0, 3.0], follow_release: release, follow_penalty: [0.0; 3], fresh_peak: vec![], windows: vec![], role_chances: vec![], role_shift: vec![],
            neutral: Neutral { hazard: vec![0.0; 10], kinds: [1.0, 0.0, 0.0], lengths: [vec![1], vec![1], vec![1]],
                               sc_gap: 0.52, red_end: 0.0 },
            strategy: Strategy { wear: 0.0, pit_loss: 0.0, sc_ratio: 1.0, stops_mean: 1.0, stint_noise: 0.0, window: 0,
                                 residuals: vec![], deviate: vec![], team: vec![], team_mean: vec![], team_sd: vec![],
                                 pit_coef: vec![], passing: 0.0, lap_seconds: 90.0 },
        };
        let (time, _) = race_times(&race, &[0.0, -3.0], &[1.0, 13.0], &mut Rand::Seq(&mut Pcg64::seed_from_u64(1)),
                                   &Diag::default(), None);
        let mut positions = vec![0i64; 2];
        rank(&time, &mut positions);
        positions
    }

    #[test]
    fn a_held_car_can_leave_its_gap_and_attack() {
        // a target beyond the battle range locks the follower out unless it can attack
        assert_eq!(attacker(0.0), vec![1, 2]);
        assert_eq!(attacker(0.5), vec![2, 1]);
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
            pit_coef: vec![],
            passing: 0.0,
            lap_seconds: 90.0,
        };
        let mut rng = Pcg64::seed_from_u64(1);
        assert_eq!(strategy.car_wear(3, &mut Rand::Seq(&mut rng)), vec![0.05; 3]);
        strategy.team = vec![0, 1, 0];
        strategy.team_mean = vec![0.01, -0.02];
        strategy.team_sd = vec![0.0, 0.0];
        let wear = strategy.car_wear(3, &mut Rand::Seq(&mut rng));
        assert!((wear[0] - 0.06).abs() < 1e-12 && (wear[1] - 0.03).abs() < 1e-12 && wear[0] == wear[2]);
        strategy.team_sd = vec![0.01, 0.01];
        let drawn = strategy.car_wear(3, &mut Rand::Seq(&mut rng));
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

//! Compare the race regression's variants with an external per-team reference
//! table: run every variant on an event and rank the variants by their agreement.
//!
//! Reference tables are read from files given at run time; no reference values
//! live in this code.

use std::collections::BTreeMap;
use std::path::Path;

use anyhow::{bail, Context, Result};

use crate::archive::Archive;
use crate::pace::{self, Agg, Compounds, LapTrend, RaceFilter, RaceModel, TeamTimes};

/// What a reference gap is measured in.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Unit {
    /// Percent of the quickest team's lap.
    Percent,
    /// Seconds per lap.
    Seconds,
}

/// A reference table: team -> gap to the quickest team.
pub struct Reference {
    pub unit: Unit,
    pub gaps: BTreeMap<String, f64>,
}

/// `team,gap` per line, optionally preceded by `unit,s` or `unit,pct` (default pct);
/// `#` starts a comment.
pub fn read_reference(path: &Path) -> Result<Reference> {
    let text = std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    let mut unit = Unit::Percent;
    let mut gaps = BTreeMap::new();
    for line in text.lines().map(str::trim).filter(|l| !l.is_empty() && !l.starts_with('#')) {
        let (key, value) = line.rsplit_once(',').context("expected team,gap")?;
        match key.trim() {
            "team" => continue,
            "unit" => {
                unit = match value.trim() {
                    "s" => Unit::Seconds,
                    "pct" | "%" => Unit::Percent,
                    other => bail!("unknown unit {other}"),
                }
            }
            team => {
                gaps.insert(team.to_string(), value.trim().parse().with_context(|| format!("bad value in {line}"))?);
            }
        }
    }
    Ok(Reference { unit, gaps })
}

/// One variant's representative lap time per team.
pub struct Candidate {
    pub name: String,
    pub times: TeamTimes,
}

impl Candidate {
    pub fn gaps(&self, unit: Unit) -> BTreeMap<String, f64> {
        match unit {
            Unit::Percent => pace::gaps_pct(&self.times),
            Unit::Seconds => pace::gaps_s(&self.times),
        }
    }
}

#[derive(Debug, Clone)]
pub struct Score {
    pub mae: f64,
    pub max_err: f64,
    pub spearman: f64,
    pub same_best: bool,
    pub missing: usize,
    /// Least-squares factor taking our gaps to the reference ones (1 = same scale).
    pub scale: f64,
}

pub fn score(candidate: &Candidate, reference: &Reference) -> Score {
    let gaps = candidate.gaps(reference.unit);
    let pairs: Vec<(f64, f64)> =
        reference.gaps.iter().filter_map(|(team, r)| gaps.get(team).map(|g| (*g, *r))).collect();
    let missing = reference.gaps.len() - pairs.len();
    let errors: Vec<f64> = pairs.iter().map(|(g, r)| (g - r).abs()).collect();
    let mae = errors.iter().sum::<f64>() / errors.len().max(1) as f64;
    let max_err = errors.iter().copied().fold(0.0, f64::max);
    let ref_best = reference.gaps.iter().min_by(|a, b| a.1.total_cmp(b.1)).map(|(t, _)| t);
    let our_best = gaps.iter().min_by(|a, b| a.1.total_cmp(b.1)).map(|(t, _)| t);
    let (num, den) = pairs.iter().fold((0.0, 0.0), |(n, d), (g, r)| (n + g * r, d + g * g));
    let scale = if den > 0.0 { num / den } else { f64::NAN };
    Score { mae, max_err, spearman: spearman(&pairs), same_best: ref_best == our_best, missing, scale }
}

fn ranks(values: &[f64]) -> Vec<f64> {
    let mut idx: Vec<usize> = (0..values.len()).collect();
    idx.sort_by(|&a, &b| values[a].total_cmp(&values[b]));
    let mut r = vec![0.0; values.len()];
    let mut i = 0;
    while i < idx.len() {
        let mut j = i;
        while j + 1 < idx.len() && values[idx[j + 1]] == values[idx[i]] {
            j += 1;
        }
        let avg = (i + j) as f64 / 2.0 + 1.0;
        for &k in &idx[i..=j] {
            r[k] = avg;
        }
        i = j + 1;
    }
    r
}

fn spearman(pairs: &[(f64, f64)]) -> f64 {
    let n = pairs.len();
    if n < 2 {
        return f64::NAN;
    }
    let a = ranks(&pairs.iter().map(|p| p.0).collect::<Vec<_>>());
    let b = ranks(&pairs.iter().map(|p| p.1).collect::<Vec<_>>());
    let mean = (n as f64 + 1.0) / 2.0;
    let (mut num, mut da, mut db) = (0.0, 0.0, 0.0);
    for i in 0..n {
        num += (a[i] - mean) * (b[i] - mean);
        da += (a[i] - mean).powi(2);
        db += (b[i] - mean).powi(2);
    }
    num / (da * db).sqrt()
}

/// Every regression variant for one event's race.
pub fn candidates(archive: &Archive, event_key: &str, min_laps: usize) -> Result<Vec<Candidate>> {
    let Some(race) = archive.session_of_type(event_key, "R")? else { bail!("{event_key} has no race") };
    let laps = archive.laps(&race.session_id)?;
    let mut out = Vec::new();
    for (compounds, cname) in [(Compounds::All, "all"), (Compounds::Slicks, "slick")] {
        for min_gap_s in [None, Some(1.0), Some(2.0)] {
            for skip_after_neutral in [0, 1, 2] {
                let filter = RaceFilter { compounds, min_gap_s, skip_after_neutral, ..RaceFilter::default() };
                let gname = min_gap_s.map_or("anygap".to_string(), |g| format!("gap>{g}"));
                let rname = if skip_after_neutral > 0 { format!(",restart{skip_after_neutral}") } else { String::new() };
                for team_wear in [false, true] {
                    for (lap_trend, tname) in
                        [(LapTrend::Linear, ""), (LapTrend::LinearPerCompound, "+lapxcomp"), (LapTrend::PerLap, "+perlap")]
                    {
                        for trim_sigma in [None, Some(3.0)] {
                            for traffic_term in [false, true] {
                                if traffic_term && min_gap_s.is_some() {
                                    continue;
                                }
                                let model = RaceModel { team_wear, lap_trend, trim_sigma, traffic_term, recheck_min_laps: false };
                                let mname = format!(
                                    "fit{}{tname}{}{}",
                                    if team_wear { "+teamwear" } else { "" },
                                    if trim_sigma.is_some() { "+trim3" } else { "" },
                                    if traffic_term { "+traffic" } else { "" },
                                );
                                for (agg, aname) in [(Agg::Best, "best"), (Agg::Mean, "mean2"), (Agg::LapBest, "lapbest")] {
                                    out.push(Candidate {
                                        name: format!("{mname},{cname},{gname}{rname},{aname}"),
                                        times: pace::race_model_team_times(&laps, &filter, &model, agg, min_laps),
                                    });
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    Ok(out)
}

//! Reproduce a published per-team pace table: run every candidate recipe on one
//! event and rank the recipes by how closely they match the published numbers.
//!
//! The published table is read from a file given at run time; no reference values
//! live in this code.

use std::collections::BTreeMap;
use std::path::Path;

use anyhow::{bail, Context, Result};

use crate::archive::Archive;
use crate::pace::{self, Agg, Compounds, LapBase, RaceFilter, RaceModel, Stat, TeamTimes};

/// Team -> published percentage gap to the quickest team.
pub type Reference = BTreeMap<String, f64>;

/// `team,pct` per line; `#` starts a comment.
pub fn read_reference(path: &Path) -> Result<Reference> {
    let text = std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    let mut out = Reference::new();
    for line in text.lines().map(str::trim).filter(|l| !l.is_empty() && !l.starts_with('#')) {
        let (team, pct) = line.rsplit_once(',').context("expected team,pct")?;
        if team == "team" {
            continue;
        }
        out.insert(team.trim().to_string(), pct.trim().parse().with_context(|| format!("bad value in {line}"))?);
    }
    Ok(out)
}

pub struct Candidate {
    pub name: String,
    pub gaps: BTreeMap<String, f64>,
}

#[derive(Debug, Clone)]
pub struct Score {
    pub mae: f64,
    pub max_err: f64,
    pub spearman: f64,
    pub same_best: bool,
    pub missing: usize,
    /// Least-squares factor taking our gaps to the published ones (1 = same scale).
    pub scale: f64,
}

pub fn score(gaps: &BTreeMap<String, f64>, reference: &Reference) -> Score {
    let pairs: Vec<(f64, f64)> =
        reference.iter().filter_map(|(team, r)| gaps.get(team).map(|g| (*g, *r))).collect();
    let missing = reference.len() - pairs.len();
    let errors: Vec<f64> = pairs.iter().map(|(g, r)| (g - r).abs()).collect();
    let mae = errors.iter().sum::<f64>() / errors.len().max(1) as f64;
    let max_err = errors.iter().copied().fold(0.0, f64::max);
    let ref_best = reference.iter().min_by(|a, b| a.1.total_cmp(b.1)).map(|(t, _)| t);
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
        for k in i..=j {
            r[idx[k]] = avg;
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

fn agg_name(a: Agg) -> &'static str {
    match a {
        Agg::Best => "best",
        Agg::Mean => "mean2",
    }
}

fn filter_name(f: &RaceFilter) -> String {
    let c = match f.compounds {
        Compounds::All => "all",
        Compounds::Slicks => "slick",
        Compounds::Inters => "inter",
    };
    let gap = f.min_gap_s.map_or("anygap".to_string(), |g| format!("gap>{g}"));
    format!("{c},{gap}{}", if f.accurate_only { ",acc" } else { "" })
}

/// Every recipe for one event: qualifying, race and 50/50 blends.
pub fn candidates(archive: &Archive, event_key: &str, min_laps: usize) -> Result<Vec<Candidate>> {
    let Some(quali) = archive.session_of_type(event_key, "Q")? else { bail!("{event_key} has no qualifying") };
    let Some(race) = archive.session_of_type(event_key, "R")? else { bail!("{event_key} has no race") };
    let q_laps = archive.laps(&quali.session_id)?;
    let r_laps = archive.laps(&race.session_id)?;

    let mut quali_methods: Vec<(String, TeamTimes)> = Vec::new();
    for agg in [Agg::Best, Agg::Mean] {
        quali_methods.push((format!("Q[{}]", agg_name(agg)), pace::one_lap_team_times(&q_laps, agg)));
    }

    let mut race_methods: Vec<(String, TeamTimes)> = Vec::new();
    for compounds in [Compounds::All, Compounds::Slicks, Compounds::Inters] {
        for min_gap_s in [None, Some(1.0), Some(2.0)] {
            for accurate_only in [true, false] {
                let filter = RaceFilter { compounds, min_gap_s, skip_laps: 1, accurate_only, fuel_s_per_lap: 0.0 };
                let fname = filter_name(&filter);
                for agg in [Agg::Best, Agg::Mean] {
                    for fuel in [0.0, 0.03, 0.06] {
                        let fuelled = RaceFilter { fuel_s_per_lap: fuel, ..filter };
                        let fuel_name = if fuel > 0.0 { format!(",fuel{fuel}") } else { String::new() };
                        for (stat, sname) in [
                            (Stat::Median, "median".to_string()),
                            (Stat::Mean, "mean".to_string()),
                            (Stat::Fastest, "fastest".to_string()),
                            (Stat::TopMean(3), "top3".to_string()),
                            (Stat::TopMean(5), "top5".to_string()),
                            (Stat::TopMean(10), "top10".to_string()),
                            (Stat::TopMean(20), "top20".to_string()),
                        ] {
                            let t = pace::race_stat_team_times(&r_laps, &fuelled, stat, agg, min_laps);
                            race_methods.push((format!("R[{sname}{fuel_name},{fname},{}]", agg_name(agg)), t));
                        }
                    }
                    for (base, bname) in [(LapBase::Fastest, "lapbest"), (LapBase::Median, "lapmedian")] {
                        for (stat, sname) in [(Stat::Median, "median"), (Stat::Mean, "mean")] {
                            let t = pace::race_per_lap_team_times(&r_laps, &filter, base, stat, agg, min_laps, 5);
                            race_methods.push((format!("R[perlap:{bname}:{sname},{fname},{}]", agg_name(agg)), t));
                        }
                    }
                    for team_wear in [false, true] {
                        for lap_slope_per_compound in [false, true] {
                            for trim_sigma in [None, Some(3.0)] {
                                for traffic_term in [false, true] {
                                    if traffic_term && min_gap_s.is_some() {
                                        continue;
                                    }
                                    let model = RaceModel { team_wear, lap_slope_per_compound, trim_sigma, traffic_term };
                                    let t = pace::race_model_team_times(&r_laps, &filter, &model, agg, min_laps);
                                    let mname = format!(
                                        "fit{}{}{}{}",
                                        if team_wear { "+teamwear" } else { "" },
                                        if lap_slope_per_compound { "+lapxcomp" } else { "" },
                                        if trim_sigma.is_some() { "+trim3" } else { "" },
                                        if traffic_term { "+traffic" } else { "" },
                                    );
                                    race_methods.push((format!("R[{mname},{fname},{}]", agg_name(agg)), t));
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    let mut out: Vec<Candidate> = Vec::new();
    for (name, t) in quali_methods.iter().chain(race_methods.iter()) {
        out.push(Candidate { name: name.clone(), gaps: pace::gaps_pct(t) });
    }
    for (qn, qt) in &quali_methods {
        let qg = pace::gaps_pct(qt);
        for (rn, rt) in &race_methods {
            let rg = pace::gaps_pct(rt);
            // Blend the gaps, then re-base so the quickest blended team is at zero.
            let mut blended: BTreeMap<String, f64> =
                qg.iter().filter_map(|(team, q)| rg.get(team).map(|r| (team.clone(), (q + r) / 2.0))).collect();
            let best = blended.values().copied().fold(f64::INFINITY, f64::min);
            blended.values_mut().for_each(|v| *v -= best);
            out.push(Candidate { name: format!("{qn}+{rn}"), gaps: blended });
        }
    }
    Ok(out)
}

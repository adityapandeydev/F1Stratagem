use std::path::PathBuf;

use anyhow::Result;
use clap::{Parser, Subcommand};

use analysis::archive::Archive;
use analysis::calibrate;
use analysis::car_pace;
use analysis::pace::{self, Agg, Compounds, LapTrend, RacePace, Recipe};
use analysis::season::{self, SeasonTable};

#[derive(Parser)]
#[command(name = "analysis", about = "F1Stratagem analysis service")]
struct Cli {
    /// Archive root (default: F1_DATA_ROOT, else the repository's data/).
    #[arg(long, global = true)]
    data_root: Option<PathBuf>,
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Rank team-pace recipes by their agreement with reference tables
    /// (one or more events; recipes are ranked by mean error over all of them).
    Calibrate {
        /// Event key, e.g. 2024-20 (repeat for several events).
        #[arg(long, required = true)]
        event: Vec<String>,
        /// CSV of team,gap for each event, in the same order.
        #[arg(long, required = true)]
        reference: Vec<PathBuf>,
        /// Recipes to list.
        #[arg(long, default_value_t = 25)]
        top: usize,
        /// Recipes to show team by team.
        #[arg(long, default_value_t = 1)]
        detail: usize,
        /// Drivers with fewer kept race laps are left out.
        #[arg(long, default_value_t = 5)]
        min_laps: usize,
        /// Only recipes whose name contains this text.
        #[arg(long)]
        filter: Option<String>,
    },
    /// Team race pace of one race: teams and drivers with intervals and evidence.
    Race {
        /// Event key, e.g. 2024-20.
        #[arg(long)]
        event: String,
        #[command(flatten)]
        recipe: RecipeArgs,
        /// Stint resamples for the intervals (0 = none).
        #[arg(long, default_value_t = 200)]
        resamples: usize,
        /// Also write the full result as JSON to this file.
        #[arg(long)]
        json: Option<PathBuf>,
    },
    /// Team race pace of every race of a season, with the season summaries.
    Season {
        #[arg(long)]
        season: i64,
        #[command(flatten)]
        recipe: RecipeArgs,
        /// Stint resamples per race for the intervals (0 = none).
        #[arg(long, default_value_t = 100)]
        resamples: usize,
        /// Also write the full result as JSON to this file.
        #[arg(long)]
        json: Option<PathBuf>,
    },
    /// Car-only race pace of every race of a season: driver skills removed.
    CarSeason {
        #[arg(long)]
        season: i64,
        /// Also write the full result as JSON to this file.
        #[arg(long)]
        json: Option<PathBuf>,
        /// Also print the internal driver skill table of the window (not published).
        #[arg(long)]
        show_skills: bool,
        /// Seasons of the driver-skill window (the season and those before it).
        #[arg(long, default_value_t = car_pace::WINDOW_SEASONS)]
        window: i64,
        /// Also use sprint races as teammate comparisons for driver skills.
        #[arg(long)]
        with_sprints: bool,
    },
    /// Car-only race pace of one race: driver skills removed.
    CarRace {
        /// Event key, e.g. 2024-20.
        #[arg(long)]
        event: String,
        /// Seasons of the driver-skill window (the season and those before it).
        #[arg(long, default_value_t = car_pace::WINDOW_SEASONS)]
        window: i64,
        /// Also use sprint races as teammate comparisons for driver skills.
        #[arg(long)]
        with_sprints: bool,
    },
}

/// Seasons whose races a car-only season needs (its driver-skill window).
fn window_runs(
    archive: &Archive,
    season: i64,
    window: i64,
    with_sprints: bool,
) -> Result<std::collections::BTreeMap<i64, Vec<car_pace::RaceRun>>> {
    let mut runs = std::collections::BTreeMap::new();
    for s in (season - window + 1)..=season {
        runs.insert(s, car_pace::season_runs(archive, s, with_sprints)?);
    }
    Ok(runs)
}

/// The standard recipe, with optional changes to one of its parts for comparisons.
#[derive(clap::Args)]
struct RecipeArgs {
    /// all (dry, intermediate, wet), slick or inter (intermediate and wet)
    #[arg(long)]
    compounds: Option<String>,
    /// Leave out laps this close or closer to the car ahead at any timing line
    /// (seconds; 0 keeps all laps).
    #[arg(long)]
    min_gap: Option<f64>,
    /// Leave out laps this many or fewer after a neutralisation.
    #[arg(long)]
    skip_after_neutral: Option<u32>,
    /// Tyre-age slope per team.
    #[arg(long)]
    team_wear: bool,
    /// linear, lapxcomp, perlap or timeblocks
    #[arg(long)]
    trend: Option<String>,
    /// Residual trim in robust standard deviations (0 = no trim).
    #[arg(long)]
    trim: Option<f64>,
    /// Gap to the car ahead as a term.
    #[arg(long)]
    traffic_term: bool,
    /// best, mean or lapbest
    #[arg(long)]
    agg: Option<String>,
}

impl RecipeArgs {
    fn recipe(&self) -> Result<Recipe> {
        let mut r = Recipe::standard();
        if let Some(c) = &self.compounds {
            r.filter.compounds = match c.as_str() {
                "all" => Compounds::All,
                "slick" => Compounds::Slicks,
                "inter" => Compounds::Inters,
                other => anyhow::bail!("unknown compounds {other}"),
            };
        }
        if let Some(g) = self.min_gap {
            r.filter.min_gap_s = (g > 0.0).then_some(g);
        }
        if let Some(n) = self.skip_after_neutral {
            r.filter.skip_after_neutral = n;
        }
        if let Some(t) = &self.trend {
            r.model.lap_trend = match t.as_str() {
                "linear" => LapTrend::Linear,
                "lapxcomp" => LapTrend::LinearPerCompound,
                "perlap" => LapTrend::PerLap,
                "timeblocks" => LapTrend::TimeBlocks,
                other => anyhow::bail!("unknown trend {other}"),
            };
        }
        if let Some(k) = self.trim {
            r.model.trim_sigma = (k > 0.0).then_some(k);
        }
        r.model.team_wear |= self.team_wear;
        r.model.traffic_term |= self.traffic_term;
        if let Some(a) = &self.agg {
            r.agg = match a.as_str() {
                "best" => Agg::Best,
                "mean" => Agg::Mean,
                "lapbest" => Agg::LapBest,
                other => anyhow::bail!("unknown agg {other}"),
            };
        }
        Ok(r)
    }
}

fn write_json<T: serde::Serialize>(path: &PathBuf, value: &T) -> Result<()> {
    std::fs::write(path, serde_json::to_vec_pretty(value)?)?;
    Ok(())
}

fn interval(i: Option<[f64; 2]>) -> String {
    i.map_or("-".to_string(), |[lo, hi]| format!("{lo:.3}..{hi:.3}"))
}

fn print_race(p: &RacePace) {
    println!(
        "tyre-corrected {}  base lap {:.3} s  laps: {} candidates, {} unknown traffic, {} kept  resamples {} ({} used)",
        p.tyre_corrected,
        p.base_lap_s,
        p.laps.candidates,
        p.laps.unknown_traffic,
        p.laps.kept,
        p.resamples,
        p.resamples_used
    );
    println!("{:<18} {:<5} {:>8} {:>15} {:>8} {:>15} {:>7}", "team", "drv", "gap %", "80% %", "gap s", "80% s", "P(best)");
    for t in &p.teams {
        println!(
            "{:<18} {:<5} {:>8.3} {:>15} {:>8.3} {:>15} {:>7}",
            t.team,
            t.driver,
            t.gap_pct,
            interval(t.interval_pct),
            t.gap_s,
            interval(t.interval_s),
            t.p_quickest.map_or("-".to_string(), |p| format!("{:.0}%", p * 100.0))
        );
    }
    println!("{:<5} {:<18} {:>8} {:>15} {:>5} {:>6} {:>7}", "drv", "team", "gap %", "80% %", "laps", "stints", "overlap");
    for d in &p.drivers {
        println!(
            "{:<5} {:<18} {:>8.3} {:>15} {:>5} {:>6} {:>7}",
            d.driver,
            d.team,
            d.gap_pct,
            interval(d.interval_pct),
            d.laps,
            d.stints,
            d.overlap_laps
        );
    }
}

fn print_table(title: &str, t: &SeasonTable) {
    println!("\n{title}: {} races", t.races.len());
    println!(
        "{:<18} {:>3} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8}",
        "team", "n", "mean%", "med%", "mean%0", "med%0", "mean s", "med s", "mean s0", "med s0"
    );
    for team in &t.teams {
        let (p, s) = (&team.pct, &team.s);
        println!(
            "{:<18} {:>3} {:>8.3} {:>8.3} {:>8.3} {:>8.3} {:>8.3} {:>8.3} {:>8.3} {:>8.3}",
            team.team,
            team.races_measured,
            p.mean,
            p.median,
            p.mean_rebased,
            p.median_rebased,
            s.mean,
            s.median,
            s.mean_rebased,
            s.median_rebased
        );
    }
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let archive = Archive::locate(cli.data_root.as_deref())?;
    match cli.command {
        Command::Calibrate { event, reference, top, detail, min_laps, filter } => {
            anyhow::ensure!(event.len() == reference.len(), "one --reference per --event");
            let mut per_event = Vec::new();
            for (e, r) in event.iter().zip(&reference) {
                let table = calibrate::read_reference(r)?;
                let candidates = calibrate::candidates(&archive, e, min_laps)?;
                per_event.push((e.clone(), table, candidates));
            }
            // Recipe names are the same for every event; keep those complete everywhere.
            let names: Vec<String> = per_event[0].2.iter().map(|c| c.name.clone()).collect();
            let mut rows = Vec::new();
            for (i, name) in names.iter().enumerate() {
                if filter.as_deref().is_some_and(|f| !name.contains(f)) {
                    continue;
                }
                let scores: Vec<_> =
                    per_event.iter().map(|(_, table, cands)| calibrate::score(&cands[i], table)).collect();
                if scores.iter().any(|s| s.missing > 0) {
                    continue;
                }
                let mean = scores.iter().map(|s| s.mae).sum::<f64>() / scores.len() as f64;
                rows.push((i, mean, scores));
            }
            rows.sort_by(|a, b| a.1.total_cmp(&b.1));
            println!("{} recipes complete for every event", rows.len());
            print!("{:>4} {:>7}", "rank", "meanMAE");
            for e in &event {
                print!(" {:>9} {:>6} {:>6}", format!("{e} MAE"), "rho", "scale");
            }
            println!("  recipe");
            for (rank, (i, mean, scores)) in rows.iter().take(top).enumerate() {
                print!("{:>4} {:>7.3}", rank + 1, mean);
                for s in scores {
                    print!(" {:>9.3} {:>6.3} {:>6.3}", s.mae, s.spearman, s.scale);
                }
                println!("  {}", names[*i]);
            }
            for (i, _, _) in rows.iter().take(detail) {
                for (e, table, cands) in &per_event {
                    let unit = if table.unit == calibrate::Unit::Seconds { "s" } else { "%" };
                    println!("\n{e}  {}  (gaps in {unit})", names[*i]);
                    println!("{:<18} {:>8} {:>8} {:>8}", "team", "ours", "theirs", "diff");
                    let gaps = cands[*i].gaps(table.unit);
                    let mut teams: Vec<_> = table.gaps.iter().collect();
                    teams.sort_by(|a, b| a.1.total_cmp(b.1));
                    for (team, theirs) in teams {
                        let ours = gaps[team];
                        println!("{:<18} {:>8.3} {:>8.3} {:>+8.3}", team, ours, theirs, ours - theirs);
                    }
                }
            }
        }
        Command::Race { event, recipe, resamples, json } => {
            let recipe = recipe.recipe()?;
            let Some(race) = archive.session_of_type(&event, "R")? else {
                anyhow::bail!("{event} has no archived race")
            };
            let laps = archive.laps(&race.session_id)?;
            match pace::race_pace(&laps, &recipe, resamples, pace::seed_for(&race.session_id)) {
                Some(p) => {
                    print_race(&p);
                    if let Some(path) = json {
                        write_json(&path, &p)?;
                    }
                }
                None => println!("no laps could be measured"),
            }
        }
        Command::Season { season, recipe, resamples, json } => {
            let recipe = recipe.recipe()?;
            let s = season::season_pace(&archive, season, &recipe, resamples)?;
            let teams: Vec<String> = s.all_races.teams.iter().map(|t| t.team.clone()).collect();
            print!("{:<18}", "race gap %");
            for r in &s.races {
                print!(" {:>6}", r.event_key.rsplit('-').next().unwrap_or(""));
            }
            println!();
            for team in &teams {
                print!("{team:<18}");
                for r in &s.races {
                    let g = r.pace.as_ref().filter(|_| r.counted).and_then(|p| p.teams.iter().find(|t| &t.team == team));
                    match g {
                        Some(t) => print!(" {:>6.2}", t.gap_pct),
                        None => print!(" {:>6}", "-"),
                    }
                }
                println!();
            }
            for r in s.races.iter().filter(|r| !r.counted) {
                println!("not counted {}: {}", r.event_key, r.reason.as_deref().unwrap_or(""));
            }
            print_table("All counted races", &s.all_races);
            print_table("Common races only", &s.common_races);
            if let Some(path) = json {
                write_json(&path, &s)?;
            }
        }
        Command::CarSeason { season, json, show_skills, window, with_sprints } => {
            let runs = window_runs(&archive, season, window, with_sprints)?;
            let s = car_pace::car_season(season, &runs, window);
            println!("car-only race pace {season}: each race measured with earlier races only ({}-{} window), shrunk skills, skill drift {} %/season, established = {}+ teammate races", s.window[0], s.window[1], s.skill_drift, s.min_teammate_races);
            let teams: Vec<String> = s.all_races.teams.iter().map(|t| t.team.clone()).collect();
            print!("{:<18}", "race gap %");
            for r in &s.races {
                print!(" {:>6}", r.event_key.rsplit('-').next().unwrap_or(""));
            }
            println!();
            for team in &teams {
                print!("{team:<18}");
                for r in &s.races {
                    let run = r.runs.iter().find(|c| Some(c.condition) == r.headline).filter(|_| r.counted);
                    match run.and_then(|c| c.teams.iter().find(|t| &t.team == team)) {
                        Some(t) => print!(" {:>6}", format!("{:.2}{}", t.gap_pct, if t.note.is_some() { "*" } else { "" })),
                        None => print!(" {:>6}", "-"),
                    }
                }
                println!();
            }
            println!("* rookie in the car: measured from the established teammate only");
            for r in &s.races {
                if let Some(reason) = &r.reason {
                    println!("not counted {}: {reason}", r.event_key);
                }
                for c in &r.runs {
                    if !c.not_measured.is_empty() {
                        println!("{} {:?}: no established driver for {}", r.event_key, c.condition, c.not_measured.join(", "));
                    }
                }
            }
            print_table("All counted races", &s.all_races);
            print_table("Common races only", &s.common_races);
            if show_skills {
                let (drift, rows) = car_pace::skills_table(season, &runs, window);
                println!();
                println!(
                    "INTERNAL driver race-pace skill, end of {season} (% vs the season's average driver; negative = faster); skill drift {drift} %/season"
                );
                println!("{:<5} {:>8} {:>8} {:>7} {:>7} {:>8} {:>6}", "drv", "measured", "shrunk", "kept", "80% +-", "wet adj", "races");
                for r in rows {
                    let wet = r.wet_adjustment.map_or("-".to_string(), |w| format!("{w:.3}"));
                    println!(
                        "{:<5} {:>8.3} {:>8.3} {:>6.0}% {:>7.3} {wet:>8} {:>6}",
                        r.driver,
                        r.raw,
                        r.shrunk,
                        r.factor * 100.0,
                        r.range80,
                        r.teammate_races
                    );
                }
            }
            if let Some(path) = json {
                write_json(&path, &s)?;
            }
        }
        Command::CarRace { event, window, with_sprints } => {
            let season: i64 = event.split('-').next().unwrap_or("").parse()?;
            let runs = window_runs(&archive, season, window, with_sprints)?;
            let s = car_pace::car_season(season, &runs, window);
            let Some(race) = s.races.iter().find(|r| r.event_key == event) else {
                anyhow::bail!("{event} has no measured race")
            };
            for run in &race.runs {
                let headline = if Some(run.condition) == race.headline { " (headline)" } else { "" };
                println!("{event} {:?} laps{headline}", run.condition);
                println!("{:<18} {:>8} {:>8} {:>15}  drivers", "team", "gap %", "gap s", "80% %");
                for t in &run.teams {
                    println!(
                        "{:<18} {:>8.3} {:>8.3} {:>15}  {}{}",
                        t.team,
                        t.gap_pct,
                        t.gap_s,
                        interval(t.interval_pct),
                        t.drivers.join(", "),
                        if t.note.is_some() { "  *rookie: teammate only" } else { "" }
                    );
                }
                if !run.not_measured.is_empty() {
                    println!("no established driver: {}", run.not_measured.join(", "));
                }
            }
        }
    }
    Ok(())
}

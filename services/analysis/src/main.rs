use std::path::PathBuf;

use anyhow::Result;
use clap::{Parser, Subcommand};

use analysis::archive::Archive;
use analysis::calibrate;
use analysis::pace::{Agg, Compounds, LapTrend, RaceFilter, RaceModel};
use analysis::season;

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
    /// Run one team race pace recipe on every race of a season and average it.
    Season {
        #[arg(long)]
        season: i64,
        #[command(flatten)]
        recipe: Recipe,
        /// Drivers (or teams) with fewer kept laps are left out.
        #[arg(long, default_value_t = 5)]
        min_laps: usize,
    },
}

/// A team race pace recipe.
#[derive(clap::Args)]
struct Recipe {
    /// all, slick or inter
    #[arg(long, default_value = "slick")]
    compounds: String,
    /// Leave out laps this close or closer to the car ahead (seconds).
    #[arg(long)]
    min_gap: Option<f64>,
    /// Leave out laps this many or fewer after a neutralisation.
    #[arg(long, default_value_t = 0)]
    skip_after_neutral: u32,
    /// Tyre-age slope per team.
    #[arg(long)]
    team_wear: bool,
    /// linear, lapxcomp or perlap
    #[arg(long, default_value = "perlap")]
    trend: String,
    /// Residual trim in robust standard deviations.
    #[arg(long)]
    trim: Option<f64>,
    /// Gap to the car ahead as a term.
    #[arg(long)]
    traffic: bool,
    /// best, mean or lapbest
    #[arg(long, default_value = "best")]
    agg: String,
}

impl Recipe {
    fn parts(&self) -> Result<(RaceFilter, RaceModel, Agg)> {
        let compounds = match self.compounds.as_str() {
            "all" => Compounds::All,
            "slick" => Compounds::Slicks,
            "inter" => Compounds::Inters,
            other => anyhow::bail!("unknown compounds {other}"),
        };
        let lap_trend = match self.trend.as_str() {
            "linear" => LapTrend::Linear,
            "lapxcomp" => LapTrend::LinearPerCompound,
            "perlap" => LapTrend::PerLap,
            other => anyhow::bail!("unknown trend {other}"),
        };
        let agg = match self.agg.as_str() {
            "best" => Agg::Best,
            "mean" => Agg::Mean,
            "lapbest" => Agg::LapBest,
            other => anyhow::bail!("unknown agg {other}"),
        };
        let filter = RaceFilter {
            compounds,
            min_gap_s: self.min_gap,
            skip_after_neutral: self.skip_after_neutral,
            ..RaceFilter::default()
        };
        let model = RaceModel { team_wear: self.team_wear, lap_trend, trim_sigma: self.trim, traffic_term: self.traffic };
        Ok((filter, model, agg))
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
        Command::Season { season, recipe, min_laps } => {
            let (filter, model, agg) = recipe.parts()?;
            let races = season::race_gaps(&archive, season, &filter, &model, agg, min_laps)?;
            let means = season::mean_gaps(&races);
            let mut teams: Vec<_> = means.iter().collect();
            teams.sort_by(|a, b| a.1 .0.total_cmp(&b.1 .0));
            print!("{:<18} {:>7} {:>3}", "team", "mean%", "n");
            for r in &races {
                print!(" {:>6}", r.event_key.rsplit('-').next().unwrap_or(""));
            }
            println!();
            for (team, (mean, n)) in teams {
                print!("{:<18} {:>7.3} {:>3}", team, mean, n);
                for r in &races {
                    match r.gaps_pct.get(team) {
                        Some(g) => print!(" {:>6.2}", g),
                        None => print!(" {:>6}", "-"),
                    }
                }
                println!();
            }
        }
    }
    Ok(())
}

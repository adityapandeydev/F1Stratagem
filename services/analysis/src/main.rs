use std::path::PathBuf;

use anyhow::Result;
use clap::{Parser, Subcommand};

use analysis::archive::Archive;
use analysis::calibrate;

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
    /// Rank team-pace recipes by how closely they reproduce a published table.
    Calibrate {
        /// Event key, e.g. 2024-20.
        #[arg(long)]
        event: String,
        /// CSV of team,pct (published gap to the quickest team).
        #[arg(long)]
        reference: PathBuf,
        /// Recipes to list.
        #[arg(long, default_value_t = 25)]
        top: usize,
        /// Recipes to show team by team.
        #[arg(long, default_value_t = 3)]
        detail: usize,
        /// Drivers with fewer kept race laps are left out.
        #[arg(long, default_value_t = 5)]
        min_laps: usize,
        /// Only recipes whose name contains this text.
        #[arg(long)]
        filter: Option<String>,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let archive = Archive::locate(cli.data_root.as_deref())?;
    match cli.command {
        Command::Calibrate { event, reference, top, detail, min_laps, filter } => {
            let reference = calibrate::read_reference(&reference)?;
            let mut scored: Vec<_> = calibrate::candidates(&archive, &event, min_laps)?
                .into_iter()
                .filter(|c| filter.as_deref().is_none_or(|f| c.name.contains(f)))
                .map(|c| {
                    let s = calibrate::score(&c.gaps, &reference);
                    (c, s)
                })
                .filter(|(_, s)| s.missing == 0)
                .collect();
            scored.sort_by(|a, b| a.1.mae.total_cmp(&b.1.mae));
            println!("{} complete recipes", scored.len());
            println!("{:>4} {:>7} {:>7} {:>6} {:>6} {:>5}  recipe", "rank", "MAE", "max", "rho", "scale", "best");
            for (i, (c, s)) in scored.iter().take(top).enumerate() {
                println!(
                    "{:>4} {:>7.3} {:>7.3} {:>6.3} {:>6.3} {:>5}  {}",
                    i + 1,
                    s.mae,
                    s.max_err,
                    s.spearman,
                    s.scale,
                    if s.same_best { "yes" } else { "no" },
                    c.name
                );
            }
            for (c, _) in scored.iter().take(detail) {
                println!("\n{}", c.name);
                println!("{:<18} {:>8} {:>8} {:>8}", "team", "ours", "theirs", "diff");
                let mut rows: Vec<_> = reference.iter().collect();
                rows.sort_by(|a, b| a.1.total_cmp(b.1));
                for (team, theirs) in rows {
                    let ours = c.gaps[team];
                    println!("{:<18} {:>8.3} {:>8.3} {:>+8.3}", team, ours, theirs, ours - theirs);
                }
            }
        }
    }
    Ok(())
}

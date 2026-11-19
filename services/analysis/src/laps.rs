//! Laps as recorded by the timing feed (`normalized/laps.parquet`), times in seconds.

use std::collections::HashMap;
use std::path::Path;

use anyhow::Result;

use crate::table::Table;

const NS_PER_S: f64 = 1e9;

#[derive(Debug, Clone)]
pub struct Lap {
    pub driver: String,
    pub team: String,
    pub lap_number: u32,
    pub lap_time: Option<f64>,
    /// Session time when the lap was completed.
    pub time: Option<f64>,
    pub stint: Option<u32>,
    pub pit_in: bool,
    pub pit_out: bool,
    pub compound: Option<String>,
    pub tyre_life: Option<f64>,
    pub fresh_tyre: Option<bool>,
    /// Every track status seen during the lap ("1" = green throughout).
    pub track_status: Option<String>,
    pub position: Option<u32>,
    pub deleted: bool,
    pub accurate: bool,
}

impl Lap {
    pub fn is_green(&self) -> bool {
        self.track_status.as_deref() == Some("1")
    }

    pub fn is_slick(&self) -> bool {
        matches!(self.compound.as_deref(), Some("SOFT" | "MEDIUM" | "HARD" | "HYPERSOFT" | "ULTRASOFT" | "SUPERSOFT"))
    }

    pub fn is_wet_tyre(&self) -> bool {
        matches!(self.compound.as_deref(), Some("INTERMEDIATE" | "WET"))
    }

    pub fn is_pit_lap(&self) -> bool {
        self.pit_in || self.pit_out
    }
}

pub fn read(path: &Path) -> Result<Vec<Lap>> {
    let t = Table::read(path)?;
    let seconds = |name: &str| -> Result<Vec<Option<f64>>> {
        Ok(t.i64s(name)?.into_iter().map(|v| v.map(|ns| ns as f64 / NS_PER_S)).collect())
    };
    let driver = t.strings("Driver")?;
    let team = t.strings("Team")?;
    let lap_number = t.f64s("LapNumber")?;
    let lap_time = seconds("LapTime")?;
    let time = seconds("Time")?;
    let stint = t.f64s("Stint")?;
    let pit_in = t.i64s("PitInTime")?;
    let pit_out = t.i64s("PitOutTime")?;
    let compound = t.strings("Compound")?;
    let tyre_life = t.f64s("TyreLife")?;
    let fresh = t.bools("FreshTyre")?;
    let status = t.strings("TrackStatus")?;
    let position = t.f64s("Position")?;
    let deleted = t.bools("Deleted")?;
    let accurate = t.bools("IsAccurate")?;

    Ok((0..t.len())
        .filter_map(|i| {
            Some(Lap {
                driver: driver[i].clone()?,
                team: team[i].clone().unwrap_or_default(),
                lap_number: lap_number[i]? as u32,
                lap_time: lap_time[i],
                time: time[i],
                stint: stint[i].map(|s| s as u32),
                pit_in: pit_in[i].is_some(),
                pit_out: pit_out[i].is_some(),
                compound: compound[i].clone(),
                tyre_life: tyre_life[i],
                fresh_tyre: fresh[i],
                track_status: status[i].clone(),
                position: position[i].map(|p| p as u32),
                deleted: deleted[i].unwrap_or(false),
                accurate: accurate[i].unwrap_or(false),
            })
        })
        .collect())
}

/// Gap in seconds to the car one place ahead at the end of the same lap, from the
/// lap completion times. `None` for the leader or when either time is missing.
pub fn gaps_ahead(laps: &[Lap]) -> Vec<Option<f64>> {
    let mut crossing: HashMap<(u32, u32), f64> = HashMap::new();
    for lap in laps {
        if let (Some(p), Some(t)) = (lap.position, lap.time) {
            crossing.insert((lap.lap_number, p), t);
        }
    }
    laps.iter()
        .map(|lap| {
            let (p, t) = (lap.position?, lap.time?);
            if p <= 1 {
                return None;
            }
            crossing.get(&(lap.lap_number, p - 1)).map(|ahead| t - ahead)
        })
        .collect()
}

/// The team each driver drove for in this session (first lap seen).
pub fn driver_teams(laps: &[Lap]) -> HashMap<String, String> {
    let mut teams = HashMap::new();
    for lap in laps {
        teams.entry(lap.driver.clone()).or_insert_with(|| lap.team.clone());
    }
    teams
}

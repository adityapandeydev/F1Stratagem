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
    /// Session times when the car crossed the first and second sector lines.
    pub sector1_time: Option<f64>,
    pub sector2_time: Option<f64>,
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
        matches!(self.compound.as_deref(), Some("SOFT" | "MEDIUM" | "HARD" | "HYPERSOFT" | "ULTRASOFT" | "SUPERSOFT" | "SUPERHARD"))
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
    let sector1_time = seconds("Sector1SessionTime")?;
    let sector2_time = seconds("Sector2SessionTime")?;
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
                sector1_time: sector1_time[i],
                sector2_time: sector2_time[i],
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

/// Gap in seconds to the car physically ahead on track when the lap was completed:
/// the last other car to cross the line before this one, whatever its race position
/// or lap count, so lapping and being lapped count as traffic. A car finishing an
/// in-lap is in the pit lane and is not traffic. `None` when no car crossed before
/// (or the time is missing).
pub fn gaps_ahead(laps: &[Lap]) -> Vec<Option<f64>> {
    let mut crossings: Vec<(f64, &str)> = laps
        .iter()
        .filter(|l| !l.pit_in)
        .filter_map(|l| l.time.map(|t| (t, l.driver.as_str())))
        .collect();
    crossings.sort_by(|a, b| a.0.total_cmp(&b.0));
    laps.iter()
        .map(|lap| {
            let t = lap.time?;
            let before = crossings.partition_point(|(c, _)| *c < t);
            crossings[..before].iter().rev().find(|(_, d)| *d != lap.driver).map(|(c, _)| t - c)
        })
        .collect()
}

/// Traffic status at one timing line.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Gap {
    /// This car's crossing time is missing: traffic unknown.
    Unknown,
    /// No other car crossed this line before it.
    NoCarAhead,
    /// Seconds behind the last other car to cross the line.
    Behind(f64),
}

impl Gap {
    pub fn clear_of(self, min_s: f64) -> Option<bool> {
        match self {
            Gap::Unknown => None,
            Gap::NoCarAhead => Some(true),
            Gap::Behind(g) => Some(g > min_s),
        }
    }
}

/// Gaps to the car physically ahead at the three timing lines of each lap (first
/// sector line, second sector line, finish line), each from the last other car to
/// cross that line before this one, whatever its position or lap count. Crossings
/// of the finish line at the end of an in-lap are in the pit lane and are not
/// traffic.
pub fn gaps_at_timing_lines(laps: &[Lap]) -> Vec<[Gap; 3]> {
    let line = |crossing: &dyn Fn(&Lap) -> Option<f64>, skip: &dyn Fn(&Lap) -> bool| -> Vec<Gap> {
        let mut all: Vec<(f64, &str)> =
            laps.iter().filter(|l| !skip(l)).filter_map(|l| crossing(l).map(|t| (t, l.driver.as_str()))).collect();
        all.sort_by(|a, b| a.0.total_cmp(&b.0));
        laps.iter()
            .map(|lap| {
                let Some(t) = crossing(lap) else { return Gap::Unknown };
                let before = all.partition_point(|(c, _)| *c < t);
                match all[..before].iter().rev().find(|(_, d)| *d != lap.driver) {
                    Some((c, _)) => Gap::Behind(t - c),
                    None => Gap::NoCarAhead,
                }
            })
            .collect()
    };
    let s1 = line(&|l| l.sector1_time, &|_| false);
    let s2 = line(&|l| l.sector2_time, &|_| false);
    let finish = line(&|l| l.time, &|l| l.pit_in);
    (0..laps.len()).map(|i| [s1[i], s2[i], finish[i]]).collect()
}

/// The team each driver drove for in this session (first lap seen).
pub fn driver_teams(laps: &[Lap]) -> HashMap<String, String> {
    let mut teams = HashMap::new();
    for lap in laps {
        teams.entry(lap.driver.clone()).or_insert_with(|| lap.team.clone());
    }
    teams
}

#[cfg(test)]
mod tests {
    use super::*;

    fn crossing(driver: &str, n: u32, time: f64, position: u32) -> Lap {
        Lap {
            driver: driver.into(),
            team: String::new(),
            lap_number: n,
            lap_time: Some(90.0),
            time: Some(time),
            sector1_time: None,
            sector2_time: None,
            stint: None,
            pit_in: false,
            pit_out: false,
            compound: Some("MEDIUM".into()),
            tyre_life: Some(1.0),
            fresh_tyre: None,
            track_status: Some("1".into()),
            position: Some(position),
            deleted: false,
            accurate: true,
        }
    }

    #[test]
    fn a_lapped_car_ahead_on_track_is_traffic() {
        // The leader (P1, lap 30) crosses 0.8 s behind a backmarker (P18, lap 29).
        let laps = vec![crossing("BACK", 29, 1000.0, 18), crossing("LEAD", 30, 1000.8, 1), crossing("P2", 30, 1010.0, 2)];
        let gaps = gaps_ahead(&laps);
        assert_eq!(gaps[0], None);
        assert!((gaps[1].unwrap() - 0.8).abs() < 1e-9);
        assert!((gaps[2].unwrap() - 9.2).abs() < 1e-9);
    }

    #[test]
    fn a_car_in_the_pit_lane_is_not_traffic() {
        let mut pitting = crossing("PIT", 10, 500.0, 3);
        pitting.pit_in = true;
        let laps = vec![crossing("AHEAD", 10, 495.0, 2), pitting, crossing("ME", 10, 500.5, 4)];
        assert!((gaps_ahead(&laps)[2].unwrap() - 5.5).abs() < 1e-9);
    }

    #[test]
    fn timing_lines_see_traffic_the_finish_line_misses() {
        // B is 0.6 s behind A at the first sector line, passes, and finishes 3 s ahead.
        let mut a = crossing("A", 5, 1003.0, 2);
        a.sector1_time = Some(930.0);
        a.sector2_time = Some(960.0);
        let mut b = crossing("B", 5, 1000.0, 1);
        b.sector1_time = Some(930.6);
        b.sector2_time = Some(959.0);
        let mut c = crossing("C", 5, 1050.0, 3);
        c.sector1_time = None;
        c.sector2_time = Some(1010.0);
        let gaps = gaps_at_timing_lines(&[a, b, c]);
        assert!(matches!(gaps[1][0], Gap::Behind(g) if (g - 0.6).abs() < 1e-9));
        assert_eq!(gaps[1][2], Gap::NoCarAhead);
        assert_eq!(gaps[2][0], Gap::Unknown);
        assert_eq!(gaps[1][0].clear_of(2.0), Some(false));
    }

    #[test]
    fn superhard_is_a_dry_tyre() {
        let mut lap = crossing("A", 1, 0.0, 1);
        lap.compound = Some("SUPERHARD".into());
        assert!(lap.is_slick());
    }
}

//! F1Stratagem analysis service: measurements computed from the archive's raw data.
//!
//! Every number is a calculation on the recorded data of the session(s) it
//! describes: no priors, no weighting or smoothing across sessions.

pub mod archive;
pub mod calibrate;
pub mod car_pace;
pub mod laps;
pub mod pace;
pub mod season;
pub mod table;

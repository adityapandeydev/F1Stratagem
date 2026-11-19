//! Read-only access to the F1Stratagem data archive (written by `packages/f1data`).

use std::path::{Path, PathBuf};

use anyhow::{bail, ensure, Context, Result};

use crate::laps::{self, Lap};
use crate::table::Table;

/// One archived session as listed in `reference/sessions.parquet`.
#[derive(Debug, Clone)]
pub struct Session {
    pub session_id: String,
    pub event_key: String,
    /// FP1, FP2, FP3, SQ, S, Q, R (testing sessions have their own names).
    pub session_type: String,
    pub event_format: String,
}

pub struct Archive {
    root: PathBuf,
}

impl Archive {
    /// The archive at `explicit`, else `F1_DATA_ROOT`, else the repository's `data/`.
    pub fn locate(explicit: Option<&Path>) -> Result<Self> {
        let root = match (explicit, std::env::var_os("F1_DATA_ROOT")) {
            (Some(path), _) => path.to_path_buf(),
            (None, Some(var)) => PathBuf::from(var),
            (None, None) => Path::new(env!("CARGO_MANIFEST_DIR")).join("../../data"),
        };
        ensure!(root.join("sessions").is_dir(), "no archive at {}", root.display());
        Ok(Self { root })
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    /// Every session of an event (`2024-20`), in slot order.
    pub fn event_sessions(&self, event_key: &str) -> Result<Vec<Session>> {
        let table = Table::read(&self.root.join("reference/sessions.parquet"))?;
        let ids = table.strings("session_id")?;
        let events = table.strings("event_key")?;
        let types = table.strings("session_type")?;
        let formats = table.strings("event_format")?;
        let slots = table.i64s("slot")?;
        let mut sessions: Vec<(i64, Session)> = (0..table.len())
            .filter(|&i| events[i].as_deref() == Some(event_key))
            .map(|i| {
                (
                    slots[i].unwrap_or(0),
                    Session {
                        session_id: ids[i].clone().unwrap_or_default(),
                        event_key: event_key.to_string(),
                        session_type: types[i].clone().unwrap_or_default(),
                        event_format: formats[i].clone().unwrap_or_default(),
                    },
                )
            })
            .collect();
        if sessions.is_empty() {
            bail!("event {event_key} is not in the archive's session reference");
        }
        sessions.sort_by_key(|(slot, _)| *slot);
        Ok(sessions.into_iter().map(|(_, s)| s).collect())
    }

    /// The event's session of one type (`Q`, `R`, ...), if it ran.
    pub fn session_of_type(&self, event_key: &str, session_type: &str) -> Result<Option<Session>> {
        Ok(self.event_sessions(event_key)?.into_iter().find(|s| s.session_type == session_type))
    }

    pub fn session_dir(&self, session_id: &str) -> Result<PathBuf> {
        let year = session_id.split('-').next().context("session id without a year")?;
        Ok(self.root.join("sessions").join(year).join(session_id))
    }

    pub fn laps(&self, session_id: &str) -> Result<Vec<Lap>> {
        laps::read(&self.session_dir(session_id)?.join("normalized/laps.parquet"))
    }
}

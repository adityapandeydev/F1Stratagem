-- 001_core_schema.up.sql
-- Core F1Stratagem database schema

-- Enable UUID extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- ============================================================
-- SEASONS
-- ============================================================
CREATE TABLE seasons (
    id          SERIAL PRIMARY KEY,
    year        INTEGER NOT NULL UNIQUE,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- ============================================================
-- CIRCUITS
-- ============================================================
CREATE TABLE circuits (
    id              SERIAL PRIMARY KEY,
    circuit_key     VARCHAR(64) NOT NULL UNIQUE,      -- e.g. "bahrain", "monza"
    name            VARCHAR(256) NOT NULL,              -- e.g. "Bahrain International Circuit"
    country         VARCHAR(128) NOT NULL,
    location        VARCHAR(128),                       -- city
    latitude        DOUBLE PRECISION,
    longitude       DOUBLE PRECISION,
    track_length_m  INTEGER,
    num_corners     INTEGER,
    num_drs_zones   INTEGER,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- ============================================================
-- EVENTS (a race weekend)
-- ============================================================
CREATE TABLE events (
    id                  SERIAL PRIMARY KEY,
    season_id           INTEGER NOT NULL REFERENCES seasons(id),
    circuit_id          INTEGER REFERENCES circuits(id),
    round_number        INTEGER NOT NULL,
    event_name          VARCHAR(256) NOT NULL,           -- "Bahrain Grand Prix"
    official_name       VARCHAR(512),
    country             VARCHAR(128),
    location            VARCHAR(128),
    event_format        VARCHAR(32) NOT NULL DEFAULT 'conventional', -- conventional / sprint / sprint_shootout / testing
    event_date          DATE,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE(season_id, round_number)
);

-- ============================================================
-- SESSIONS (FP1, FP2, FP3, Qualifying, Sprint, Race)
-- ============================================================
CREATE TABLE sessions (
    id                  SERIAL PRIMARY KEY,
    event_id            INTEGER NOT NULL REFERENCES events(id),
    session_name        VARCHAR(64) NOT NULL,             -- "FP1", "Qualifying", "Race"
    session_type        VARCHAR(32) NOT NULL,              -- "practice", "qualifying", "sprint", "race"
    session_date        TIMESTAMPTZ,
    total_laps          INTEGER,
    data_status         VARCHAR(32) NOT NULL DEFAULT 'none', -- none / pending / downloading / processing / ready / error
    data_error          TEXT,
    f1_api_support      BOOLEAN DEFAULT true,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE(event_id, session_name)
);

CREATE INDEX idx_sessions_data_status ON sessions(data_status);

-- ============================================================
-- DRIVERS
-- ============================================================
CREATE TABLE drivers (
    id              SERIAL PRIMARY KEY,
    driver_number   VARCHAR(4),
    broadcast_name  VARCHAR(128),
    abbreviation    VARCHAR(4) NOT NULL,
    driver_id       VARCHAR(64) UNIQUE,                    -- FastF1/Ergast identifier
    first_name      VARCHAR(128) NOT NULL,
    last_name       VARCHAR(128) NOT NULL,
    full_name       VARCHAR(256),
    country_code    VARCHAR(4),
    headshot_url    TEXT,
    date_of_birth   DATE,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- ============================================================
-- TEAMS
-- ============================================================
CREATE TABLE teams (
    id              SERIAL PRIMARY KEY,
    team_id         VARCHAR(64) UNIQUE,
    name            VARCHAR(256) NOT NULL,
    color           VARCHAR(8),                             -- hex color e.g. "FF8700"
    country         VARCHAR(128),
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- ============================================================
-- DRIVER-SEASON (maps a driver to a team in a specific season)
-- ============================================================
CREATE TABLE driver_seasons (
    id              SERIAL PRIMARY KEY,
    driver_id       INTEGER NOT NULL REFERENCES drivers(id),
    season_id       INTEGER NOT NULL REFERENCES seasons(id),
    team_id         INTEGER REFERENCES teams(id),
    driver_number   VARCHAR(4),
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE(driver_id, season_id)
);

-- ============================================================
-- SESSION RESULTS
-- ============================================================
CREATE TABLE session_results (
    id                  SERIAL PRIMARY KEY,
    session_id          INTEGER NOT NULL REFERENCES sessions(id),
    driver_id           INTEGER NOT NULL REFERENCES drivers(id),
    team_id             INTEGER REFERENCES teams(id),
    position            INTEGER,
    classified_position VARCHAR(8),
    grid_position       INTEGER,
    q1_time_ms          INTEGER,
    q2_time_ms          INTEGER,
    q3_time_ms          INTEGER,
    race_time_ms        BIGINT,                              -- total race time or gap
    status              VARCHAR(128),                         -- "Finished", "+1 Lap", "Retired", etc.
    points              REAL DEFAULT 0,
    fastest_lap_rank    INTEGER,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE(session_id, driver_id)
);

CREATE INDEX idx_session_results_session ON session_results(session_id);

-- ============================================================
-- LAPS
-- ============================================================
CREATE TABLE laps (
    id                  SERIAL PRIMARY KEY,
    session_id          INTEGER NOT NULL REFERENCES sessions(id),
    driver_id           INTEGER NOT NULL REFERENCES drivers(id),
    lap_number          INTEGER NOT NULL,
    lap_time_ms         INTEGER,                              -- milliseconds
    sector1_ms          INTEGER,
    sector2_ms          INTEGER,
    sector3_ms          INTEGER,
    speed_i1            REAL,
    speed_i2            REAL,
    speed_fl            REAL,
    speed_st            REAL,
    compound            VARCHAR(16),                           -- SOFT, MEDIUM, HARD, INTERMEDIATE, WET
    tyre_life           REAL,
    fresh_tyre          BOOLEAN,
    stint               INTEGER,
    is_personal_best    BOOLEAN DEFAULT false,
    is_deleted          BOOLEAN DEFAULT false,
    deleted_reason      VARCHAR(128),
    track_status        VARCHAR(16),
    position            INTEGER,
    is_accurate         BOOLEAN DEFAULT true,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE(session_id, driver_id, lap_number)
);

CREATE INDEX idx_laps_session ON laps(session_id);
CREATE INDEX idx_laps_driver ON laps(driver_id);
CREATE INDEX idx_laps_session_driver ON laps(session_id, driver_id);

-- ============================================================
-- WEATHER DATA
-- ============================================================
CREATE TABLE weather_data (
    id              SERIAL PRIMARY KEY,
    session_id      INTEGER NOT NULL REFERENCES sessions(id),
    time_offset_ms  BIGINT NOT NULL,                          -- milliseconds from session start
    air_temp        REAL,
    track_temp      REAL,
    humidity        REAL,
    pressure        REAL,
    wind_speed      REAL,
    wind_direction  INTEGER,
    rainfall        BOOLEAN DEFAULT false,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_weather_session ON weather_data(session_id);

-- ============================================================
-- TELEMETRY DATASETS (metadata about stored Parquet files)
-- ============================================================
CREATE TABLE telemetry_datasets (
    id              SERIAL PRIMARY KEY,
    session_id      INTEGER NOT NULL REFERENCES sessions(id),
    driver_id       INTEGER NOT NULL REFERENCES drivers(id),
    dataset_type    VARCHAR(32) NOT NULL,                      -- "car_telemetry", "position", "combined"
    file_path       TEXT NOT NULL,
    file_size_bytes BIGINT,
    sample_count    INTEGER,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE(session_id, driver_id, dataset_type)
);

CREATE INDEX idx_telemetry_session ON telemetry_datasets(session_id);

-- ============================================================
-- INGESTION JOBS (track data fetch status)
-- ============================================================
CREATE TABLE ingestion_jobs (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    session_id      INTEGER NOT NULL REFERENCES sessions(id),
    status          VARCHAR(32) NOT NULL DEFAULT 'pending',   -- pending / downloading / processing / ready / error
    progress        INTEGER DEFAULT 0,                         -- 0-100
    error_message   TEXT,
    started_at      TIMESTAMPTZ,
    completed_at    TIMESTAMPTZ,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Only one active (non-terminal) job per session
CREATE UNIQUE INDEX idx_ingestion_jobs_active
    ON ingestion_jobs(session_id)
    WHERE status NOT IN ('ready', 'error');

CREATE INDEX idx_ingestion_jobs_session ON ingestion_jobs(session_id);
CREATE INDEX idx_ingestion_jobs_status ON ingestion_jobs(status);

-- ============================================================
-- REGULATION ERAS (for historical context)
-- ============================================================
CREATE TABLE regulation_eras (
    id              SERIAL PRIMARY KEY,
    name            VARCHAR(128) NOT NULL,
    start_year      INTEGER NOT NULL,
    end_year        INTEGER,
    engine_type     VARCHAR(64),
    aero_rules      VARCHAR(256),
    description     TEXT,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Seed initial eras
INSERT INTO regulation_eras (name, start_year, end_year, engine_type, aero_rules) VALUES
    ('V10 Era', 1995, 2005, 'V10 3.0L NA', 'Various'),
    ('V8 Era', 2006, 2013, 'V8 2.4L NA', 'Various'),
    ('V6 Hybrid Era', 2014, 2021, 'V6 1.6L Turbo Hybrid', 'High downforce'),
    ('Ground Effect Era', 2022, NULL, 'V6 1.6L Turbo Hybrid', 'Ground effect aero');

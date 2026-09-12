package repository

import (
	"context"
	"fmt"
	"time"

	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"

	"github.com/aditya/f1stratagem/api/internal/model"
)

type Repository struct {
	pool *pgxpool.Pool
}

func New(pool *pgxpool.Pool) *Repository {
	return &Repository{pool: pool}
}

// ---------------------------------------------------------------------
// Seasons
// ---------------------------------------------------------------------

func (r *Repository) GetSeasons(ctx context.Context) ([]model.Season, error) {
	rows, err := r.pool.Query(ctx, "SELECT id, year, created_at, updated_at FROM seasons ORDER BY year DESC")
	if err != nil {
		return nil, fmt.Errorf("query seasons: %w", err)
	}
	defer rows.Close()

	var seasons []model.Season
	for rows.Next() {
		var s model.Season
		if err := rows.Scan(&s.ID, &s.Year, &s.CreatedAt, &s.UpdatedAt); err != nil {
			return nil, fmt.Errorf("scan season: %w", err)
		}
		seasons = append(seasons, s)
	}
	return seasons, nil
}

func (r *Repository) GetSeasonByYear(ctx context.Context, year int) (*model.Season, error) {
	var s model.Season
	err := r.pool.QueryRow(ctx, "SELECT id, year, created_at, updated_at FROM seasons WHERE year = $1", year).
		Scan(&s.ID, &s.Year, &s.CreatedAt, &s.UpdatedAt)
	if err != nil {
		if err == pgx.ErrNoRows {
			return nil, nil
		}
		return nil, fmt.Errorf("query season by year: %w", err)
	}
	return &s, nil
}

// ---------------------------------------------------------------------
// Circuits
// ---------------------------------------------------------------------

func (r *Repository) GetCircuits(ctx context.Context) ([]model.Circuit, error) {
	rows, err := r.pool.Query(ctx, `
		SELECT id, circuit_key, name, country, location, latitude, longitude, track_length_m, num_corners, num_drs_zones 
		FROM circuits ORDER BY name ASC
	`)
	if err != nil {
		return nil, fmt.Errorf("query circuits: %w", err)
	}
	defer rows.Close()

	var circuits []model.Circuit
	for rows.Next() {
		var c model.Circuit
		if err := rows.Scan(&c.ID, &c.CircuitKey, &c.Name, &c.Country, &c.Location, &c.Latitude, &c.Longitude, &c.TrackLengthM, &c.NumCorners, &c.NumDRSZones); err != nil {
			return nil, fmt.Errorf("scan circuit: %w", err)
		}
		circuits = append(circuits, c)
	}
	return circuits, nil
}

// ---------------------------------------------------------------------
// Events
// ---------------------------------------------------------------------

func (r *Repository) GetEventsBySeasonYear(ctx context.Context, year int) ([]model.Event, error) {
	query := `
		SELECT e.id, e.season_id, e.circuit_id, e.round_number, e.event_name, e.official_name, 
		       e.country, e.location, e.event_format, e.event_date::text,
		       c.id, c.circuit_key, c.name, c.country, c.location, c.latitude, c.longitude, c.track_length_m, c.num_corners, c.num_drs_zones
		FROM events e
		JOIN seasons s ON s.id = e.season_id
		LEFT JOIN circuits c ON c.id = e.circuit_id
		WHERE s.year = $1
		ORDER BY e.round_number ASC
	`
	rows, err := r.pool.Query(ctx, query, year)
	if err != nil {
		return nil, fmt.Errorf("query events: %w", err)
	}
	defer rows.Close()

	var events []model.Event
	for rows.Next() {
		var e model.Event
		var cID *int
		var cKey, cName, cCountry, cLoc *string
		var cLat, cLon *float64
		var cLen, cCorners, cDRS *int

		if err := rows.Scan(
			&e.ID, &e.SeasonID, &e.CircuitID, &e.RoundNumber, &e.EventName, &e.OfficialName,
			&e.Country, &e.Location, &e.EventFormat, &e.EventDate,
			&cID, &cKey, &cName, &cCountry, &cLoc, &cLat, &cLon, &cLen, &cCorners, &cDRS,
		); err != nil {
			return nil, fmt.Errorf("scan event: %w", err)
		}

		if cID != nil && cKey != nil && cName != nil {
			e.Circuit = &model.Circuit{
				ID:           *cID,
				CircuitKey:   *cKey,
				Name:         *cName,
				Country:      *cCountry,
				Location:     cLoc,
				Latitude:     cLat,
				Longitude:    cLon,
				TrackLengthM: cLen,
				NumCorners:   cCorners,
				NumDRSZones:  cDRS,
			}
		}
		events = append(events, e)
	}
	return events, nil
}

// ---------------------------------------------------------------------
// Sessions
// ---------------------------------------------------------------------

func (r *Repository) GetSessionsByEventID(ctx context.Context, eventID int) ([]model.Session, error) {
	query := `
		SELECT id, event_id, session_name, session_type, session_date, total_laps, data_status, data_error
		FROM sessions
		WHERE event_id = $1
		ORDER BY id ASC
	`
	rows, err := r.pool.Query(ctx, query, eventID)
	if err != nil {
		return nil, fmt.Errorf("query sessions: %w", err)
	}
	defer rows.Close()

	var sessions []model.Session
	for rows.Next() {
		var s model.Session
		if err := rows.Scan(&s.ID, &s.EventID, &s.SessionName, &s.SessionType, &s.SessionDate, &s.TotalLaps, &s.DataStatus, &s.DataError); err != nil {
			return nil, fmt.Errorf("scan session: %w", err)
		}
		sessions = append(sessions, s)
	}
	return sessions, nil
}

func (r *Repository) GetSessionByID(ctx context.Context, sessionID int) (*model.Session, error) {
	query := `
		SELECT s.id, s.event_id, s.session_name, s.session_type, s.session_date::text, s.total_laps, s.data_status, s.data_error,
		       e.id, e.season_id, e.circuit_id, e.round_number, e.event_name, e.official_name, e.country, e.location, e.event_format, e.event_date::text
		FROM sessions s
		JOIN events e ON e.id = s.event_id
		WHERE s.id = $1
	`
	var s model.Session
	var e model.Event
	err := r.pool.QueryRow(ctx, query, sessionID).Scan(
		&s.ID, &s.EventID, &s.SessionName, &s.SessionType, &s.SessionDate, &s.TotalLaps, &s.DataStatus, &s.DataError,
		&e.ID, &e.SeasonID, &e.CircuitID, &e.RoundNumber, &e.EventName, &e.OfficialName, &e.Country, &e.Location, &e.EventFormat, &e.EventDate,
	)
	if err != nil {
		if err == pgx.ErrNoRows {
			return nil, nil
		}
		return nil, fmt.Errorf("query session by id: %w", err)
	}
	s.Event = &e
	return &s, nil
}

// ---------------------------------------------------------------------
// Drivers & Teams
// ---------------------------------------------------------------------

func (r *Repository) GetDrivers(ctx context.Context) ([]model.Driver, error) {
	rows, err := r.pool.Query(ctx, `
		SELECT id, driver_number, broadcast_name, abbreviation, driver_id, first_name, last_name, full_name, country_code, headshot_url
		FROM drivers ORDER BY last_name ASC
	`)
	if err != nil {
		return nil, fmt.Errorf("query drivers: %w", err)
	}
	defer rows.Close()

	var drivers []model.Driver
	for rows.Next() {
		var d model.Driver
		if err := rows.Scan(&d.ID, &d.DriverNumber, &d.BroadcastName, &d.Abbreviation, &d.DriverID, &d.FirstName, &d.LastName, &d.FullName, &d.CountryCode, &d.HeadshotURL); err != nil {
			return nil, fmt.Errorf("scan driver: %w", err)
		}
		drivers = append(drivers, d)
	}
	return drivers, nil
}

func (r *Repository) GetTeams(ctx context.Context) ([]model.Team, error) {
	rows, err := r.pool.Query(ctx, "SELECT id, team_id, name, color, country FROM teams ORDER BY name ASC")
	if err != nil {
		return nil, fmt.Errorf("query teams: %w", err)
	}
	defer rows.Close()

	var teams []model.Team
	for rows.Next() {
		var t model.Team
		if err := rows.Scan(&t.ID, &t.TeamID, &t.Name, &t.Color, &t.Country); err != nil {
			return nil, fmt.Errorf("scan team: %w", err)
		}
		teams = append(teams, t)
	}
	return teams, nil
}

// ---------------------------------------------------------------------
// Session Results
// ---------------------------------------------------------------------

func (r *Repository) GetSessionResults(ctx context.Context, sessionID int) ([]model.SessionResult, error) {
	query := `
		SELECT sr.id, sr.session_id, sr.driver_id, sr.team_id, sr.position, sr.classified_position, 
		       sr.grid_position, sr.q1_time_ms, sr.q2_time_ms, sr.q3_time_ms, sr.race_time_ms, sr.status, sr.points,
		       d.id, d.driver_number, d.broadcast_name, d.abbreviation, d.driver_id, d.first_name, d.last_name, d.full_name, d.country_code, d.headshot_url,
		       t.id, t.team_id, t.name, t.color, t.country
		FROM session_results sr
		JOIN drivers d ON d.id = sr.driver_id
		LEFT JOIN teams t ON t.id = sr.team_id
		WHERE sr.session_id = $1
		ORDER BY sr.position ASC NULLS LAST
	`
	rows, err := r.pool.Query(ctx, query, sessionID)
	if err != nil {
		return nil, fmt.Errorf("query session results: %w", err)
	}
	defer rows.Close()

	var results []model.SessionResult
	for rows.Next() {
		var res model.SessionResult
		var d model.Driver
		var tID *int
		var tTeamID, tName, tColor, tCountry *string

		if err := rows.Scan(
			&res.ID, &res.SessionID, &res.DriverID, &res.TeamID, &res.Position, &res.ClassifiedPosition,
			&res.GridPosition, &res.Q1TimeMs, &res.Q2TimeMs, &res.Q3TimeMs, &res.RaceTimeMs, &res.Status, &res.Points,
			&d.ID, &d.DriverNumber, &d.BroadcastName, &d.Abbreviation, &d.DriverID, &d.FirstName, &d.LastName, &d.FullName, &d.CountryCode, &d.HeadshotURL,
			&tID, &tTeamID, &tName, &tColor, &tCountry,
		); err != nil {
			return nil, fmt.Errorf("scan session result: %w", err)
		}

		res.Driver = &d
		if tID != nil && tName != nil {
			res.Team = &model.Team{
				ID:      *tID,
				TeamID:  tTeamID,
				Name:    *tName,
				Color:   tColor,
				Country: tCountry,
			}
		}
		results = append(results, res)
	}
	return results, nil
}

// ---------------------------------------------------------------------
// Laps
// ---------------------------------------------------------------------

func (r *Repository) GetSessionLaps(ctx context.Context, sessionID int) ([]model.Lap, error) {
	query := `
		SELECT id, session_id, driver_id, lap_number, lap_time_ms, sector1_ms, sector2_ms, sector3_ms,
		       speed_i1, speed_i2, speed_fl, speed_st, compound, tyre_life, fresh_tyre, stint,
		       is_personal_best, is_deleted, deleted_reason, track_status, position
		FROM laps
		WHERE session_id = $1
		ORDER BY lap_number ASC, driver_id ASC
	`
	rows, err := r.pool.Query(ctx, query, sessionID)
	if err != nil {
		return nil, fmt.Errorf("query session laps: %w", err)
	}
	defer rows.Close()

	var laps []model.Lap
	for rows.Next() {
		var l model.Lap
		if err := rows.Scan(
			&l.ID, &l.SessionID, &l.DriverID, &l.LapNumber, &l.LapTimeMs, &l.Sector1Ms, &l.Sector2Ms, &l.Sector3Ms,
			&l.SpeedI1, &l.SpeedI2, &l.SpeedFL, &l.SpeedST, &l.Compound, &l.TyreLife, &l.FreshTyre, &l.Stint,
			&l.IsPersonalBest, &l.IsDeleted, &l.DeletedReason, &l.TrackStatus, &l.Position,
		); err != nil {
			return nil, fmt.Errorf("scan lap: %w", err)
		}
		laps = append(laps, l)
	}
	return laps, nil
}

// ---------------------------------------------------------------------
// Weather
// ---------------------------------------------------------------------

func (r *Repository) GetSessionWeather(ctx context.Context, sessionID int) ([]model.WeatherData, error) {
	query := `
		SELECT id, session_id, time_offset_ms, air_temp, track_temp, humidity, pressure, wind_speed, wind_direction, rainfall
		FROM weather_data
		WHERE session_id = $1
		ORDER BY time_offset_ms ASC
	`
	rows, err := r.pool.Query(ctx, query, sessionID)
	if err != nil {
		return nil, fmt.Errorf("query weather data: %w", err)
	}
	defer rows.Close()

	var list []model.WeatherData
	for rows.Next() {
		var w model.WeatherData
		if err := rows.Scan(&w.ID, &w.SessionID, &w.TimeOffsetMs, &w.AirTemp, &w.TrackTemp, &w.Humidity, &w.Pressure, &w.WindSpeed, &w.WindDirection, &w.Rainfall); err != nil {
			return nil, fmt.Errorf("scan weather: %w", err)
		}
		list = append(list, w)
	}
	return list, nil
}

// ---------------------------------------------------------------------
// Ingestion Jobs
// ---------------------------------------------------------------------

func (r *Repository) CreateIngestionJob(ctx context.Context, sessionID int) (*model.IngestionJob, error) {
	query := `
		INSERT INTO ingestion_jobs (session_id, status, progress, started_at)
		VALUES ($1, 'pending', 0, NOW())
		RETURNING id::text, session_id, status, progress, started_at
	`
	var job model.IngestionJob
	var startedAt time.Time
	err := r.pool.QueryRow(ctx, query, sessionID).Scan(&job.ID, &job.SessionID, &job.Status, &job.Progress, &startedAt)
	if err != nil {
		return nil, fmt.Errorf("create ingestion job: %w", err)
	}
	s := startedAt.Format(time.RFC3339)
	job.StartedAt = &s
	return &job, nil
}

func (r *Repository) GetLatestIngestionJobBySession(ctx context.Context, sessionID int) (*model.IngestionJob, error) {
	query := `
		SELECT id::text, session_id, status, progress, error_message, started_at, completed_at
		FROM ingestion_jobs
		WHERE session_id = $1
		ORDER BY started_at DESC
		LIMIT 1
	`
	var job model.IngestionJob
	var startedAt, completedAt *time.Time
	err := r.pool.QueryRow(ctx, query, sessionID).Scan(&job.ID, &job.SessionID, &job.Status, &job.Progress, &job.ErrorMessage, &startedAt, &completedAt)
	if err != nil {
		if err == pgx.ErrNoRows {
			return nil, nil
		}
		return nil, fmt.Errorf("query ingestion job: %w", err)
	}
	if startedAt != nil {
		s := startedAt.Format(time.RFC3339)
		job.StartedAt = &s
	}
	if completedAt != nil {
		s := completedAt.Format(time.RFC3339)
		job.CompletedAt = &s
	}
	return &job, nil
}

func (r *Repository) UpdateIngestionJob(ctx context.Context, id string, status string, progress int, errorMsg *string) error {
	var completedAt *time.Time
	if status == "completed" || status == "failed" {
		now := time.Now()
		completedAt = &now
	}
	_, err := r.pool.Exec(ctx, `
		UPDATE ingestion_jobs 
		SET status = $2, progress = $3, error_message = $4, completed_at = $5, updated_at = NOW()
		WHERE id = $1
	`, id, status, progress, errorMsg, completedAt)
	return err
}

// ---------------------------------------------------------------------
// Upserts & Data Loading
// ---------------------------------------------------------------------

func (r *Repository) UpsertSeason(ctx context.Context, year int) (int, error) {
	var id int
	query := `
		INSERT INTO seasons (year, created_at, updated_at)
		VALUES ($1, NOW(), NOW())
		ON CONFLICT (year) DO UPDATE SET updated_at = NOW()
		RETURNING id
	`
	err := r.pool.QueryRow(ctx, query, year).Scan(&id)
	return id, err
}

func (r *Repository) UpsertCircuit(ctx context.Context, c *model.Circuit) (int, error) {
	var id int
	query := `
		INSERT INTO circuits (circuit_key, name, country, location, latitude, longitude, track_length_m, num_corners, num_drs_zones, created_at, updated_at)
		VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, NOW(), NOW())
		ON CONFLICT (circuit_key) DO UPDATE SET 
			name = EXCLUDED.name,
			country = EXCLUDED.country,
			location = EXCLUDED.location,
			latitude = EXCLUDED.latitude,
			longitude = EXCLUDED.longitude,
			track_length_m = EXCLUDED.track_length_m,
			num_corners = EXCLUDED.num_corners,
			num_drs_zones = EXCLUDED.num_drs_zones,
			updated_at = NOW()
		RETURNING id
	`
	err := r.pool.QueryRow(ctx, query, c.CircuitKey, c.Name, c.Country, c.Location, c.Latitude, c.Longitude, c.TrackLengthM, c.NumCorners, c.NumDRSZones).Scan(&id)
	return id, err
}

func (r *Repository) UpsertEvent(ctx context.Context, e *model.Event) (int, error) {
	var id int
	query := `
		INSERT INTO events (season_id, circuit_id, round_number, event_name, official_name, country, location, event_format, event_date, created_at, updated_at)
		VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, NOW(), NOW())
		ON CONFLICT (season_id, round_number) DO UPDATE SET
			circuit_id = EXCLUDED.circuit_id,
			event_name = EXCLUDED.event_name,
			official_name = EXCLUDED.official_name,
			country = EXCLUDED.country,
			location = EXCLUDED.location,
			event_format = EXCLUDED.event_format,
			event_date = EXCLUDED.event_date,
			updated_at = NOW()
		RETURNING id
	`
	err := r.pool.QueryRow(ctx, query, e.SeasonID, e.CircuitID, e.RoundNumber, e.EventName, e.OfficialName, e.Country, e.Location, e.EventFormat, e.EventDate).Scan(&id)
	return id, err
}

func (r *Repository) UpsertSession(ctx context.Context, s *model.Session) (int, error) {
	var id int
	query := `
		INSERT INTO sessions (event_id, session_name, session_type, session_date, total_laps, data_status, data_error, created_at, updated_at)
		VALUES ($1, $2, $3, $4, $5, $6, $7, NOW(), NOW())
		ON CONFLICT (event_id, session_name) DO UPDATE SET
			session_type = EXCLUDED.session_type,
			session_date = EXCLUDED.session_date,
			total_laps = EXCLUDED.total_laps,
			data_status = EXCLUDED.data_status,
			data_error = EXCLUDED.data_error,
			updated_at = NOW()
		RETURNING id
	`
	err := r.pool.QueryRow(ctx, query, s.EventID, s.SessionName, s.SessionType, s.SessionDate, s.TotalLaps, s.DataStatus, s.DataError).Scan(&id)
	return id, err
}

func (r *Repository) UpsertDriver(ctx context.Context, d *model.Driver) (int, error) {
	var id int
	query := `
		INSERT INTO drivers (driver_number, broadcast_name, abbreviation, driver_id, first_name, last_name, full_name, country_code, headshot_url, created_at, updated_at)
		VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, NOW(), NOW())
		ON CONFLICT (abbreviation) DO UPDATE SET
			driver_number = EXCLUDED.driver_number,
			broadcast_name = EXCLUDED.broadcast_name,
			driver_id = EXCLUDED.driver_id,
			first_name = EXCLUDED.first_name,
			last_name = EXCLUDED.last_name,
			full_name = EXCLUDED.full_name,
			country_code = EXCLUDED.country_code,
			headshot_url = EXCLUDED.headshot_url,
			updated_at = NOW()
		RETURNING id
	`
	err := r.pool.QueryRow(ctx, query, d.DriverNumber, d.BroadcastName, d.Abbreviation, d.DriverID, d.FirstName, d.LastName, d.FullName, d.CountryCode, d.HeadshotURL).Scan(&id)
	return id, err
}

func (r *Repository) UpsertTeam(ctx context.Context, t *model.Team) (int, error) {
	var id int
	query := `
		INSERT INTO teams (team_id, name, color, country, created_at, updated_at)
		VALUES ($1, $2, $3, $4, NOW(), NOW())
		ON CONFLICT (name) DO UPDATE SET
			team_id = EXCLUDED.team_id,
			color = EXCLUDED.color,
			country = EXCLUDED.country,
			updated_at = NOW()
		RETURNING id
	`
	err := r.pool.QueryRow(ctx, query, t.TeamID, t.Name, t.Color, t.Country).Scan(&id)
	return id, err
}

func (r *Repository) InsertLaps(ctx context.Context, laps []model.Lap) error {
	if len(laps) == 0 {
		return nil
	}
	// Delete existing laps for this session to allow idempotency
	_, _ = r.pool.Exec(ctx, "DELETE FROM laps WHERE session_id = $1", laps[0].SessionID)

	for _, l := range laps {
		_, err := r.pool.Exec(ctx, `
			INSERT INTO laps (session_id, driver_id, lap_number, lap_time_ms, sector1_ms, sector2_ms, sector3_ms,
			                 speed_i1, speed_i2, speed_fl, speed_st, compound, tyre_life, fresh_tyre, stint,
			                 is_personal_best, is_deleted, deleted_reason, track_status, position, created_at)
			VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19, $20, NOW())
		`, l.SessionID, l.DriverID, l.LapNumber, l.LapTimeMs, l.Sector1Ms, l.Sector2Ms, l.Sector3Ms,
			l.SpeedI1, l.SpeedI2, l.SpeedFL, l.SpeedST, l.Compound, l.TyreLife, l.FreshTyre, l.Stint,
			l.IsPersonalBest, l.IsDeleted, l.DeletedReason, l.TrackStatus, l.Position)
		if err != nil {
			return fmt.Errorf("insert lap: %w", err)
		}
	}
	return nil
}

func (r *Repository) InsertSessionResults(ctx context.Context, results []model.SessionResult) error {
	if len(results) == 0 {
		return nil
	}
	_, _ = r.pool.Exec(ctx, "DELETE FROM session_results WHERE session_id = $1", results[0].SessionID)

	for _, res := range results {
		_, err := r.pool.Exec(ctx, `
			INSERT INTO session_results (session_id, driver_id, team_id, position, classified_position,
			                            grid_position, q1_time_ms, q2_time_ms, q3_time_ms, race_time_ms, status, points, created_at)
			VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, NOW())
		`, res.SessionID, res.DriverID, res.TeamID, res.Position, res.ClassifiedPosition,
			res.GridPosition, res.Q1TimeMs, res.Q2TimeMs, res.Q3TimeMs, res.RaceTimeMs, res.Status, res.Points)
		if err != nil {
			return fmt.Errorf("insert result: %w", err)
		}
	}
	return nil
}

func (r *Repository) InsertWeatherData(ctx context.Context, weather []model.WeatherData) error {
	if len(weather) == 0 {
		return nil
	}
	_, _ = r.pool.Exec(ctx, "DELETE FROM weather_data WHERE session_id = $1", weather[0].SessionID)

	for _, w := range weather {
		_, err := r.pool.Exec(ctx, `
			INSERT INTO weather_data (session_id, time_offset_ms, air_temp, track_temp, humidity, pressure, wind_speed, wind_direction, rainfall, created_at)
			VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, NOW())
		`, w.SessionID, w.TimeOffsetMs, w.AirTemp, w.TrackTemp, w.Humidity, w.Pressure, w.WindSpeed, w.WindDirection, w.Rainfall)
		if err != nil {
			return fmt.Errorf("insert weather: %w", err)
		}
	}
	return nil
}

func (r *Repository) UpsertTelemetryDataset(ctx context.Context, sessionID int, parquetPath string, totalPoints int64) error {
	query := `
		INSERT INTO telemetry_datasets (session_id, storage_provider, file_path, row_count, created_at)
		VALUES ($1, 'local', $2, $3, NOW())
		ON CONFLICT (session_id) DO UPDATE SET
			file_path = EXCLUDED.file_path,
			row_count = EXCLUDED.row_count
	`
	_, err := r.pool.Exec(ctx, query, sessionID, parquetPath, totalPoints)
	return err
}

func (r *Repository) UpdateSessionDataStatus(ctx context.Context, sessionID int, status string, dataError *string) error {
	_, err := r.pool.Exec(ctx, `
		UPDATE sessions
		SET data_status = $2, data_error = $3, updated_at = NOW()
		WHERE id = $1
	`, sessionID, status, dataError)
	return err
}

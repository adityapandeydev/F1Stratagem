package model

import (
	"time"
)

type Season struct {
	ID        int       `json:"id"`
	Year      int       `json:"year"`
	CreatedAt time.Time `json:"created_at"`
	UpdatedAt time.Time `json:"updated_at"`
}

type Circuit struct {
	ID           int      `json:"id"`
	CircuitKey   string   `json:"circuit_key"`
	Name         string   `json:"name"`
	Country      string   `json:"country"`
	Location     *string  `json:"location,omitempty"`
	Latitude     *float64 `json:"latitude,omitempty"`
	Longitude    *float64 `json:"longitude,omitempty"`
	TrackLengthM *int     `json:"track_length_m,omitempty"`
	NumCorners   *int     `json:"num_corners,omitempty"`
	NumDRSZones  *int     `json:"num_drs_zones,omitempty"`
}

type Event struct {
	ID           int      `json:"id"`
	SeasonID     int      `json:"season_id"`
	CircuitID    *int     `json:"circuit_id,omitempty"`
	RoundNumber  int      `json:"round_number"`
	EventName    string   `json:"event_name"`
	OfficialName *string  `json:"official_name,omitempty"`
	Country      *string  `json:"country,omitempty"`
	Location     *string  `json:"location,omitempty"`
	EventFormat  string   `json:"event_format"`
	EventDate    *string  `json:"event_date,omitempty"`
	Circuit      *Circuit `json:"circuit,omitempty"`
}

type Session struct {
	ID          int     `json:"id"`
	EventID     int     `json:"event_id"`
	SessionName string  `json:"session_name"`
	SessionType string  `json:"session_type"`
	SessionDate *string `json:"session_date,omitempty"`
	TotalLaps   *int    `json:"total_laps,omitempty"`
	DataStatus  string  `json:"data_status"`
	DataError   *string `json:"data_error,omitempty"`
	Event       *Event  `json:"event,omitempty"`
}

type Driver struct {
	ID            int     `json:"id"`
	DriverNumber  *string `json:"driver_number,omitempty"`
	BroadcastName *string `json:"broadcast_name,omitempty"`
	Abbreviation  string  `json:"abbreviation"`
	DriverID      *string `json:"driver_id,omitempty"`
	FirstName     string  `json:"first_name"`
	LastName      string  `json:"last_name"`
	FullName      *string `json:"full_name,omitempty"`
	CountryCode   *string `json:"country_code,omitempty"`
	HeadshotURL   *string `json:"headshot_url,omitempty"`
}

type Team struct {
	ID      int     `json:"id"`
	TeamID  *string `json:"team_id,omitempty"`
	Name    string  `json:"name"`
	Color   *string `json:"color,omitempty"`
	Country *string `json:"country,omitempty"`
}

type SessionResult struct {
	ID                 int     `json:"id"`
	SessionID          int     `json:"session_id"`
	DriverID           int     `json:"driver_id"`
	TeamID             *int    `json:"team_id,omitempty"`
	Position           *int    `json:"position,omitempty"`
	ClassifiedPosition *string `json:"classified_position,omitempty"`
	GridPosition       *int    `json:"grid_position,omitempty"`
	Q1TimeMs           *int    `json:"q1_time_ms,omitempty"`
	Q2TimeMs           *int    `json:"q2_time_ms,omitempty"`
	Q3TimeMs           *int    `json:"q3_time_ms,omitempty"`
	RaceTimeMs         *int64  `json:"race_time_ms,omitempty"`
	Status             *string `json:"status,omitempty"`
	Points             float32 `json:"points"`
	Driver             *Driver `json:"driver,omitempty"`
	Team               *Team   `json:"team,omitempty"`
}

type Lap struct {
	ID             int      `json:"id"`
	SessionID      int      `json:"session_id"`
	DriverID       int      `json:"driver_id"`
	LapNumber      int      `json:"lap_number"`
	LapTimeMs      *int     `json:"lap_time_ms,omitempty"`
	Sector1Ms      *int     `json:"sector1_ms,omitempty"`
	Sector2Ms      *int     `json:"sector2_ms,omitempty"`
	Sector3Ms      *int     `json:"sector3_ms,omitempty"`
	SpeedI1        *float32 `json:"speed_i1,omitempty"`
	SpeedI2        *float32 `json:"speed_i2,omitempty"`
	SpeedFL        *float32 `json:"speed_fl,omitempty"`
	SpeedST        *float32 `json:"speed_st,omitempty"`
	Compound       *string  `json:"compound,omitempty"`
	TyreLife       *float32 `json:"tyre_life,omitempty"`
	FreshTyre      *bool    `json:"fresh_tyre,omitempty"`
	Stint          *int     `json:"stint,omitempty"`
	IsPersonalBest bool     `json:"is_personal_best"`
	IsDeleted      bool     `json:"is_deleted"`
	DeletedReason  *string  `json:"deleted_reason,omitempty"`
	TrackStatus    *string  `json:"track_status,omitempty"`
	Position       *int     `json:"position,omitempty"`
}

type WeatherData struct {
	ID            int      `json:"id"`
	SessionID     int      `json:"session_id"`
	TimeOffsetMs  int64    `json:"time_offset_ms"`
	AirTemp       *float32 `json:"air_temp,omitempty"`
	TrackTemp     *float32 `json:"track_temp,omitempty"`
	Humidity      *float32 `json:"humidity,omitempty"`
	Pressure      *float32 `json:"pressure,omitempty"`
	WindSpeed     *float32 `json:"wind_speed,omitempty"`
	WindDirection *int     `json:"wind_direction,omitempty"`
	Rainfall      bool     `json:"rainfall"`
}

type TelemetryPoint struct {
	Distance float64 `json:"d"`
	Speed    float64 `json:"spd"`
	Throttle float64 `json:"thr"`
	Brake    int     `json:"brk"`
	RPM      float64 `json:"rpm"`
	Gear     int     `json:"gear"`
	DRS      int     `json:"drs"`
	X        float64 `json:"x"`
	Y        float64 `json:"y"`
	Z        float64 `json:"z"`
}

type TelemetryComparison struct {
	Driver1    Driver           `json:"driver1"`
	Driver2    Driver           `json:"driver2"`
	Lap1       int              `json:"lap1"`
	Lap2       int              `json:"lap2"`
	Telemetry1 []TelemetryPoint `json:"telemetry1"`
	Telemetry2 []TelemetryPoint `json:"telemetry2"`
	TimeDelta  []float64        `json:"time_delta"`
}

type IngestionJob struct {
	ID           string  `json:"id"`
	SessionID    int     `json:"session_id"`
	Status       string  `json:"status"`
	Progress     int     `json:"progress"`
	ErrorMessage *string `json:"error_message,omitempty"`
	StartedAt    *string `json:"started_at,omitempty"`
	CompletedAt  *string `json:"completed_at,omitempty"`
}
package service

import (
	"context"
	"encoding/json"
	"fmt"
	"log"

	"github.com/aditya/f1stratagem/api/internal/model"
	"github.com/aditya/f1stratagem/api/internal/repository"
	"github.com/aditya/f1stratagem/api/internal/worker"
)

type IngestionService struct {
	repo         *repository.Repository
	workerClient *worker.Client
}

func NewIngestionService(repo *repository.Repository, workerClient *worker.Client) *IngestionService {
	return &IngestionService{
		repo:         repo,
		workerClient: workerClient,
	}
}

// SyncSchedule fetches the calendar schedule from worker and populates DB
func (s *IngestionService) SyncSchedule(ctx context.Context, year int) error {
	log.Printf("[ingestion] Syncing schedule for year %d", year)
	seasonID, err := s.repo.UpsertSeason(ctx, year)
	if err != nil {
		return fmt.Errorf("upsert season: %w", err)
	}

	data, err := s.workerClient.GetSchedule(ctx, year)
	if err != nil {
		return fmt.Errorf("worker get schedule: %w", err)
	}

	eventsRaw, ok := data["events"].([]interface{})
	if !ok {
		return fmt.Errorf("invalid events array in worker response")
	}

	for _, eRaw := range eventsRaw {
		eMap, ok := eRaw.(map[string]interface{})
		if !ok {
			continue
		}

		roundNum := int(eMap["round_number"].(float64))
		eventName := eMap["event_name"].(string)
		eventFormat := "conventional"
		if f, ok := eMap["event_format"].(string); ok {
			eventFormat = f
		}
		var officialName, country, location, eventDate *string
		if str, ok := eMap["official_name"].(string); ok {
			officialName = &str
		}
		if str, ok := eMap["country"].(string); ok {
			country = &str
		}
		if str, ok := eMap["location"].(string); ok {
			location = &str
		}
		if str, ok := eMap["event_date"].(string); ok {
			eventDate = &str
		}

		// Circuit
		var circuitID *int
		if cRaw, ok := eMap["circuit"].(map[string]interface{}); ok {
			cKey := cRaw["circuit_key"].(string)
			cName := cRaw["name"].(string)
			cCountry := cRaw["country"].(string)
			var cLoc *string
			if l, ok := cRaw["location"].(string); ok {
				cLoc = &l
			}
			cModel := &model.Circuit{
				CircuitKey: cKey,
				Name:       cName,
				Country:    cCountry,
				Location:   cLoc,
			}
			cid, err := s.repo.UpsertCircuit(ctx, cModel)
			if err == nil {
				circuitID = &cid
			}
		}

		// Event
		eventModel := &model.Event{
			SeasonID:     seasonID,
			CircuitID:    circuitID,
			RoundNumber:  roundNum,
			EventName:    eventName,
			OfficialName: officialName,
			Country:      country,
			Location:     location,
			EventFormat:  eventFormat,
			EventDate:    eventDate,
		}
		eID, err := s.repo.UpsertEvent(ctx, eventModel)
		if err != nil {
			log.Printf("[ingestion] Error upserting event %s: %v", eventName, err)
			continue
		}

		// Sessions
		if sessionsRaw, ok := eMap["sessions"].([]interface{}); ok {
			for _, sRaw := range sessionsRaw {
				sMap, ok := sRaw.(map[string]interface{})
				if !ok {
					continue
				}
				sName := sMap["session_name"].(string)
				sType := sMap["session_type"].(string)
				var sDate *string
				if sd, ok := sMap["session_date"].(string); ok {
					sDate = &sd
				}

				sessionModel := &model.Session{
					EventID:     eID,
					SessionName: sName,
					SessionType: sType,
					SessionDate: sDate,
					DataStatus:  "pending",
				}
				_, err := s.repo.UpsertSession(ctx, sessionModel)
				if err != nil {
					log.Printf("[ingestion] Error upserting session %s: %v", sName, err)
				}
			}
		}
	}

	log.Printf("[ingestion] Successfully synced schedule for year %d", year)
	return nil
}

// TriggerIngestion begins extracting session data asynchronously
func (s *IngestionService) TriggerIngestion(ctx context.Context, sessionID int) (*model.IngestionJob, error) {
	session, err := s.repo.GetSessionByID(ctx, sessionID)
	if err != nil || session == nil {
		return nil, fmt.Errorf("session not found: %d", sessionID)
	}

	// Create job
	job, err := s.repo.CreateIngestionJob(ctx, sessionID)
	if err != nil {
		return nil, fmt.Errorf("create job: %w", err)
	}

	// Mark session processing
	_ = s.repo.UpdateSessionDataStatus(ctx, sessionID, "processing", nil)

	// Launch async extraction
	go s.processSessionIngestion(job.ID, session)

	return job, nil
}

func (s *IngestionService) processSessionIngestion(jobID string, session *model.Session) {
	ctx := context.Background()
	log.Printf("[ingestion] Starting worker extraction for session %d (%s)", session.ID, session.SessionName)

	_ = s.repo.UpdateIngestionJob(ctx, jobID, "processing", 15, nil)

	// FastF1 session identifier mapping
	sIdent := session.SessionName
	if session.SessionType == "race" {
		sIdent = "R"
	} else if session.SessionType == "qualifying" {
		sIdent = "Q"
	} else if session.SessionType == "sprint" {
		sIdent = "S"
	}

	// Round number & Year
	roundNum := 1
	year := 2024
	if session.Event != nil {
		roundNum = session.Event.RoundNumber
		// lookup season year
		seasons, err := s.repo.GetSeasons(ctx)
		if err == nil {
			for _, sn := range seasons {
				if sn.ID == session.Event.SeasonID {
					year = sn.Year
					break
				}
			}
		}
	}

	resp, err := s.workerClient.ExtractSession(ctx, year, roundNum, sIdent)
	if err != nil {
		errMsg := err.Error()
		log.Printf("[ingestion] Worker failed for session %d: %v", session.ID, err)
		_ = s.repo.UpdateIngestionJob(ctx, jobID, "failed", 0, &errMsg)
		_ = s.repo.UpdateSessionDataStatus(ctx, session.ID, "failed", &errMsg)
		return
	}

	_ = s.repo.UpdateIngestionJob(ctx, jobID, "processing", 50, nil)

	data := resp.Data

	// 1. Ingest drivers & teams
	driverIDMap := make(map[string]int)
	teamIDMap := make(map[string]int)

	if teamsRaw, ok := data["teams"].([]interface{}); ok {
		for _, tRaw := range teamsRaw {
			b, _ := json.Marshal(tRaw)
			var t model.Team
			if err := json.Unmarshal(b, &t); err == nil {
				tid, err := s.repo.UpsertTeam(ctx, &t)
				if err == nil {
					teamIDMap[t.Name] = tid
				}
			}
		}
	}

	if driversRaw, ok := data["drivers"].([]interface{}); ok {
		for _, dRaw := range driversRaw {
			b, _ := json.Marshal(dRaw)
			var d model.Driver
			if err := json.Unmarshal(b, &d); err == nil {
				did, err := s.repo.UpsertDriver(ctx, &d)
				if err == nil {
					driverIDMap[d.Abbreviation] = did
				}
			}
		}
	}

	_ = s.repo.UpdateIngestionJob(ctx, jobID, "processing", 70, nil)

	// 2. Ingest Session Results
	if resRaw, ok := data["results"].([]interface{}); ok {
		var results []model.SessionResult
		for _, rItem := range resRaw {
			rMap, ok := rItem.(map[string]interface{})
			if !ok {
				continue
			}
			abbr := rMap["driver_abbreviation"].(string)
			did := driverIDMap[abbr]
			if did == 0 {
				continue
			}

			var teamID *int
			if tm, ok := rMap["team_name"].(string); ok {
				if tid, found := teamIDMap[tm]; found {
					teamID = &tid
				}
			}

			res := model.SessionResult{
				SessionID: session.ID,
				DriverID:  did,
				TeamID:    teamID,
			}
			if pos, ok := rMap["position"].(float64); ok {
				p := int(pos)
				res.Position = &p
			}
			if grid, ok := rMap["grid_position"].(float64); ok {
				g := int(grid)
				res.GridPosition = &g
			}
			if pts, ok := rMap["points"].(float64); ok {
				res.Points = float32(pts)
			}
			if st, ok := rMap["status"].(string); ok {
				res.Status = &st
			}
			if q1, ok := rMap["q1_time_ms"].(float64); ok {
				v := int(q1)
				res.Q1TimeMs = &v
			}
			if q2, ok := rMap["q2_time_ms"].(float64); ok {
				v := int(q2)
				res.Q2TimeMs = &v
			}
			if q3, ok := rMap["q3_time_ms"].(float64); ok {
				v := int(q3)
				res.Q3TimeMs = &v
			}
			if rt, ok := rMap["race_time_ms"].(float64); ok {
				v := int64(rt)
				res.RaceTimeMs = &v
			}
			results = append(results, res)
		}
		_ = s.repo.InsertSessionResults(ctx, results)
	}

	_ = s.repo.UpdateIngestionJob(ctx, jobID, "processing", 85, nil)

	// 3. Ingest Laps
	if lapsRaw, ok := data["laps"].([]interface{}); ok {
		var laps []model.Lap
		for _, lItem := range lapsRaw {
			b, _ := json.Marshal(lItem)
			var l model.Lap
			if err := json.Unmarshal(b, &l); err == nil {
				if lMap, ok := lItem.(map[string]interface{}); ok {
					if abbr, ok := lMap["driver_abbreviation"].(string); ok {
						l.DriverID = driverIDMap[abbr]
					}
				}
				l.SessionID = session.ID
				if l.DriverID > 0 {
					laps = append(laps, l)
				}
			}
		}
		_ = s.repo.InsertLaps(ctx, laps)
	}

	// 4. Ingest Weather
	if weatherRaw, ok := data["weather"].([]interface{}); ok {
		var wList []model.WeatherData
		for _, wItem := range weatherRaw {
			b, _ := json.Marshal(wItem)
			var w model.WeatherData
			if err := json.Unmarshal(b, &w); err == nil {
				w.SessionID = session.ID
				wList = append(wList, w)
			}
		}
		_ = s.repo.InsertWeatherData(ctx, wList)
	}

	// 5. Ingest Telemetry Dataset
	if telFile, ok := data["telemetry_file"].(string); ok && telFile != "" {
		var rowCount int64
		if rc, ok := data["telemetry_row_count"].(float64); ok {
			rowCount = int64(rc)
		}
		_ = s.repo.UpsertTelemetryDataset(ctx, session.ID, telFile, rowCount)
	}

	// Mark success
	_ = s.repo.UpdateIngestionJob(ctx, jobID, "completed", 100, nil)
	_ = s.repo.UpdateSessionDataStatus(ctx, session.ID, "ready", nil)
	log.Printf("[ingestion] Successfully ingested session %d", session.ID)
}

func (s *IngestionService) CompareMultiTelemetry(ctx context.Context, sessionID int, drivers []worker.DriverLapRequest) (map[string]interface{}, error) {
	session, err := s.repo.GetSessionByID(ctx, sessionID)
	if err != nil || session == nil {
		return nil, fmt.Errorf("session not found: %d", sessionID)
	}

	sIdent := session.SessionName
	if session.SessionType == "race" {
		sIdent = "R"
	} else if session.SessionType == "qualifying" {
		sIdent = "Q"
	}

	roundNum := 1
	year := 2024
	if session.Event != nil {
		roundNum = session.Event.RoundNumber
		seasons, err := s.repo.GetSeasons(ctx)
		if err == nil {
			for _, sn := range seasons {
				if sn.ID == session.Event.SeasonID {
					year = sn.Year
					break
				}
			}
		}
	}

	return s.workerClient.CompareMultiTelemetry(ctx, year, roundNum, sIdent, drivers)
}

func (s *IngestionService) CompareTelemetry(ctx context.Context, sessionID int, d1 string, lap1 int, d2 string, lap2 int) (map[string]interface{}, error) {
	return s.CompareMultiTelemetry(ctx, sessionID, []worker.DriverLapRequest{
		{Driver: d1, Lap: lap1},
		{Driver: d2, Lap: lap2},
	})
}

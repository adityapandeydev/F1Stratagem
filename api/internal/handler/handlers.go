package handler

import (
	"encoding/json"
	"github.com/aditya/f1stratagem/api/internal/model"
	"net/http"
	"strconv"
	"time"

	"github.com/aditya/f1stratagem/api/internal/repository"
	"github.com/aditya/f1stratagem/api/internal/service"
)

type Handler struct {
	repo         *repository.Repository
	ingestionSvc *service.IngestionService
}

func New(repo *repository.Repository, ingestionSvc *service.IngestionService) *Handler {
	return &Handler{
		repo:         repo,
		ingestionSvc: ingestionSvc,
	}
}

func writeJSON(w http.ResponseWriter, status int, data interface{}) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(data)
}

func writeError(w http.ResponseWriter, status int, message string) {
	writeJSON(w, status, map[string]string{"error": message})
}

func (h *Handler) HealthCheck(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, http.StatusOK, map[string]interface{}{
		"status":  "healthy",
		"service": "f1stratagem-api",
		"time":    time.Now().UTC().Format(time.RFC3339),
	})
}

func (h *Handler) GetSeasons(w http.ResponseWriter, r *http.Request) {
	seasons, err := h.repo.GetSeasons(r.Context())
	if err != nil {
		writeError(w, http.StatusInternalServerError, err.Error())
		return
	}
	if seasons == nil {
		seasons = []model.Season{}
	}
	writeJSON(w, http.StatusOK, seasons)
}

func (h *Handler) GetSeasonByYear(w http.ResponseWriter, r *http.Request) {
	yearStr := r.PathValue("year")
	year, err := strconv.Atoi(yearStr)
	if err != nil {
		writeError(w, http.StatusBadRequest, "invalid year")
		return
	}

	season, err := h.repo.GetSeasonByYear(r.Context(), year)
	if err != nil {
		writeError(w, http.StatusInternalServerError, err.Error())
		return
	}
	if season == nil {
		writeError(w, http.StatusNotFound, "season not found")
		return
	}
	writeJSON(w, http.StatusOK, season)
}

func (h *Handler) SyncSeasonSchedule(w http.ResponseWriter, r *http.Request) {
	yearStr := r.PathValue("year")
	year, err := strconv.Atoi(yearStr)
	if err != nil {
		writeError(w, http.StatusBadRequest, "invalid year")
		return
	}

	if err := h.ingestionSvc.SyncSchedule(r.Context(), year); err != nil {
		writeError(w, http.StatusInternalServerError, err.Error())
		return
	}

	writeJSON(w, http.StatusOK, map[string]string{
		"status":  "success",
		"message": "Schedule synchronized successfully",
	})
}

func (h *Handler) GetEventsBySeason(w http.ResponseWriter, r *http.Request) {
	yearStr := r.PathValue("year")
	year, err := strconv.Atoi(yearStr)
	if err != nil {
		writeError(w, http.StatusBadRequest, "invalid year")
		return
	}

	events, err := h.repo.GetEventsBySeasonYear(r.Context(), year)
	if err != nil {
		writeError(w, http.StatusInternalServerError, err.Error())
		return
	}
	if events == nil {
		events = []model.Event{}
	}
	writeJSON(w, http.StatusOK, events)
}

func (h *Handler) GetSessionsByEvent(w http.ResponseWriter, r *http.Request) {
	eventIDStr := r.PathValue("eventId")
	eventID, err := strconv.Atoi(eventIDStr)
	if err != nil {
		writeError(w, http.StatusBadRequest, "invalid event id")
		return
	}

	sessions, err := h.repo.GetSessionsByEventID(r.Context(), eventID)
	if err != nil {
		writeError(w, http.StatusInternalServerError, err.Error())
		return
	}
	if sessions == nil {
		sessions = []model.Session{}
	}
	writeJSON(w, http.StatusOK, sessions)
}

func (h *Handler) GetSession(w http.ResponseWriter, r *http.Request) {
	sIDStr := r.PathValue("sessionId")
	sID, err := strconv.Atoi(sIDStr)
	if err != nil {
		writeError(w, http.StatusBadRequest, "invalid session id")
		return
	}

	session, err := h.repo.GetSessionByID(r.Context(), sID)
	if err != nil {
		writeError(w, http.StatusInternalServerError, err.Error())
		return
	}
	if session == nil {
		writeError(w, http.StatusNotFound, "session not found")
		return
	}
	writeJSON(w, http.StatusOK, session)
}

func (h *Handler) GetSessionResults(w http.ResponseWriter, r *http.Request) {
	sIDStr := r.PathValue("sessionId")
	sID, err := strconv.Atoi(sIDStr)
	if err != nil {
		writeError(w, http.StatusBadRequest, "invalid session id")
		return
	}

	results, err := h.repo.GetSessionResults(r.Context(), sID)
	if err != nil {
		writeError(w, http.StatusInternalServerError, err.Error())
		return
	}
	if results == nil {
		results = []model.SessionResult{}
	}
	writeJSON(w, http.StatusOK, results)
}

func (h *Handler) GetSessionLaps(w http.ResponseWriter, r *http.Request) {
	sIDStr := r.PathValue("sessionId")
	sID, err := strconv.Atoi(sIDStr)
	if err != nil {
		writeError(w, http.StatusBadRequest, "invalid session id")
		return
	}

	laps, err := h.repo.GetSessionLaps(r.Context(), sID)
	if err != nil {
		writeError(w, http.StatusInternalServerError, err.Error())
		return
	}
	if laps == nil {
		laps = []model.Lap{}
	}
	writeJSON(w, http.StatusOK, laps)
}

func (h *Handler) GetSessionWeather(w http.ResponseWriter, r *http.Request) {
	sIDStr := r.PathValue("sessionId")
	sID, err := strconv.Atoi(sIDStr)
	if err != nil {
		writeError(w, http.StatusBadRequest, "invalid session id")
		return
	}

	weather, err := h.repo.GetSessionWeather(r.Context(), sID)
	if err != nil {
		writeError(w, http.StatusInternalServerError, err.Error())
		return
	}
	if weather == nil {
		weather = []model.WeatherData{}
	}
	writeJSON(w, http.StatusOK, weather)
}

func (h *Handler) GetSessionStints(w http.ResponseWriter, r *http.Request) {
	sIDStr := r.PathValue("sessionId")
	sID, err := strconv.Atoi(sIDStr)
	if err != nil {
		writeError(w, http.StatusBadRequest, "invalid session id")
		return
	}

	laps, err := h.repo.GetSessionLaps(r.Context(), sID)
	if err != nil {
		writeError(w, http.StatusInternalServerError, err.Error())
		return
	}

	// Compute stint summaries from laps
	type StintSummary struct {
		DriverID  int     `json:"driver_id"`
		Stint     int     `json:"stint"`
		Compound  string  `json:"compound"`
		StartLap  int     `json:"start_lap"`
		EndLap    int     `json:"end_lap"`
		LapCount  int     `json:"lap_count"`
		FreshTyre bool    `json:"fresh_tyre"`
	}

	var stints []StintSummary
	stintMap := make(map[string]*StintSummary)

	for _, l := range laps {
		if l.Stint == nil {
			continue
		}
		key := strconv.Itoa(l.DriverID) + "_" + strconv.Itoa(*l.Stint)
		compound := ""
		if l.Compound != nil {
			compound = *l.Compound
		}
		fresh := false
		if l.FreshTyre != nil {
			fresh = *l.FreshTyre
		}

		if s, found := stintMap[key]; found {
			if l.LapNumber > s.EndLap {
				s.EndLap = l.LapNumber
			}
			if l.LapNumber < s.StartLap {
				s.StartLap = l.LapNumber
			}
			s.LapCount++
		} else {
			stintMap[key] = &StintSummary{
				DriverID:  l.DriverID,
				Stint:     *l.Stint,
				Compound:  compound,
				StartLap:  l.LapNumber,
				EndLap:    l.LapNumber,
				LapCount:  1,
				FreshTyre: fresh,
			}
		}
	}

	for _, s := range stintMap {
		stints = append(stints, *s)
	}

	writeJSON(w, http.StatusOK, stints)
}

func (h *Handler) TriggerIngestion(w http.ResponseWriter, r *http.Request) {
	sIDStr := r.PathValue("sessionId")
	sID, err := strconv.Atoi(sIDStr)
	if err != nil {
		writeError(w, http.StatusBadRequest, "invalid session id")
		return
	}

	job, err := h.ingestionSvc.TriggerIngestion(r.Context(), sID)
	if err != nil {
		writeError(w, http.StatusInternalServerError, err.Error())
		return
	}
	writeJSON(w, http.StatusAccepted, job)
}

func (h *Handler) GetIngestionStatus(w http.ResponseWriter, r *http.Request) {
	sIDStr := r.PathValue("sessionId")
	sID, err := strconv.Atoi(sIDStr)
	if err != nil {
		writeError(w, http.StatusBadRequest, "invalid session id")
		return
	}

	job, err := h.repo.GetLatestIngestionJobBySession(r.Context(), sID)
	if err != nil {
		writeError(w, http.StatusInternalServerError, err.Error())
		return
	}
	if job == nil {
		writeJSON(w, http.StatusOK, map[string]string{"status": "not_started"})
		return
	}
	writeJSON(w, http.StatusOK, job)
}

func (h *Handler) GetTelemetryComparison(w http.ResponseWriter, r *http.Request) {
	sIDStr := r.PathValue("sessionId")
	sID, err := strconv.Atoi(sIDStr)
	if err != nil {
		writeError(w, http.StatusBadRequest, "invalid session id")
		return
	}

	q := r.URL.Query()
	d1 := q.Get("driver1")
	d2 := q.Get("driver2")
	lap1, _ := strconv.Atoi(q.Get("lap1"))
	lap2, _ := strconv.Atoi(q.Get("lap2"))

	if d1 == "" || d2 == "" {
		writeError(w, http.StatusBadRequest, "driver1 and driver2 query params are required")
		return
	}

	data, err := h.ingestionSvc.CompareTelemetry(r.Context(), sID, d1, lap1, d2, lap2)
	if err != nil {
		writeError(w, http.StatusInternalServerError, err.Error())
		return
	}
	writeJSON(w, http.StatusOK, data)
}

func (h *Handler) GetDrivers(w http.ResponseWriter, r *http.Request) {
	drivers, err := h.repo.GetDrivers(r.Context())
	if err != nil {
		writeError(w, http.StatusInternalServerError, err.Error())
		return
	}
	if drivers == nil {
		drivers = []model.Driver{}
	}
	writeJSON(w, http.StatusOK, drivers)
}

func (h *Handler) GetTeams(w http.ResponseWriter, r *http.Request) {
	teams, err := h.repo.GetTeams(r.Context())
	if err != nil {
		writeError(w, http.StatusInternalServerError, err.Error())
		return
	}
	if teams == nil {
		teams = []model.Team{}
	}
	writeJSON(w, http.StatusOK, teams)
}

func (h *Handler) GetCircuits(w http.ResponseWriter, r *http.Request) {
	circuits, err := h.repo.GetCircuits(r.Context())
	if err != nil {
		writeError(w, http.StatusInternalServerError, err.Error())
		return
	}
	if circuits == nil {
		circuits = []model.Circuit{}
	}
	writeJSON(w, http.StatusOK, circuits)
}


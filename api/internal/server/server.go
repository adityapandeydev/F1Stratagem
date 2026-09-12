package server

import (
	"context"
	"fmt"
	"log"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/aditya/f1stratagem/api/internal/config"
	"github.com/aditya/f1stratagem/api/internal/handler"
	"github.com/aditya/f1stratagem/api/internal/middleware"
)

type Server struct {
	cfg    *config.Config
	router http.Handler
}

func New(cfg *config.Config, h *handler.Handler) (*Server, error) {
	mux := http.NewServeMux()

	// Health check
	mux.HandleFunc("GET /health", h.HealthCheck)

	// API v1 routes
	mux.HandleFunc("GET /api/v1/seasons", h.GetSeasons)
	mux.HandleFunc("POST /api/v1/seasons/{year}/sync", h.SyncSeasonSchedule)
	mux.HandleFunc("GET /api/v1/seasons/{year}", h.GetSeasonByYear)
	mux.HandleFunc("GET /api/v1/seasons/{year}/events", h.GetEventsBySeason)
	mux.HandleFunc("GET /api/v1/events/{eventId}/sessions", h.GetSessionsByEvent)
	mux.HandleFunc("GET /api/v1/sessions/{sessionId}", h.GetSession)
	mux.HandleFunc("GET /api/v1/sessions/{sessionId}/results", h.GetSessionResults)
	mux.HandleFunc("GET /api/v1/sessions/{sessionId}/laps", h.GetSessionLaps)
	mux.HandleFunc("GET /api/v1/sessions/{sessionId}/telemetry", h.GetTelemetryComparison)
	mux.HandleFunc("GET /api/v1/sessions/{sessionId}/weather", h.GetSessionWeather)
	mux.HandleFunc("GET /api/v1/sessions/{sessionId}/stints", h.GetSessionStints)
	mux.HandleFunc("POST /api/v1/sessions/{sessionId}/ingest", h.TriggerIngestion)
	mux.HandleFunc("GET /api/v1/sessions/{sessionId}/ingest/status", h.GetIngestionStatus)
	mux.HandleFunc("GET /api/v1/drivers", h.GetDrivers)
	mux.HandleFunc("GET /api/v1/teams", h.GetTeams)
	mux.HandleFunc("GET /api/v1/circuits", h.GetCircuits)

	// Apply middleware
	routes := middleware.CORS(middleware.Logger(middleware.Recover(mux)))

	return &Server{cfg: cfg, router: routes}, nil
}

func (s *Server) Start() error {
	addr := fmt.Sprintf("%s:%s", s.cfg.APIHost, s.cfg.APIPort)
	httpServer := &http.Server{
		Addr:         addr,
		Handler:      s.router,
		ReadTimeout:  15 * time.Second,
		WriteTimeout: 60 * time.Second,
		IdleTimeout:  120 * time.Second,
	}

	// Graceful shutdown channel
	shutdown := make(chan os.Signal, 1)
	signal.Notify(shutdown, os.Interrupt, syscall.SIGTERM)

	serverErrors := make(chan error, 1)
	go func() {
		log.Printf("[server] Starting F1Stratagem API on %s", addr)
		serverErrors <- httpServer.ListenAndServe()
	}()

	select {
	case err := <-serverErrors:
		return fmt.Errorf("server error: %w", err)
	case sig := <-shutdown:
		log.Printf("[server] Shutdown signal received: %v", sig)
		ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer cancel()
		return httpServer.Shutdown(ctx)
	}
}

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

func New(cfg *config.Config) (*Server, error) {
	mux := http.NewServeMux()

	// Health check
	mux.HandleFunc("GET /health", handler.HealthCheck)

	// API v1 routes
	mux.HandleFunc("GET /api/v1/seasons", handler.GetSeasons)
	mux.HandleFunc("GET /api/v1/seasons/{year}", handler.GetSeasonByYear)
	mux.HandleFunc("GET /api/v1/seasons/{year}/events", handler.GetEventsBySeason)
	mux.HandleFunc("GET /api/v1/events/{eventId}/sessions", handler.GetSessionsByEvent)
	mux.HandleFunc("GET /api/v1/sessions/{sessionId}", handler.GetSession)
	mux.HandleFunc("GET /api/v1/sessions/{sessionId}/results", handler.GetSessionResults)
	mux.HandleFunc("GET /api/v1/sessions/{sessionId}/laps", handler.GetSessionLaps)
	mux.HandleFunc("GET /api/v1/sessions/{sessionId}/telemetry", handler.GetTelemetryComparison)
	mux.HandleFunc("GET /api/v1/sessions/{sessionId}/weather", handler.GetSessionWeather)
	mux.HandleFunc("GET /api/v1/sessions/{sessionId}/stints", handler.GetSessionStints)
	mux.HandleFunc("POST /api/v1/sessions/{sessionId}/ingest", handler.TriggerIngestion)
	mux.HandleFunc("GET /api/v1/sessions/{sessionId}/ingest/status", handler.GetIngestionStatus)
	mux.HandleFunc("GET /api/v1/drivers", handler.GetDrivers)
	mux.HandleFunc("GET /api/v1/teams", handler.GetTeams)
	mux.HandleFunc("GET /api/v1/circuits", handler.GetCircuits)

	// Apply middleware
	h := middleware.CORS(middleware.Logger(middleware.Recover(mux)))

	return &Server{cfg: cfg, router: h}, nil
}

func (s *Server) Start(addr string) error {
	srv := &http.Server{
		Addr:         addr,
		Handler:      s.router,
		ReadTimeout:  15 * time.Second,
		WriteTimeout: 60 * time.Second,
		IdleTimeout:  60 * time.Second,
	}

	// Graceful shutdown
	done := make(chan os.Signal, 1)
	signal.Notify(done, os.Interrupt, syscall.SIGINT, syscall.SIGTERM)

	go func() {
		if err := srv.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			log.Fatalf("listen: %s\n", err)
		}
	}()

	fmt.Println("  Press Ctrl+C to stop")

	<-done
	log.Println("Shutting down server...")

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	return srv.Shutdown(ctx)
}

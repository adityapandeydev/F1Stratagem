package main

import (
	"context"
	"fmt"
	"log"
	"os"
	"time"

	"github.com/aditya/f1stratagem/api/internal/config"
	"github.com/aditya/f1stratagem/api/internal/handler"
	"github.com/aditya/f1stratagem/api/internal/repository"
	"github.com/aditya/f1stratagem/api/internal/server"
	"github.com/aditya/f1stratagem/api/internal/service"
	"github.com/aditya/f1stratagem/api/internal/storage"
	"github.com/aditya/f1stratagem/api/internal/worker"
)

func main() {
	cfg, err := config.Load()
	if err != nil {
		log.Fatalf("Failed to load config: %v", err)
	}

	// Connect to PostgreSQL
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	db, err := storage.NewPostgresPool(ctx, cfg.DatabaseURL)
	if err != nil {
		log.Fatalf("Failed to connect to PostgreSQL: %v", err)
	}
	defer db.Close()

	// Initialize layers
	repo := repository.New(db.Pool)
	workerClient := worker.NewClient(cfg.WorkerURL)
	ingestionSvc := service.NewIngestionService(repo, workerClient)
	h := handler.New(repo, ingestionSvc)

	srv, err := server.New(cfg, h)
	if err != nil {
		log.Fatalf("Failed to create server: %v", err)
	}

	addr := fmt.Sprintf("%s:%s", cfg.APIHost, cfg.APIPort)
	fmt.Fprintf(os.Stdout, "\n  F1Stratagem API\n  Listening on %s\n\n", addr)

	if err := srv.Start(); err != nil {
		log.Fatalf("Server error: %v", err)
	}
}

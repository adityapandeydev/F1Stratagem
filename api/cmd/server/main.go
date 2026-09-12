package main

import (
	"fmt"
	"log"
	"os"

	"github.com/aditya/f1stratagem/api/internal/config"
	"github.com/aditya/f1stratagem/api/internal/server"
)

func main() {
	cfg, err := config.Load()
	if err != nil {
		log.Fatalf("Failed to load config: %v", err)
	}

	srv, err := server.New(cfg)
	if err != nil {
		log.Fatalf("Failed to create server: %v", err)
	}

	addr := fmt.Sprintf("%s:%s", cfg.APIHost, cfg.APIPort)
	fmt.Fprintf(os.Stdout, "\n  F1Stratagem API\n  Listening on %s\n\n", addr)

	if err := srv.Start(addr); err != nil {
		log.Fatalf("Server error: %v", err)
	}
}

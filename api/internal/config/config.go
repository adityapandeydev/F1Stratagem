package config

import (
	"fmt"
	"log"
	"os"

	"github.com/joho/godotenv"
)

type Config struct {
	// Database
	DatabaseURL string
	DBHost      string
	DBPort      string
	DBUser      string
	DBPassword  string
	DBName      string

	// API
	APIPort string
	APIHost string

	// Analysis service
	WorkerURL string

	// Storage
	StorageProvider string

	// Logging
	LogLevel string
}

func Load() (*Config, error) {
	// The repository's .env, found from the API directory or the repository root.
	// godotenv.Load stops at the first missing file, so load only one that exists.
	for _, path := range []string{".env", "../.env", "../../.env"} {
		if _, err := os.Stat(path); err == nil {
			if err := godotenv.Load(path); err != nil {
				return nil, fmt.Errorf("load %s: %w", path, err)
			}
			break
		}
	}

	cfg := &Config{
		DatabaseURL:     getEnv("DATABASE_URL", ""),
		DBHost:          getEnv("DB_HOST", "localhost"),
		DBPort:          getEnv("DB_PORT", "5432"),
		DBUser:          getEnv("DB_USER", "postgres"),
		DBPassword:      getEnv("DB_PASSWORD", ""),
		DBName:          getEnv("DB_NAME", "f1"),
		APIPort:         getEnv("API_PORT", "8080"),
		APIHost:         getEnv("API_HOST", "0.0.0.0"),
		WorkerURL:       getEnv("WORKER_URL", "http://localhost:8081"),
		StorageProvider: getEnv("STORAGE_PROVIDER", "local"),
		LogLevel:        getEnv("LOG_LEVEL", "debug"),
	}

	if cfg.DatabaseURL == "" {
		return nil, fmt.Errorf("DATABASE_URL is not set; copy .env.example to .env")
	}

	log.Printf("[config] Loaded configuration with DB host: %s, port: %s, db: %s", cfg.DBHost, cfg.DBPort, cfg.DBName)
	return cfg, nil
}

func getEnv(key, fallback string) string {
	if val, ok := os.LookupEnv(key); ok && val != "" {
		return val
	}
	return fallback
}

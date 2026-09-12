package config

import (
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

	// Worker
	WorkerURL string

	// Storage
	TelemetryStoragePath string
	StorageProvider      string

	// FastF1
	FastF1CacheDir string

	// Logging
	LogLevel string
}

func Load() (*Config, error) {
	// Attempt to load .env from several likely locations
	_ = godotenv.Load(".env", "../.env", "../../.env")

	cfg := &Config{
		DatabaseURL:          getEnv("DATABASE_URL", "postgres://postgres:6243@localhost:5432/f1?sslmode=disable"),
		DBHost:               getEnv("DB_HOST", "localhost"),
		DBPort:               getEnv("DB_PORT", "5432"),
		DBUser:               getEnv("DB_USER", "postgres"),
		DBPassword:           getEnv("DB_PASSWORD", "6243"),
		DBName:               getEnv("DB_NAME", "f1"),
		APIPort:              getEnv("API_PORT", "8080"),
		APIHost:              getEnv("API_HOST", "0.0.0.0"),
		WorkerURL:            getEnv("WORKER_URL", "http://localhost:8081"),
		TelemetryStoragePath: getEnv("TELEMETRY_STORAGE_PATH", "./data/telemetry"),
		StorageProvider:      getEnv("STORAGE_PROVIDER", "local"),
		FastF1CacheDir:       getEnv("FASTF1_CACHE_DIR", "./data/fastf1_cache"),
		LogLevel:             getEnv("LOG_LEVEL", "debug"),
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

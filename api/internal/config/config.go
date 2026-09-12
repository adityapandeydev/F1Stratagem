package config

import (
	"os"
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
	cfg := &Config{
		DatabaseURL:          getEnv("DATABASE_URL", "postgres://f1stratagem:f1stratagem@localhost:5432/f1stratagem?sslmode=disable"),
		DBHost:               getEnv("DB_HOST", "localhost"),
		DBPort:               getEnv("DB_PORT", "5432"),
		DBUser:               getEnv("DB_USER", "f1stratagem"),
		DBPassword:           getEnv("DB_PASSWORD", "f1stratagem"),
		DBName:               getEnv("DB_NAME", "f1stratagem"),
		APIPort:              getEnv("API_PORT", "8080"),
		APIHost:              getEnv("API_HOST", "0.0.0.0"),
		WorkerURL:            getEnv("WORKER_URL", "http://localhost:8081"),
		TelemetryStoragePath: getEnv("TELEMETRY_STORAGE_PATH", "./data/telemetry"),
		StorageProvider:      getEnv("STORAGE_PROVIDER", "local"),
		FastF1CacheDir:       getEnv("FASTF1_CACHE_DIR", "./data/fastf1_cache"),
		LogLevel:             getEnv("LOG_LEVEL", "debug"),
	}
	return cfg, nil
}

func getEnv(key, fallback string) string {
	if val, ok := os.LookupEnv(key); ok {
		return val
	}
	return fallback
}

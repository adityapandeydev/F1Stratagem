.PHONY: api worker frontend dev migrate-up migrate-down

# --- API ---
api:
	cd api && go run ./cmd/server

api-build:
	cd api && go build -o bin/server.exe ./cmd/server

api-test:
	cd api && go test ./...

# --- Worker ---
worker:
	cd worker && python -m uvicorn src.main:app --host 0.0.0.0 --port 8081 --reload

worker-install:
	cd worker && pip install -r requirements.txt

# --- Frontend ---
frontend:
	cd frontend && bun run dev

frontend-install:
	cd frontend && bun install

frontend-build:
	cd frontend && bun run build

# --- Database ---
migrate-up:
	cd api && go run ./cmd/migrate up

migrate-down:
	cd api && go run ./cmd/migrate down

# --- Dev (all services) ---
dev:
	@echo "Start each service in a separate terminal:"
	@echo "  make api"
	@echo "  make worker"
	@echo "  make frontend"

# --- Setup ---
setup:
	@echo "1. Copy .env.example to .env and configure"
	@echo "2. Create PostgreSQL database: f1stratagem"
	@echo "3. Run: make migrate-up"
	@echo "4. Run: make worker-install"
	@echo "5. Run: make frontend-install"
	@echo "6. Start services: make dev"

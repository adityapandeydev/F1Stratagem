.PHONY: api api-build api-test analysis analysis-install frontend frontend-install frontend-build \
        f1data-install f1data-test archive-verify predictor-install predictor-test migrate-up migrate-down dev setup

# --- API ---
api:
	cd api && go run ./cmd/server

api-build:
	cd api && go build -o bin/server.exe ./cmd/server

api-test:
	cd api && go test ./...

# --- Analysis service ---
analysis:
	cd services/analysis && python -m uvicorn src.main:app --host 0.0.0.0 --port 8081 --reload

analysis-install:
	cd services/analysis && pip install -r requirements.txt

# --- Data archive ---
f1data-install:
	uv pip install -e "packages/f1data[dev]"

f1data-test:
	python -m pytest packages/f1data

archive-verify:
	f1data verify

# --- Prediction engine ---
predictor-install:
	uv pip install -e "services/predictor[dev]"

predictor-test:
	python -m pytest services/predictor

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
	@echo "  make analysis"
	@echo "  make frontend"

# --- Setup ---
setup:
	@echo "1. Copy .env.example to .env and configure"
	@echo "2. Create PostgreSQL database: f1stratagem"
	@echo "3. Run: make migrate-up"
	@echo "4. Run: make analysis-install f1data-install"
	@echo "5. Run: make frontend-install"
	@echo "6. Start services: make dev"

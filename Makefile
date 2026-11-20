.PHONY: api api-build api-test analysis analysis-test frontend frontend-install frontend-build \
        f1data-install f1data-test archive-verify predictor-install predictor-native predictor-test migrate-up migrate-down dev setup

# --- API ---
api:
	cd api && go run ./cmd/server

api-build:
	cd api && go build -o bin/server.exe ./cmd/server

api-test:
	cd api && go test ./...

# --- Analysis service ---
analysis:
	cd services/analysis && cargo build --release

analysis-test:
	cd services/analysis && cargo test --release

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
	$(MAKE) predictor-native

# the Rust kernels: the pace filter (kernels/pace) and the race simulation (kernels/race)
predictor-native:
	cd services/predictor/kernels/pace && maturin develop --release
	cd services/predictor/kernels/race && maturin develop --release

predictor-test:
	python -m pytest services/predictor
	cd services/predictor/kernels/race && cargo test --release

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
	@echo "4. Run: make analysis f1data-install"
	@echo "5. Run: make frontend-install"
	@echo "6. Start services: make dev"

# LegacyCoinTrader Microservices - Makefile
.PHONY: help build up down test clean dev prod logs health status

# Default target
help: ## Show this help message
	@echo "LegacyCoinTrader Microservices"
	@echo ""
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-20s %s\n", $$1, $$2}'

# Development Environment
dev: ## Start development environment
	docker-compose -f docker-compose.yml -f docker-compose.dev.yml up -d
	@echo "Development environment started"
	@echo "API Gateway: http://localhost:8000"
	@echo "Frontend: http://localhost:5000"

dev-build: ## Build and start development environment
	docker-compose -f docker-compose.yml -f docker-compose.dev.yml up -d --build
	@echo "Development environment built and started"

dev-logs: ## View development logs
	docker-compose -f docker-compose.yml -f docker-compose.dev.yml logs -f

dev-stop: ## Stop development environment
	docker-compose -f docker-compose.yml -f docker-compose.dev.yml down

# Production Environment
prod: ## Start production environment
	docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d
	@echo "Production environment started"
	@echo "API Gateway: http://localhost:80"
	@echo "Frontend: http://localhost"

prod-build: ## Build and start production environment
	docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d --build
	@echo "Production environment built and started"

prod-logs: ## View production logs
	docker-compose -f docker-compose.yml -f docker-compose.prod.yml logs -f

prod-stop: ## Stop production environment
	docker-compose -f docker-compose.yml -f docker-compose.prod.yml down

# Testing
test: ## Run end-to-end tests
	docker-compose -f docker-compose.yml -f docker-compose.test.yml up --abort-on-container-exit e2e-tests

test-build: ## Build and run end-to-end tests
	docker-compose -f docker-compose.yml -f docker-compose.test.yml up --build --abort-on-container-exit e2e-tests

test-local: ## Run tests locally (requires running services)
	cd tests/e2e && python run_e2e_tests.py

test-results: ## View test results
	@echo "Test results:"
	@ls -la test_results/ 2>/dev/null || echo "No test results found"

observability-check: ## Validate observability instrumentation
	pytest tests/test_observability.py

migration-audit: ## Run post-migration data integrity validation against fixture snapshots
	@mkdir -p test_results
	python tools/migration/tenant_migration.py audit \
		--legacy-snapshot tests/fixtures/migration/legacy_snapshot.json \
		--modern-snapshot tests/fixtures/migration/modern_snapshot.json \
		--report-path test_results/migration_audit_report.json \
		--tolerance 0.0 \
		--fail-on-drift

# Service Management
build: ## Build all services
	docker-compose build

up: ## Start all services (development)
	docker-compose up -d

down: ## Stop all services
	docker-compose down

restart: ## Restart all services
	docker-compose restart

# Monitoring and Health
health: ## Check health of all services
	@echo "Checking service health..."
	@docker-compose ps
	@echo ""
	@echo "Health checks:"
	@curl -s http://localhost:8000/health | jq . 2>/dev/null || echo "API Gateway: DOWN"
	@curl -s http://localhost:8001/health | jq . 2>/dev/null || echo "Trading Engine: DOWN"
	@curl -s http://localhost:8002/health | jq . 2>/dev/null || echo "Market Data: DOWN"
	@curl -s http://localhost:8003/health | jq . 2>/dev/null || echo "Portfolio: DOWN"
	@curl -s http://localhost:8004/health | jq . 2>/dev/null || echo "Strategy Engine: DOWN"
	@curl -s http://localhost:8005/health | jq . 2>/dev/null || echo "Token Discovery: DOWN"
	@curl -s http://localhost:8006/health | jq . 2>/dev/null || echo "Execution: DOWN"
	@curl -s http://localhost:8007/health | jq . 2>/dev/null || echo "Monitoring: DOWN"
	@curl -s http://localhost:5000/health | jq . 2>/dev/null || echo "Frontend: DOWN"

status: ## Show status of all services
	docker-compose ps

logs: ## View logs from all services
	docker-compose logs -f

logs-%: ## View logs from specific service (e.g., make logs-api-gateway)
	docker-compose logs -f $*

# Database
db-init: ## Initialize database (production)
	docker-compose -f docker-compose.yml -f docker-compose.prod.yml exec postgres psql -U prod_user -d legacy_coin_trader_prod -f /docker-entrypoint-initdb.d/init.sql

db-backup: ## Backup database (production)
	docker-compose -f docker-compose.yml -f docker-compose.prod.yml exec postgres pg_dump -U prod_user legacy_coin_trader_prod > backup_$(date +%Y%m%d_%H%M%S).sql

# Scaling
scale-trading-engine: ## Scale trading engine to 3 instances
	docker-compose up -d --scale trading-engine=3

scale-frontend: ## Scale frontend to 3 instances
	docker-compose up -d --scale frontend=3

# Cleanup
clean: ## Remove all containers, volumes, and images
	docker-compose down -v --rmi all
	docker system prune -f

clean-volumes: ## Remove all volumes
	docker-compose down -v

clean-images: ## Remove all images
	docker rmi $(docker images -q) 2>/dev/null || true

# Development Helpers
shell-%: ## Open shell in service container (e.g., make shell-api-gateway)
	docker-compose exec $* /bin/bash || docker-compose exec $* /bin/sh

format: ## Format Python code
	find . -name "*.py" -not -path "./venv/*" -not -path "./__pycache__/*" | xargs black --line-length 100

lint: ## Lint Python code
	find . -name "*.py" -not -path "./venv/*" -not -path "./__pycache__/*" | xargs flake8 --max-line-length 100

# CI/CD
ci-test: ## Run tests in CI environment
	$(MAKE) migration-audit
	$(MAKE) observability-check
	docker-compose -f docker-compose.yml -f docker-compose.test.yml up --abort-on-container-exit --build e2e-tests

ci-deploy-dev: ## Deploy to development environment
	docker-compose -f docker-compose.yml -f docker-compose.dev.yml up -d --build

ci-deploy-prod: ## Deploy to production environment
	docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d --build

# Quick Commands
quick-start: ## Quick start for development
	@echo "Starting LegacyCoinTrader in development mode..."
	@make dev-build
	@echo "Waiting for services to be healthy..."
	@sleep 30
	@make health

quick-test: ## Quick test run
	@echo "Running E2E tests..."
	@make test-build
	@make test-results

# Info
info: ## Show system information
	@echo "Docker version: $$(docker --version)"
	@echo "Docker Compose version: $$(docker-compose --version)"
	@echo "Available services:"
	@docker-compose config --services | cat

# Emergency
emergency-stop: ## Emergency stop all containers
	docker stop $$(docker ps -q) 2>/dev/null || true
	docker rm $$(docker ps -a -q) 2>/dev/null || true

emergency-clean: ## Emergency cleanup (removes everything)
	docker system prune -a -f --volumes

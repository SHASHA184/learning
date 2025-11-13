# Docker & Containerization Learning Module

## Overview
Docker is an essential tool for modern backend development, enabling consistent environments across development, testing, and production. This module covers containerization concepts crucial for the Namecheap role.

## Learning Objectives
- Understand containerization concepts and benefits
- Build and run Docker containers
- Write efficient Dockerfiles
- Use Docker Compose for multi-container applications
- Containerize Django applications with PostgreSQL
- Implement development and production configurations
- Debug containerized applications

## Prerequisites
- Linux/Unix command-line basics
- Basic understanding of networking
- Python and Django fundamentals

## Module Structure

### 1. Docker Basics
**Time estimate**: 2-3 days

Topics:
- What is containerization?
- Docker vs Virtual Machines
- Docker architecture (daemon, client, registry)
- Images vs Containers
- Docker Hub and registries
- Basic Docker commands

Practice:
- `web-backend/docker/01_basics/` - Basic Docker exercises

Essential commands:
```bash
# Pull an image
docker pull python:3.11-slim

# List images
docker images

# Run a container
docker run -it python:3.11-slim python

# Run container in background
docker run -d --name my-nginx -p 8080:80 nginx

# List running containers
docker ps
docker ps -a  # Include stopped containers

# Stop/start/restart container
docker stop my-nginx
docker start my-nginx
docker restart my-nginx

# View logs
docker logs my-nginx
docker logs -f my-nginx  # Follow logs

# Execute command in running container
docker exec -it my-nginx bash

# Remove container
docker rm my-nginx

# Remove image
docker rmi python:3.11-slim
```

### 2. Dockerfile Creation
**Time estimate**: 3-4 days

Topics:
- Dockerfile syntax and instructions
- Base images selection
- Layer caching and optimization
- Multi-stage builds
- .dockerignore file
- Best practices for Python/Django

Practice:
- `web-backend/docker/02_dockerfile/` - Dockerfile examples

Basic Dockerfile structure:
```dockerfile
# Simple Python application
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "app.py"]
```

Django application Dockerfile:
```dockerfile
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    postgresql-client \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy project
COPY . .

# Collect static files
RUN python manage.py collectstatic --noinput

# Run gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "myproject.wsgi:application"]
```

Multi-stage build for smaller images:
```dockerfile
# Build stage
FROM python:3.11-slim as builder

WORKDIR /app

RUN apt-get update && apt-get install -y \
    gcc \
    postgresql-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip wheel --no-cache-dir --no-deps --wheel-dir /app/wheels -r requirements.txt

# Runtime stage
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y \
    libpq5 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --from=builder /app/wheels /wheels
COPY --from=builder /app/requirements.txt .

RUN pip install --no-cache /wheels/*

COPY . .

CMD ["gunicorn", "--bind", "0.0.0.0:8000", "myproject.wsgi:application"]
```

.dockerignore example:
```
__pycache__
*.pyc
*.pyo
*.pyd
.Python
env/
venv/
.env
.git
.gitignore
.vscode
.idea
*.md
docker-compose*.yml
Dockerfile*
.pytest_cache
.coverage
htmlcov/
dist/
build/
*.egg-info
```

### 3. Docker Networking
**Time estimate**: 2-3 days

Topics:
- Network types (bridge, host, none)
- Container communication
- Port mapping
- DNS resolution in Docker
- Custom networks

Practice:
- `web-backend/docker/03_networking/` - Networking examples

Commands:
```bash
# Create network
docker network create myapp-network

# Run containers on same network
docker run -d --name db --network myapp-network postgres:15
docker run -d --name web --network myapp-network -p 8000:8000 myapp

# Inspect network
docker network inspect myapp-network

# Containers can reach each other by name
# In web container, connect to: postgresql://db:5432
```

### 4. Docker Volumes & Data Persistence
**Time estimate**: 2-3 days

Topics:
- Volume types (named volumes, bind mounts, tmpfs)
- Data persistence strategies
- Backup and restore
- Development vs production volumes

Practice:
- `web-backend/docker/04_volumes/` - Volume examples

Commands:
```bash
# Create named volume
docker volume create postgres-data

# Run with volume
docker run -d \
    --name postgres \
    -v postgres-data:/var/lib/postgresql/data \
    -e POSTGRES_PASSWORD=secret \
    postgres:15

# Bind mount (for development)
docker run -d \
    --name django-dev \
    -v $(pwd):/app \
    -p 8000:8000 \
    myapp

# List volumes
docker volume ls

# Inspect volume
docker volume inspect postgres-data

# Remove volume
docker volume rm postgres-data

# Backup volume
docker run --rm \
    -v postgres-data:/data \
    -v $(pwd):/backup \
    ubuntu tar czf /backup/postgres-backup.tar.gz /data
```

### 5. Docker Compose
**Time estimate**: 4-5 days

Topics:
- Docker Compose file format
- Service definition
- Multi-container orchestration
- Environment variables and .env files
- Compose networking and volumes
- Development workflows

Practice:
- `web-backend/docker/05_compose/` - Compose examples

Basic docker-compose.yml:
```yaml
version: '3.8'

services:
  db:
    image: postgres:15
    volumes:
      - postgres-data:/var/lib/postgresql/data
    environment:
      POSTGRES_DB: myapp
      POSTGRES_USER: myapp
      POSTGRES_PASSWORD: ${DB_PASSWORD:-secret}
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U myapp"]
      interval: 10s
      timeout: 5s
      retries: 5

  redis:
    image: redis:7-alpine
    volumes:
      - redis-data:/data

  web:
    build: .
    command: gunicorn myproject.wsgi:application --bind 0.0.0.0:8000
    volumes:
      - .:/app
      - static-volume:/app/staticfiles
    ports:
      - "8000:8000"
    environment:
      DATABASE_URL: postgresql://myapp:${DB_PASSWORD:-secret}@db:5432/myapp
      REDIS_URL: redis://redis:6379/0
      DEBUG: ${DEBUG:-False}
    depends_on:
      db:
        condition: service_healthy
      redis:
        condition: service_started

  celery:
    build: .
    command: celery -A myproject worker -l info
    volumes:
      - .:/app
    environment:
      DATABASE_URL: postgresql://myapp:${DB_PASSWORD:-secret}@db:5432/myapp
      REDIS_URL: redis://redis:6379/0
    depends_on:
      - db
      - redis

  celery-beat:
    build: .
    command: celery -A myproject beat -l info
    volumes:
      - .:/app
    environment:
      DATABASE_URL: postgresql://myapp:${DB_PASSWORD:-secret}@db:5432/myapp
      REDIS_URL: redis://redis:6379/0
    depends_on:
      - db
      - redis

volumes:
  postgres-data:
  redis-data:
  static-volume:
```

Development-specific docker-compose.override.yml:
```yaml
version: '3.8'

services:
  web:
    build:
      context: .
      dockerfile: Dockerfile.dev
    command: python manage.py runserver 0.0.0.0:8000
    volumes:
      - .:/app
    environment:
      DEBUG: "True"
    ports:
      - "8000:8000"

  celery:
    command: watchdog -p "*.py" --restart celery -A myproject worker -l debug
    environment:
      DEBUG: "True"
```

Compose commands:
```bash
# Start all services
docker-compose up

# Start in background
docker-compose up -d

# View logs
docker-compose logs
docker-compose logs -f web

# Execute command in service
docker-compose exec web python manage.py migrate
docker-compose exec web python manage.py shell

# Stop services
docker-compose stop

# Stop and remove containers
docker-compose down

# Remove volumes too
docker-compose down -v

# Rebuild images
docker-compose build

# Scale service
docker-compose up -d --scale celery=3
```

### 6. Django + PostgreSQL with Docker
**Time estimate**: 3-4 days

Topics:
- Complete Django project containerization
- Database initialization and migrations
- Static files handling
- Development vs production configurations
- Environment variable management
- Health checks

Practice:
- `django/09_docker_deployment/` - Complete Django Docker setup

Project structure:
```
myproject/
├── docker-compose.yml
├── docker-compose.override.yml  # Dev overrides
├── docker-compose.prod.yml      # Production config
├── Dockerfile
├── Dockerfile.dev
├── .dockerignore
├── .env.example
├── requirements.txt
├── requirements-dev.txt
├── entrypoint.sh
└── myproject/
    ├── manage.py
    └── ...
```

entrypoint.sh:
```bash
#!/bin/bash
set -e

# Wait for PostgreSQL
echo "Waiting for PostgreSQL..."
while ! nc -z db 5432; do
  sleep 0.1
done
echo "PostgreSQL started"

# Run migrations
python manage.py migrate --noinput

# Collect static files
python manage.py collectstatic --noinput

exec "$@"
```

### 7. Docker Debugging & Troubleshooting
**Time estimate**: 2-3 days

Topics:
- Container logs analysis
- Interactive debugging
- Network troubleshooting
- Performance monitoring
- Common issues and solutions

Practice:
- `web-backend/docker/07_debugging/` - Debugging scenarios

Debugging commands:
```bash
# View container logs
docker-compose logs -f web

# Execute shell in container
docker-compose exec web bash

# View container processes
docker-compose top

# View container stats
docker stats

# Inspect container
docker inspect <container-id>

# View container changes
docker diff <container-id>

# Debug networking
docker-compose exec web ping db
docker-compose exec web nslookup db

# Check health status
docker-compose ps
```

Common issues:
```bash
# Issue: Container exits immediately
# Solution: Check logs
docker-compose logs web

# Issue: Cannot connect to database
# Solution: Verify network and wait for DB
docker-compose exec web ping db
# Add depends_on with health check

# Issue: Permission denied on volumes
# Solution: Fix ownership
docker-compose exec web chown -R appuser:appuser /app

# Issue: Port already in use
# Solution: Change port mapping or stop conflicting service
docker-compose down
netstat -tulpn | grep 8000
```

### 8. Production Best Practices
**Time estimate**: 3-4 days

Topics:
- Security considerations
- Image optimization
- Secrets management
- Logging and monitoring
- Resource limits
- Health checks
- CI/CD integration

Practice:
- `web-backend/docker/08_production/` - Production configurations

Production Dockerfile:
```dockerfile
FROM python:3.11-slim

# Create non-root user
RUN useradd -m -u 1000 appuser && \
    mkdir -p /app && \
    chown -R appuser:appuser /app

# Install dependencies as root
COPY requirements.txt /tmp/
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Switch to non-root user
USER appuser
WORKDIR /app

COPY --chown=appuser:appuser . .

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8000/health/ || exit 1

EXPOSE 8000

CMD ["gunicorn", "--bind", "0.0.0.0:8000", \
     "--workers", "4", \
     "--timeout", "120", \
     "--access-logfile", "-", \
     "--error-logfile", "-", \
     "myproject.wsgi:application"]
```

Production docker-compose.yml:
```yaml
version: '3.8'

services:
  web:
    image: myapp:${VERSION:-latest}
    restart: always
    deploy:
      resources:
        limits:
          cpus: '2'
          memory: 2G
        reservations:
          cpus: '1'
          memory: 1G
    environment:
      - DATABASE_URL=${DATABASE_URL}
      - SECRET_KEY=${SECRET_KEY}
      - REDIS_URL=${REDIS_URL}
    logging:
      driver: "json-file"
      options:
        max-size: "10m"
        max-file: "3"
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health/"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s
```

## Practical Projects

### Project 1: Containerized Blog API
**Time**: 1 week
**Components**:
- Django REST API
- PostgreSQL database
- Redis for caching
- Celery for background tasks
- Nginx as reverse proxy

### Project 2: Microservices Setup
**Time**: 1.5 weeks
**Components**:
- Multiple Django services
- Shared PostgreSQL
- Inter-service communication
- API gateway

### Project 3: Full Production Setup
**Time**: 2 weeks
**Components**:
- Multi-stage builds
- Health checks
- Monitoring with Prometheus
- Log aggregation
- Secrets management
- CI/CD pipeline

## Interview Preparation

### Common Docker Questions
1. What is Docker and why use it?
2. Difference between image and container?
3. What is a Dockerfile and what are its key instructions?
4. Explain Docker layers and caching
5. How do containers communicate with each other?
6. What are Docker volumes and why use them?
7. What is Docker Compose?
8. How do you debug a failing container?
9. Explain multi-stage builds
10. What are Docker security best practices?

### Practical Scenarios
- Containerize an existing Django application
- Debug a container that won't start
- Set up development environment with Docker Compose
- Optimize a large Docker image
- Handle database migrations in containers

## Tools & Resources

### Installation
```bash
# Docker installation (Ubuntu)
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Add user to docker group
sudo usermod -aG docker $USER

# Docker Compose (if not included)
sudo apt install docker-compose-plugin
```

### Useful Tools
- **Docker Desktop**: GUI for Docker (Windows/Mac)
- **Portainer**: Web-based Docker management
- **Dive**: Analyze Docker images
- **Hadolint**: Dockerfile linter

### Resources
- [Docker Documentation](https://docs.docker.com/)
- [Docker Compose Documentation](https://docs.docker.com/compose/)
- [Docker Best Practices](https://docs.docker.com/develop/dev-best-practices/)
- [Play with Docker](https://labs.play-with-docker.com/)

## Integration with Other Modules

### With Django
- See `django/09_docker_deployment/`
- Containerized Django REST API projects

### With Databases
- PostgreSQL container setup
- Database initialization scripts
- Migration handling

### With CI/CD
- GitHub Actions with Docker
- Building and pushing images
- Container testing

## Progress Tracking
- [ ] Docker basics understood
- [ ] Dockerfile creation mastered
- [ ] Docker networking learned
- [ ] Volumes and persistence understood
- [ ] Docker Compose proficiency achieved
- [ ] Django + PostgreSQL containerized
- [ ] Debugging skills developed
- [ ] Production best practices learned
- [ ] Blog API project containerized
- [ ] Microservices setup completed
- [ ] Production setup completed

## Next Steps
After completing this module:
1. Containerize all Django projects
2. Set up local development environment with Docker
3. Learn Kubernetes basics (optional)
4. Explore Docker Swarm (optional)
5. Integrate with CI/CD pipelines

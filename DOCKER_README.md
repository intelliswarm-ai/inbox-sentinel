# Docker Setup for Inbox Sentinel

This document provides comprehensive instructions for running Inbox Sentinel using Docker containers.

## Overview

Inbox Sentinel uses a microservices architecture with separate containers for each ML model and an orchestrator service that coordinates between them.

## Services Architecture

| Service | Container Name | Port | Description |
|---------|---------------|------|-------------|
| **Orchestrator** | `inbox-sentinel` | 8006 | Main service that coordinates model requests |
| **Naive Bayes** | `inbox-sentinel-naive-bayes` | 8001 | Naive Bayes classifier model |
| **SVM** | `inbox-sentinel-svm` | 8002 | Support Vector Machine model |
| **Random Forest** | `inbox-sentinel-random-forest` | 8003 | Random Forest classifier model |
| **Logistic Regression** | `inbox-sentinel-logistic-regression` | 8004 | Logistic Regression model |
| **Neural Network** | `inbox-sentinel-neural-network` | 8005 | Neural Network classifier model |

## Prerequisites

- Docker Engine 20.10+
- Docker Compose 2.0+
- At least 4GB RAM (recommended 8GB for all models)
- 10GB free disk space

## Quick Start

1. **Clone and navigate to the repository:**
   ```bash
   cd /path/to/inbox-sentinel
   ```

2. **Build and start all services:**
   ```bash
   docker compose up --build
   ```

3. **Run in detached mode:**
   ```bash
   docker compose up -d
   ```

4. **Check service status:**
   ```bash
   docker compose ps
   ```

## Individual Service Management

### Start specific services:
```bash
# Start only orchestrator and naive-bayes
docker compose up inbox-sentinel naive-bayes

# Start all model servers except neural-network
docker compose up naive-bayes svm random-forest logistic-regression
```

### View logs:
```bash
# View all logs
docker compose logs

# View specific service logs
docker compose logs inbox-sentinel
docker compose logs -f naive-bayes  # Follow logs
```

### Restart services:
```bash
# Restart all services
docker compose restart

# Restart specific service
docker compose restart svm
```

## API Endpoints

Once running, the services expose the following endpoints:

### Orchestrator (Port 8006)
- **Health Check:** `GET http://localhost:8006/health`
- **Classify Email:** `POST http://localhost:8006/classify`
- **Model Status:** `GET http://localhost:8006/models/status`

### Individual Models
Each model service exposes:
- **Health Check:** `GET http://localhost:800[1-5]/health`
- **Classify:** `POST http://localhost:800[1-5]/classify`
- **Model Info:** `GET http://localhost:800[1-5]/info`

Example:
```bash
# Test naive-bayes directly
curl -X POST http://localhost:8001/classify \
  -H "Content-Type: application/json" \
  -d '{"email": "Suspicious email content here"}'

# Test orchestrator (uses all models)
curl -X POST http://localhost:8006/classify \
  -H "Content-Type: application/json" \
  -d '{"email": "Email to analyze"}'
```

## Data and Logs

### Volumes
- `./data:/app/data` - Model data and training files
- `./logs:/app/logs` - Application logs

### Accessing logs:
```bash
# View logs from host
tail -f ./logs/inbox-sentinel.log

# View logs from container
docker compose exec inbox-sentinel tail -f /app/logs/inbox-sentinel.log
```

## Development

### Building images:
```bash
# Build all images
docker compose build

# Build specific service
docker compose build inbox-sentinel

# Build without cache
docker compose build --no-cache
```

### Interactive debugging:
```bash
# Access container shell
docker compose exec inbox-sentinel bash

# Run commands in container
docker compose exec inbox-sentinel inbox-sentinel --help
```

### Development with live reload:
```bash
# Mount source code for development
docker compose -f docker-compose.yml -f docker-compose.dev.yml up
```

## Troubleshooting

### Port Conflicts
If you see "port already allocated" errors:

1. **Check what's using the ports:**
   ```bash
   ss -tlnp | grep :8006
   ```

2. **Stop conflicting services:**
   ```bash
   docker stop $(docker ps -q --filter "publish=8000-8006")
   ```

3. **Change ports in docker-compose.yml if needed**

### Container Issues

**Container keeps restarting:**
```bash
# Check logs for errors
docker compose logs [service-name]

# Check resource usage
docker stats
```

**Models not loading:**
```bash
# Ensure data directory exists and has model files
ls -la ./data/

# Check container permissions
docker compose exec inbox-sentinel ls -la /app/data/
```

**Memory issues:**
```bash
# Check memory usage
docker stats --no-stream

# Increase Docker memory allocation in Docker settings
```

### Cleanup

```bash
# Stop and remove containers
docker compose down

# Remove containers and volumes
docker compose down -v

# Remove containers, volumes, and images
docker compose down -v --rmi all

# Clean up Docker system
docker system prune -a
```

## Performance Tuning

### Resource Limits
Add resource limits to docker-compose.yml:

```yaml
services:
  inbox-sentinel:
    # ... other config
    deploy:
      resources:
        limits:
          memory: 2G
          cpus: '1.0'
        reservations:
          memory: 1G
          cpus: '0.5'
```

### Scaling
Run multiple instances of model services:

```bash
# Scale naive-bayes to 3 instances
docker compose up --scale naive-bayes=3

# Scale multiple services
docker compose up --scale naive-bayes=2 --scale svm=2
```

## Security Notes

- Containers run as non-root user (`appuser`)
- No sensitive data should be in environment variables
- Use Docker secrets for production deployments
- Regularly update base images for security patches

## Production Deployment

For production environments:

1. Use Docker Swarm or Kubernetes
2. Set up proper logging (ELK stack, etc.)
3. Configure health checks and monitoring
4. Use environment-specific configuration files
5. Implement backup strategies for data volumes

## Support

For issues with Docker deployment:

1. Check container logs: `docker compose logs`
2. Verify system requirements
3. Ensure no port conflicts
4. Check Docker daemon status
5. Review this documentation

---

**Note:** This setup is optimized for development and testing. For production deployments, additional configuration for security, monitoring, and scalability should be implemented.
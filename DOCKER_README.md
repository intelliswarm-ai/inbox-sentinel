# Docker Setup for Inbox Sentinel

This document provides comprehensive instructions for running Inbox Sentinel using Docker containers.

## Overview

Inbox Sentinel runs as a single containerized application that provides CLI access to all machine learning models and email analysis capabilities.

## Services Architecture

| Service | Container Name | Description |
|---------|---------------|-------------|
| **Inbox Sentinel** | `inbox-sentinel` | Main CLI container for all model operations and analysis |

The architecture has been simplified to use a single container with all models available through CLI commands. This approach is more suitable for development and eliminates the complexity of running multiple MCP server containers.

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

## Running CLI Commands in Docker

All `inbox-sentinel` CLI commands can be executed within the Docker containers using `docker compose exec`.

### Model Management

```bash
# View available models and their status
docker compose exec inbox-sentinel inbox-sentinel models list

# Train all models (requires training data in ./data/)
docker compose exec inbox-sentinel inbox-sentinel models train

# Verify trained models
docker compose exec inbox-sentinel inbox-sentinel models verify
```

### Email Analysis

```bash
# Analyze an email with content, subject, and sender
docker compose exec inbox-sentinel inbox-sentinel analyze \
  -c "Email content here" \
  -s "Subject line" \
  -f "sender@email.com"

# Analyze a forwarded Gmail email (file must be in ./data/ or ./testfiles/)
docker compose exec inbox-sentinel inbox-sentinel analyze \
  -F /app/data/forwarded_email.txt --forwarded

# Alternative: Copy file to container and analyze
docker cp forwarded_email.txt inbox-sentinel:/app/
docker compose exec inbox-sentinel inbox-sentinel analyze \
  -F forwarded_email.txt --forwarded
```

### Model Orchestration

```bash
# Orchestrate multiple models with consensus (from file)
docker compose exec inbox-sentinel inbox-sentinel orchestrate \
  -F /app/data/email.txt --forwarded

# Orchestrate with direct content (must be in forwarded email format)
docker compose exec inbox-sentinel inbox-sentinel orchestrate \
  -c "---------- Forwarded message ---------
From: sender@suspicious.com
Subject: Urgent Action Required
To: you@email.com

Your account has been suspended! Click here immediately..." \
  --forwarded

# Use Ollama for LLM-enhanced orchestration (requires Ollama server)
docker compose exec inbox-sentinel inbox-sentinel orchestrate \
  -F /app/data/email.txt --forwarded --llm-provider ollama
```

### Server Management

```bash
# Start specific MCP servers (already handled by docker-compose)
docker compose exec inbox-sentinel inbox-sentinel server start neural-network

# Check server status
docker compose exec inbox-sentinel inbox-sentinel server status

# View server logs
docker compose exec inbox-sentinel inbox-sentinel server logs neural-network
```

### Working with Files

To analyze local email files, you have several options:

**Option 1: Use mounted volumes (recommended)**
```bash
# Place your email files in ./data/ or ./testfiles/
cp my_email.txt ./data/
docker compose exec inbox-sentinel inbox-sentinel analyze -F /app/data/my_email.txt --forwarded
```

**Option 2: Copy files to container**
```bash
# Copy file to running container
docker cp my_email.txt inbox-sentinel:/app/
docker compose exec inbox-sentinel inbox-sentinel analyze -F my_email.txt --forwarded
```

**Option 3: Use stdin**
```bash
# Pipe content directly
echo "Email content here" | docker compose exec -T inbox-sentinel inbox-sentinel analyze -c -
```

### Interactive Shell

For multiple commands or debugging:

```bash
# Access interactive shell
docker compose exec inbox-sentinel bash

# Inside container, run commands normally:
# inbox-sentinel models list
# inbox-sentinel analyze -c "content" -s "subject" -f "from@email.com"
# exit
```

## Testing the Setup

You can verify the installation works correctly:

```bash
# Check all models are available
docker compose exec inbox-sentinel inbox-sentinel models list

# Get system information
docker compose exec inbox-sentinel inbox-sentinel info

# Test analysis with demo content
docker compose exec inbox-sentinel inbox-sentinel analyze \
  -c "Congratulations! You've won $1 million! Click here to claim your prize now!" \
  -s "Urgent: Claim Your Prize!" \
  -f "winner@lottery-scam.com"
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
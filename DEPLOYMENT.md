# Deployment Guide

This guide covers deploying the Heart Attack Risk API to production environments.

## Prerequisites

- Python 3.8+
- Git
- 2GB RAM minimum
- Port 5001 available

## Local Deployment

### 1. Clone and Setup
```bash
# Clone repository
git clone <your-repo-url>
cd heart_attack_risk_api

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # Mac/Linux
# venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Train Model (First Time Only)
```bash
python3 train_model.py
```

This creates:
- `models/trained_model.pkl`
- `models/scaler.pkl`
- `models/model_metadata.json`

### 3. Run Tests
```bash
pytest tests/ -v
```

### 4. Start API
```bash
python3 src/api.py
```

API runs on `http://localhost:5001`

### 5. Verify Deployment
```bash
# Health check
curl http://localhost:5001/health

# Test prediction
curl -X POST http://localhost:5001/predict \
  -H "Content-Type: application/json" \
  -d '{"age": 45, "sex": 1, "cp": 0, "trtbps": 120, "chol": 200, "fbs": 0, "restecg": 0, "thalachh": 150, "exng": 0, "oldpeak": 0.0, "slp": 2, "caa": 0, "thall": 2}'
```

## Docker Deployment

### Create Dockerfile
```dockerfile
FROM python:3.12-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY src/ ./src/
COPY models/ ./models/

# Expose port
EXPOSE 5001

# Run application
CMD ["python", "src/api.py"]
```

### Build and Run
```bash
# Build image
docker build -t heart-attack-api:latest .

# Run container
docker run -d -p 5001:5001 --name heart-api heart-attack-api:latest

# Check logs
docker logs heart-api

# Stop container
docker stop heart-api
```

## Production Deployment

### Environment Variables

Create `.env` file:
```bash
FLASK_ENV=production
API_PORT=5001
MODEL_PATH=models/
LOG_LEVEL=INFO
```

Load in application:
```python
from dotenv import load_dotenv
import os

load_dotenv()

port = int(os.getenv('API_PORT', 5001))
app.run(host='0.0.0.0', port=port)
```

### Production Server (Gunicorn)

Install:
```bash
pip install gunicorn
```

Run:
```bash
gunicorn -w 4 -b 0.0.0.0:5001 src.api:app
```

Configuration (`gunicorn.conf.py`):
```python
bind = "0.0.0.0:5001"
workers = 4
worker_class = "sync"
timeout = 60
keepalive = 5
errorlog = "logs/error.log"
accesslog = "logs/access.log"
loglevel = "info"
```

### Systemd Service (Linux)

Create `/etc/systemd/system/heart-api.service`:
```ini
[Unit]
Description=Heart Attack Risk API
After=network.target

[Service]
Type=simple
User=www-data
WorkingDirectory=/var/www/heart_attack_risk_api
Environment="PATH=/var/www/heart_attack_risk_api/venv/bin"
ExecStart=/var/www/heart_attack_risk_api/venv/bin/gunicorn -c gunicorn.conf.py src.api:app
Restart=always

[Install]
WantedBy=multi-user.target
```

Enable and start:
```bash
sudo systemctl enable heart-api
sudo systemctl start heart-api
sudo systemctl status heart-api
```

### Nginx Reverse Proxy

Configuration (`/etc/nginx/sites-available/heart-api`):
```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://localhost:5001;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

Enable:
```bash
sudo ln -s /etc/nginx/sites-available/heart-api /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl reload nginx
```

## Cloud Deployment

### AWS EC2

1. Launch EC2 instance (t2.small or larger)
2. SSH into instance
3. Install Python 3.12
4. Follow local deployment steps
5. Configure security group (port 5001)
6. Use Elastic IP for static IP
7. Set up CloudWatch for monitoring

### Heroku
```bash
# Create Procfile
echo "web: gunicorn src.api:app" > Procfile

# Deploy
heroku create heart-attack-api
git push heroku main
heroku ps:scale web=1
```

### Google Cloud Run
```bash
# Build container
gcloud builds submit --tag gcr.io/PROJECT-ID/heart-api

# Deploy
gcloud run deploy heart-api \
  --image gcr.io/PROJECT-ID/heart-api \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated
```

## Monitoring

### Health Checks
```bash
# Automated health check (cron)
*/5 * * * * curl -f http://localhost:5001/health || echo "API DOWN" | mail -s "Alert" admin@example.com
```

### Logging

Configure structured logging:
```python
import logging
import json

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/app.log'),
        logging.StreamHandler()
    ]
)
```

### Metrics

Track:
- Request count
- Response times
- Error rates
- Model prediction distribution

## Security

### 1. Environment Variables

Never commit:
- API keys
- Database passwords
- Secret keys

Use `.env` file (add to `.gitignore`)

### 2. HTTPS

Use Let's Encrypt with Nginx:
```bash
sudo apt install certbot python3-certbot-nginx
sudo certbot --nginx -d your-domain.com
```

### 3. Rate Limiting

Install Flask-Limiter:
```bash
pip install Flask-Limiter
```

Add to API:
```python
from flask_limiter import Limiter

limiter = Limiter(
    app,
    key_func=lambda: request.remote_addr,
    default_limits=["100 per hour"]
)

@app.route('/predict', methods=['POST'])
@limiter.limit("10 per minute")
def predict():
    ...
```

### 4. Authentication (Future)

Consider:
- API keys
- JWT tokens
- OAuth 2.0

## Backup and Recovery

### Model Backup
```bash
# Backup models
tar -czf models-backup-$(date +%Y%m%d).tar.gz models/

# Upload to S3
aws s3 cp models-backup-$(date +%Y%m%d).tar.gz s3://your-bucket/backups/
```

### Database Backup (if implemented)
```bash
# SQLite backup
cp logs/predictions.db logs/predictions.db.backup

# PostgreSQL backup
pg_dump dbname > backup.sql
```

## Scaling

### Horizontal Scaling

Use load balancer (Nginx/HAProxy) with multiple instances:
```nginx
upstream heart_api {
    server localhost:5001;
    server localhost:5002;
    server localhost:5003;
}

server {
    location / {
        proxy_pass http://heart_api;
    }
}
```

### Vertical Scaling

Increase resources:
- More CPU cores
- More RAM
- Faster disk

## Troubleshooting

### API Won't Start
```bash
# Check port availability
lsof -i :5001

# Check logs
tail -f logs/app.log

# Verify model files exist
ls -la models/
```

### High Memory Usage
```bash
# Monitor memory
top -p $(pgrep -f "python src/api.py")

# Reduce workers
gunicorn -w 2 src.api:app
```

### Slow Predictions

- Check if model loads on each request (should load once)
- Verify scaler is being reused
- Profile with cProfile

## Maintenance

### Update Model
```bash
# Train new model
python3 train_model.py

# Test new model
pytest tests/ -v

# Restart API
sudo systemctl restart heart-api
```

### Update Dependencies
```bash
# Update requirements
pip install --upgrade -r requirements.txt
pip freeze > requirements.txt

# Test
pytest tests/ -v

# Deploy
git add requirements.txt
git commit -m "Update dependencies"
```

## Checklist

Before deploying to production:

- [ ] All tests passing
- [ ] Environment variables configured
- [ ] HTTPS enabled
- [ ] Rate limiting enabled
- [ ] Monitoring configured
- [ ] Backups automated
- [ ] Documentation updated
- [ ] Security review completed
- [ ] Load testing performed
- [ ] Rollback plan documented

## Support

For issues:
1. Check logs: `logs/app.log`
2. Run tests: `pytest tests/ -v`
3. Verify health: `curl http://localhost:5001/health`
4. Review documentation

---

**Remember:** This API is for demonstration purposes. Full medical applications require additional validation, compliance, and regulatory approval.

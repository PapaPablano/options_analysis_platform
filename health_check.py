#!/usr/bin/env python3
"""
Health check script for the Stock Market Trends Platform
"""

import requests
import sys
import time
import psutil
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def check_app_health(url, timeout=10):
    """Check if the application is responding"""
    try:
        response = requests.get(url, timeout=timeout)
        return response.status_code == 200
    except requests.exceptions.RequestException as e:
        logger.error(f"Health check failed: {e}")
        return False

def check_system_resources():
    """Check system resource usage"""
    memory_percent = psutil.virtual_memory().percent
    cpu_percent = psutil.cpu_percent(interval=1)
    disk_percent = psutil.disk_usage('/').percent
    
    logger.info(f"Memory usage: {memory_percent}%")
    logger.info(f"CPU usage: {cpu_percent}%")
    logger.info(f"Disk usage: {disk_percent}%")
    
    # Alert if resources are high
    if memory_percent > 80:
        logger.warning("High memory usage detected!")
        return False
    if cpu_percent > 90:
        logger.warning("High CPU usage detected!")
        return False
    if disk_percent > 90:
        logger.warning("High disk usage detected!")
        return False
    
    return True

def main():
    """Main health check function"""
    logger.info("Starting health check...")
    
    # Check system resources
    if not check_system_resources():
        logger.error("System resource check failed")
        sys.exit(1)
    
    # Check application health
    urls_to_check = [
        "http://localhost:8501",  # Options dashboard
        "http://localhost:8502"   # Trends dashboard (if running)
    ]
    
    all_healthy = True
    for url in urls_to_check:
        if check_app_health(url):
            logger.info(f"✅ {url} is healthy")
        else:
            logger.error(f"❌ {url} is down")
            all_healthy = False
    
    if all_healthy:
        logger.info("✅ All health checks passed")
        sys.exit(0)
    else:
        logger.error("❌ Some health checks failed")
        sys.exit(1)

if __name__ == "__main__":
    main()

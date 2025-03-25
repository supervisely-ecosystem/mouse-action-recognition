#!/bin/bash

echo "Starting services..."
docker compose -f inference_script/docker-compose.yml up -d

MVD_CONTAINER=$(docker ps -q -f name=mvd)

echo "Waiting for MVD service to complete..."
docker wait $MVD_CONTAINER

echo "MVD service completed. Shutting down all services..."
docker compose -f inference_script/docker-compose.yml down

echo "All services have been stopped."

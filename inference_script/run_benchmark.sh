#!/bin/bash

echo "Starting benchmark"

docker compose -f inference_script/docker-compose.yml up benchmark

echo "Benchmark completed"

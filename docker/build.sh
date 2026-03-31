#!/usr/bin/env bash
# Build the Grand Challenge submission image from the repo root
REPO_ROOT="$( cd "$(dirname "$0")/.." ; pwd -P )"
docker build --platform linux/amd64 -t synthrad_algorithm -f "$REPO_ROOT/docker/Dockerfile" "$REPO_ROOT"

#!/bin/bash

set -e

echo "Creating MLOps platform structure..."

# =========================
# Platform directories
# =========================

mkdir -p platform/config
mkdir -p platform/core
mkdir -p platform/interfaces
mkdir -p platform/data
mkdir -p platform/training
mkdir -p platform/validation
mkdir -p platform/registry
mkdir -p platform/packaging
mkdir -p platform/container
mkdir -p platform/deployment
mkdir -p platform/monitoring
mkdir -p platform/pipeline

# =========================
# Python package files
# =========================

touch platform/__init__.py

touch platform/config/__init__.py
touch platform/core/__init__.py
touch platform/interfaces/__init__.py
touch platform/data/__init__.py
touch platform/training/__init__.py
touch platform/validation/__init__.py
touch platform/registry/__init__.py
touch platform/packaging/__init__.py
touch platform/container/__init__.py
touch platform/deployment/__init__.py
touch platform/monitoring/__init__.py
touch platform/pipeline/__init__.py

# =========================
# Configuration
# =========================

touch platform/config/loader.py
touch platform/config/entrypoint.py

# =========================
# Core
# =========================

touch platform/core/exceptions.py
touch platform/core/project.py
touch platform/core/result.py

# =========================
# Interfaces
# =========================

touch platform/interfaces/data_validator.py
touch platform/interfaces/trainer.py
touch platform/interfaces/validator.py
touch platform/interfaces/registry.py
touch platform/interfaces/deployer.py
touch platform/interfaces/monitor.py

# =========================
# Data
# =========================

touch platform/data/spark_data_validator.py

# =========================
# Training
# =========================

touch platform/training/model_trainer.py

# =========================
# Model validation
# =========================

touch platform/validation/model_validator.py
touch platform/validation/classification_validator.py
touch platform/validation/regression_validator.py

# =========================
# Model registry
# =========================

touch platform/registry/model_registry.py

# =========================
# Model packaging
# =========================

touch platform/packaging/model_image_builder.py

# =========================
# Container registry
# =========================

touch platform/container/container_registry.py

# =========================
# Deployment
# =========================

touch platform/deployment/model_deployer.py

# =========================
# Monitoring
# =========================

touch platform/monitoring/model_monitor.py

# =========================
# Pipeline
# =========================

touch platform/pipeline/pipeline.py
touch platform/pipeline/runner.py

# =========================
# Project layer
# =========================

mkdir -p project
touch project/config.yaml
touch project/project.py
touch project/train.py
touch project/predict.py

# =========================
# Tests
# =========================

mkdir -p tests
touch tests/__init__.py

# =========================
# Setup
# =========================

mkdir -p setup
touch setup/requirements.txt
touch setup/setup.sh

# =========================
# Templates
# =========================

mkdir -p templates

# =========================
# Root files
# =========================

touch .gitignore
touch README.md

echo
echo "========================================="
echo " MLOps platform structure created"
echo "========================================="
echo

find . -type f \
    -not -path "./.git/*" \
    -not -path "./venv/*" \
    | sort

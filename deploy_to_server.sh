#!/bin/bash
# Deployment script for HBT Analysis on lab server

echo "Deploying HBT Analysis to server..."

# Install GPU requirements
echo "Installing GPU requirements..."
pip install -r requirements_gpu.txt

# Run validation
echo "Running server validation..."
python scripts/server_validation.py

# Test GPU setup
echo "Testing GPU setup..."
python test_gpu_setup.py

echo "Deployment complete!"
echo "Run: python scripts/optimization/optimize_hbt_parameters_gpu.py --use_gpu"

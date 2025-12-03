#!/bin/bash
# Simple script to build documentation locally

set -e

echo "Building Tarang.jl documentation..."

# Navigate to repository root
cd "$(dirname "$0")/.."

# Install dependencies (only needed first time)
echo "Installing documentation dependencies..."
julia --project=docs/ -e 'using Pkg; Pkg.develop(PackageSpec(path=pwd())); Pkg.instantiate()'

# Build documentation
echo "Building documentation..."
julia --project=docs/ docs/make.jl

echo "Documentation built successfully!"
echo "To preview, run:"
echo "  cd docs/build && python3 -m http.server 8000"
echo "Then open http://localhost:8000 in your browser"

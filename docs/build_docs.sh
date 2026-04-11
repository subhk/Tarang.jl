#!/bin/bash
# Simple script to build documentation locally

set -e

echo "Building Tarang.jl documentation..."

# Navigate to repository root
cd "$(dirname "$0")/.."

# Clean old build artifacts to ensure fresh build
echo "Cleaning old build artifacts..."
rm -rf docs/build
rm -f docs/Manifest.toml

# Force re-develop to pick up latest code changes
echo "Installing documentation dependencies (forcing fresh install)..."
julia --project=docs/ -e '
using Pkg
# Remove old Tarang entry if it exists to force fresh develop
try
    Pkg.rm("Tarang")
catch
end
# Re-add local development version
Pkg.develop(PackageSpec(path=pwd()))
Pkg.instantiate()
# Force precompilation of fresh code
Pkg.precompile()
'

# Build documentation
echo "Building documentation..."
julia --project=docs/ docs/make.jl

echo "Documentation built successfully!"
echo "To preview, run:"
echo "  cd docs/build && python3 -m http.server 8000"
echo "Then open http://localhost:8000 in your browser"

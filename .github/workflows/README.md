# GitHub Actions Workflows

This directory contains GitHub Actions workflows for continuous integration and documentation deployment.

## Workflows

### CI.yml - Continuous Integration

Comprehensive testing workflow that runs on every push and pull request.

**Test Matrix:**
- Julia versions: 1.6 (LTS), 1.9 (Stable), 1.x (Latest)
- Operating systems: Ubuntu, macOS, Windows
- Architecture: x64

**Jobs:**

1. **test** - Main test suite
   - Installs MPI on all platforms
   - Runs the test suite with `Pkg.test()`
   - Generates coverage reports (Ubuntu + latest Julia only)
   - Uploads to Codecov

2. **docs** - Documentation build
   - Builds documentation with Documenter.jl
   - Deploys to gh-pages on main branch
   - Requires `DOCUMENTER_KEY` secret

3. **code-quality** - Code quality checks
   - Runs JuliaFormatter to check formatting
   - Can add other linting tools

4. **mpi-tests** - MPI-specific tests
   - Tests with different process counts (1, 2, 4, 8)
   - Ensures parallel code works correctly
   - Ubuntu only for consistency

5. **performance** - Performance benchmarks
   - Runs on main branch pushes only
   - Executes benchmark suite
   - Uploads results as artifacts

**Triggers:**
- Push to `main`, `master`, or `dev` branches
- Pull requests
- Manual workflow dispatch
- Git tags

**Environment Variables:**
- `JULIA_NUM_THREADS=2`: Enable Julia threading
- `OMP_NUM_THREADS=1`: Prevent OpenMP oversubscription

### Documentation.yml - Documentation Deployment

Builds and deploys documentation to GitHub Pages.

**When it runs:**
- Push to main branch
- Git tags
- Pull requests (build only, no deploy)

**Requirements:**
1. GitHub Pages enabled in repository settings
2. `DOCUMENTER_KEY` secret configured (see setup below)

**Output:** Documentation deployed to `https://<username>.github.io/<repo>/`

## Setup Instructions

### 1. Enable GitHub Actions

GitHub Actions are enabled by default for public repositories. For private repositories:
1. Go to Settings → Actions → General
2. Enable "Allow all actions and reusable workflows"

### 2. Configure Secrets

#### DOCUMENTER_KEY (Required for docs deployment)

Generate SSH key for Documenter:

```bash
julia -e 'using DocumenterTools; DocumenterTools.genkeys(user="subhajitkar", repo="Tarang.jl")'
```

This outputs two keys:

1. **Public key**: Add to Settings → Deploy Keys
   - Title: "Documenter"
   - Key: [paste public key]
   - ✅ Allow write access

2. **Private key**: Add to Settings → Secrets → Actions
   - Name: `DOCUMENTER_KEY`
   - Secret: [paste private key]

#### CODECOV_TOKEN (Optional for coverage)

If using Codecov for coverage reports:

1. Sign up at https://codecov.io with your GitHub account
2. Add your repository
3. Copy the token
4. Add to Settings → Secrets → Actions
   - Name: `CODECOV_TOKEN`
   - Secret: [paste token]

### 3. Configure GitHub Pages

For documentation deployment:

1. Go to Settings → Pages
2. Source: Deploy from a branch
3. Branch: `gh-pages`
4. Folder: `/` (root)
5. Save

After first successful documentation build, your docs will be at:
```
https://subhajitkar.github.io/Tarang.jl/
```

## Status Badges

Add these badges to your README.md:

```markdown
[![CI](https://github.com/subhajitkar/Tarang.jl/workflows/CI/badge.svg)](https://github.com/subhajitkar/Tarang.jl/actions/workflows/CI.yml)
[![Documentation](https://github.com/subhajitkar/Tarang.jl/workflows/Documentation/badge.svg)](https://github.com/subhajitkar/Tarang.jl/actions/workflows/Documentation.yml)
[![codecov](https://codecov.io/gh/subhajitkar/Tarang.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/subhajitkar/Tarang.jl)
```

## Workflow Customization

### Modify Test Matrix

Edit `.github/workflows/CI.yml`:

```yaml
matrix:
  version:
    - '1.6'  # Add/remove Julia versions
    - '1.9'
    - '1'
  os:
    - ubuntu-latest  # Add/remove operating systems
    - macOS-latest
    - windows-latest
```

### Add More MPI Process Counts

Edit the `mpi-tests` job:

```yaml
matrix:
  nprocs: [1, 2, 4, 8, 16]  # Add more process counts
```

### Disable Certain Jobs

Comment out or remove jobs you don't need:

```yaml
# code-quality:  # Commented out
#   name: Code Quality
#   ...
```

### Add Custom Test Arguments

In the test step:

```yaml
- name: Run tests
  run: julia --project -e 'using Pkg; Pkg.test(test_args=["--custom-arg"])'
```

## Troubleshooting

### Tests Failing on Specific OS

Check the specific job log:
1. Go to Actions tab
2. Click on the failed workflow run
3. Click on the failed job
4. Expand the failed step

Common issues:
- **macOS**: May need different MPI installation
- **Windows**: Path issues with MPI/HDF5
- **Linux**: Usually most stable

### Documentation Not Deploying

Check:
1. `DOCUMENTER_KEY` is correctly configured
2. GitHub Pages is enabled and set to `gh-pages` branch
3. Documentation build succeeds without errors
4. Check workflow logs for permission errors

### MPI Tests Failing

Common causes:
- MPI not installed correctly
- Wrong number of processes
- Process binding issues

Set environment variables:
```yaml
env:
  OMPI_ALLOW_RUN_AS_ROOT: 1
  OMPI_ALLOW_RUN_AS_ROOT_CONFIRM: 1
```

### Coverage Not Uploading

Ensure:
1. `CODECOV_TOKEN` is set (if repo is private)
2. Coverage step only runs on one job (to avoid duplicates)
3. Codecov repository is linked

## Local Testing

Test workflows locally before pushing:

### Using act

Install [act](https://github.com/nektos/act):

```bash
# macOS
brew install act

# Run CI workflow locally
act -j test
```

### Manual local testing

```bash
# Run tests locally with MPI
mpiexec -n 4 julia --project -e 'using Pkg; Pkg.test()'

# Build documentation locally
julia --project=docs/ -e 'using Pkg; Pkg.develop(PackageSpec(path=pwd())); Pkg.instantiate()'
julia --project=docs/ docs/make.jl
```

## Performance Optimization

### Cache Artifacts

Workflows use `julia-actions/cache@v1` to cache:
- Julia packages
- Compiled artifacts
- MPI builds

This significantly speeds up subsequent runs.

### Reduce Matrix Size

For faster CI, reduce the test matrix:

```yaml
# Minimal matrix
matrix:
  version: ['1']  # Only latest
  os: [ubuntu-latest]  # Only Linux
```

### Conditional Jobs

Run expensive jobs only on main branch:

```yaml
performance:
  if: github.ref == 'refs/heads/main'
  # ...
```

## Best Practices

1. **Keep workflows fast**: < 10 minutes for tests
2. **Use caching**: Speeds up subsequent runs
3. **Fail fast**: Set `fail-fast: false` in matrix
4. **Informative names**: Clear job and step names
5. **Timeouts**: Set reasonable timeouts to catch hangs
6. **Conditional execution**: Skip jobs when not needed

## Resources

- [GitHub Actions Documentation](https://docs.github.com/actions)
- [Julia Actions](https://github.com/julia-actions)
- [Documenter.jl CI Guide](https://juliadocs.github.io/Documenter.jl/stable/man/hosting/)
- [MPI.jl CI Examples](https://github.com/JuliaParallel/MPI.jl)

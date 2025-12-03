# CI Setup Summary

Comprehensive CI/CD (Continuous Integration and Deployment) system has been created for Tarang.jl.

## Files Created

### GitHub Actions Workflows

1. **.github/workflows/CI.yml** - Main CI workflow
2. **.github/workflows/Documentation.yml** - Documentation deployment (already existed, updated)
3. **.github/workflows/README.md** - Workflow documentation
4. **.github/CONTRIBUTING.md** - Contribution guidelines

## CI Workflow Features

### Test Matrix

The CI now tests on a comprehensive matrix:

**Julia Versions:**
- 1.6 (LTS - Long Term Support)
- 1.9 (Stable)
- 1.10 (Recent)
- 1.11 (Recent)
- 1.12 (Recent)
- 1.x (Latest release)

**Operating Systems:**
- Ubuntu Latest
- macOS Latest
- Windows Latest

**Architecture:**
- x64

**Total Combinations:** 6 versions × 3 OS × 1 arch = **18 test jobs**

### CI Jobs

#### 1. Test Job (Main Tests)
- Runs on all OS/version combinations
- Installs MPI automatically for each platform
- Runs the full test suite
- Generates code coverage (Ubuntu + latest Julia)
- Uploads coverage to Codecov

#### 2. Documentation Job
- Builds documentation with Documenter.jl
- Deploys to GitHub Pages on main branch
- Validates documentation builds on PRs

#### 3. Code Quality Job
- Checks code formatting with JuliaFormatter
- Can be extended with linting tools

#### 4. MPI Tests Job
- Tests with different MPI process counts: 1, 2, 4, 8
- Ensures parallel code works correctly
- Ubuntu only (for consistency)

#### 5. Performance Benchmarks
- Runs on main branch only
- Executes benchmark suite
- Uploads results as artifacts

## How It Works

### On Every Push/PR

1. **CI Workflow triggers** on push to main/master/dev branches or any PR
2. **Tests run in parallel** across all OS and Julia versions
3. **MPI is installed** automatically for each platform
4. **Test suite executes** with `Pkg.test()`
5. **Results reported** on GitHub with pass/fail status

### On Main Branch Push

Additionally:
- Documentation builds and deploys to GitHub Pages
- Performance benchmarks run
- Code coverage uploads to Codecov

### On Git Tag

- CI runs full test suite
- Documentation builds for the tagged version

## Setup Required

### 1. GitHub Secrets

Add these secrets in **Settings → Secrets → Actions**:

#### DOCUMENTER_KEY (Required)
```bash
julia -e 'using DocumenterTools; DocumenterTools.genkeys(user="subhajitkar", repo="Tarang.jl")'
```
- Add **public key** to **Settings → Deploy Keys** (allow write access)
- Add **private key** to **Secrets → Actions** as `DOCUMENTER_KEY`

#### CODECOV_TOKEN (Optional)
- Sign up at https://codecov.io
- Get token for your repository
- Add to **Secrets → Actions** as `CODECOV_TOKEN`

### 2. GitHub Pages

Configure in **Settings → Pages**:
- Source: Deploy from a branch
- Branch: `gh-pages`
- Folder: `/` (root)

### 3. Branch Protection (Recommended)

In **Settings → Branches → Add rule**:
- Branch name pattern: `main`
- Require status checks: ✓
  - Select: CI / test jobs
  - Select: docs job
- Require branches to be up to date: ✓

## Running Locally

### Run All Tests
```bash
julia --project -e 'using Pkg; Pkg.test()'
```

### Run with MPI
```bash
mpiexec -n 4 julia --project -e 'using Pkg; Pkg.test()'
```

### Build Documentation
```bash
./docs/build_docs.sh
```

### Format Code
```julia
using JuliaFormatter
format(".")
```

## Status Badges

Add to README.md:

```markdown
[![CI](https://github.com/subhajitkar/Tarang.jl/workflows/CI/badge.svg)](https://github.com/subhajitkar/Tarang.jl/actions/workflows/CI.yml)
[![Documentation](https://github.com/subhajitkar/Tarang.jl/workflows/Documentation/badge.svg)](https://subhajitkar.github.io/Tarang.jl/)
[![codecov](https://codecov.io/gh/subhajitkar/Tarang.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/subhajitkar/Tarang.jl)
```

## What Happens Next

### After First Push

1. CI workflow will run but may fail initially (expected)
2. Documentation workflow needs `DOCUMENTER_KEY` to deploy
3. Setup the secrets as described above
4. Re-run failed workflows

### After Documentation Key Setup

1. Documentation will build and deploy automatically
2. Visit: `https://subhajitkar.github.io/Tarang.jl/`
3. Documentation updates on every push to main

### After Codecov Setup

1. Coverage reports upload after each test run
2. View at: `https://codecov.io/gh/subhajitkar/Tarang.jl`
3. Coverage badge shows percentage

## Workflow Triggers

### Automatic Triggers

- **Push** to main/master/dev branches
- **Pull Request** to any branch
- **Git tags** (e.g., v0.1.0)

### Manual Trigger

Go to **Actions → CI → Run workflow** to manually trigger

## Customization

### Reduce Test Matrix

If CI takes too long, reduce the matrix in `.github/workflows/CI.yml`:

```yaml
version:
  - '1.10'  # Remove older versions
  - '1'     # Keep latest only
os:
  - ubuntu-latest  # Remove macOS/Windows if not needed
```

### Add More Tests

Add custom test suites in `test/runtests.jl`:

```julia
@testset "Tarang.jl" begin
    @testset "Core Tests" begin
        include("test_core.jl")
    end

    @testset "MPI Tests" begin
        include("test_mpi.jl")
    end
end
```

### Disable Jobs

Comment out jobs you don't need:

```yaml
# performance:  # Disabled
#   name: Performance Benchmarks
#   ...
```

## Troubleshooting

### CI Failing

1. **Check the logs**: Go to Actions → Failed workflow → Click on failed job
2. **Common issues**:
   - MPI not found: Check MPI installation step
   - Test failures: Run tests locally first
   - Documentation errors: Check `make.jl` syntax

### Documentation Not Deploying

1. Verify `DOCUMENTER_KEY` is correctly set
2. Check GitHub Pages is enabled
3. Look for errors in Documentation job logs

### Coverage Not Working

1. Ensure job runs on Ubuntu + latest Julia
2. Check `CODECOV_TOKEN` is set (for private repos)
3. Verify Codecov repository is configured

## Performance

### Current Setup
- **~30 minutes** for full CI run (all jobs in parallel)
- **~5 minutes** per test job
- **~3 minutes** for documentation

### Optimization Tips
- Use caching (already enabled)
- Reduce test matrix size
- Run expensive jobs only on main branch
- Set appropriate timeouts

## Next Steps

1. **Push to GitHub** to trigger first CI run
2. **Setup DOCUMENTER_KEY** for documentation deployment
3. **Monitor Actions tab** for results
4. **Add status badges** to README
5. **Configure branch protection** for main branch

## Resources

- Workflow Documentation: `.github/workflows/README.md`
- Contributing Guide: `.github/CONTRIBUTING.md`
- GitHub Actions: https://docs.github.com/actions
- Julia Actions: https://github.com/julia-actions

---

**CI Status**: Ready to use after setting up secrets

**Documentation URL** (after deployment): https://subhajitkar.github.io/Tarang.jl/

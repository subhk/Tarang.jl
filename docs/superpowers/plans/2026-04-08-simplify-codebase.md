# Simplify Codebase Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Split 3 large files (field.jl, problems.jl, solvers.jl) into focused sub-files, flatten deeply nested functions, and remove duplicated code patterns.

**Architecture:** Follow the existing hub-file pattern used by `operators/operators.jl` and `timesteppers/timesteppers.jl` — each large file becomes a directory with a hub file that includes sub-files in dependency order. No logic changes, just moving code.

**Tech Stack:** Julia, no new dependencies

---

## Task 1: Split `field.jl` (3032 lines → 4 files)

**Files:**
- Create: `src/core/field/field_types.jl`
- Create: `src/core/field/field_data.jl`
- Create: `src/core/field/field_layout.jl`
- Create: `src/core/field/field_operations.jl`
- Replace: `src/core/field.jl` → hub file (includes the 4 sub-files)

Split boundaries (based on existing `# ====` section markers in the file):

| New file | Lines from field.jl | What it contains |
|----------|-------------------|------------------|
| `field_types.jl` | 1–310 | Abstract types, storage modes, ScalarField, VectorField, TensorField, LockedField struct definitions and constructors |
| `field_data.jl` | 311–1225 | Property access, component stack/unstack, data allocation, distributed shape computation |
| `field_layout.jl` | 1226–2180 | Scale management, layout transitions, ensure_layout!, forward/backward transforms |
| `field_operations.jl` | 2181–3031 | fill_random, integrate, arithmetic, filtering, unit vectors, exports |

- [ ] **Step 1:** Create directory `src/core/field/`
- [ ] **Step 2:** Copy lines 1–310 from `field.jl` into `src/core/field/field_types.jl`
- [ ] **Step 3:** Copy lines 311–1225 into `src/core/field/field_data.jl`
- [ ] **Step 4:** Copy lines 1226–2180 into `src/core/field/field_layout.jl`
- [ ] **Step 5:** Copy lines 2181–3031 into `src/core/field/field_operations.jl`
- [ ] **Step 6:** Replace `src/core/field.jl` with a hub file:

```julia
"""
Field types and operations for Tarang.jl

Split into sub-files for readability:
- field_types.jl: ScalarField, VectorField, TensorField definitions
- field_data.jl: Data access, allocation, distributed shapes
- field_layout.jl: Layout transitions, transforms
- field_operations.jl: Arithmetic, integration, filtering
"""

include("field/field_types.jl")
include("field/field_data.jl")
include("field/field_layout.jl")
include("field/field_operations.jl")
```

- [ ] **Step 7:** Verify no test breaks — run: `julia --project -e 'using Tarang; println("OK")'`

---

## Task 2: Split `problems.jl` (3419 lines → 4 files)

**Files:**
- Create: `src/core/problems/problem_types.jl`
- Create: `src/core/problems/problem_parsing.jl`
- Create: `src/core/problems/problem_matrices.jl`
- Create: `src/core/problems/problem_utils.jl`
- Replace: `src/core/problems.jl` → hub file

Split boundaries:

| New file | Lines from problems.jl | What it contains |
|----------|----------------------|------------------|
| `problem_types.jl` | 1–619 | IVP/LBVP/NLBVP/EVP definitions, constructors, add_equation!, add_bc!, stochastic forcing, parameter management |
| `problem_parsing.jl` | 620–1620 | Expression parsing, evaluation, helper operators (ZeroOperator, ConstantOperator), coerce_constant_value |
| `problem_matrices.jl` | 1621–2605 | Matrix building, build_expression_matrix_block, RHS forcing vectors, operator processing |
| `problem_utils.jl` | 2606–3419 | Domain setup, validation, substitution, expression_to_string, namespace handling, introspection, exports |

- [ ] **Step 1:** Create directory `src/core/problems/`
- [ ] **Step 2–5:** Copy each section into its file
- [ ] **Step 6:** Replace `src/core/problems.jl` with hub file:

```julia
"""
Problem definitions and equation parsing for Tarang.jl

Split into sub-files for readability:
- problem_types.jl: IVP, LBVP, NLBVP, EVP definitions
- problem_parsing.jl: Expression parsing and evaluation
- problem_matrices.jl: Matrix building for solvers
- problem_utils.jl: Validation, substitution, introspection
"""

include("problems/problem_types.jl")
include("problems/problem_parsing.jl")
include("problems/problem_matrices.jl")
include("problems/problem_utils.jl")
```

- [ ] **Step 7:** Verify — run: `julia --project -e 'using Tarang; println("OK")'`

---

## Task 3: Split `solvers.jl` (2518 lines → 4 files)

**Files:**
- Create: `src/core/solvers/solver_types.jl`
- Create: `src/core/solvers/solver_stepping.jl`
- Create: `src/core/solvers/solver_compiled_rhs.jl`
- Create: `src/core/solvers/solver_utils.jl`
- Replace: `src/core/solvers.jl` → hub file

Split boundaries:

| New file | Lines from solvers.jl | What it contains |
|----------|---------------------|------------------|
| `solver_types.jl` | 1–670 | All type definitions (Solver, SolverBaseData, RHS instructions, InitialValueSolver, BoundaryValueSolver, EigenvalueSolver), constructors, dispatch |
| `solver_stepping.jl` | 671–963 | step!, proceed, BC application, linear/nonlinear solve, eigenvalue solve |
| `solver_compiled_rhs.jl` | 964–2348 | Vector ↔ field conversions, expression evaluation, RHS compilation, instruction execution |
| `solver_utils.jl` | 2349–2518 | Diagnostics, performance logging, exports |

- [ ] **Step 1:** Create directory `src/core/solvers/`
- [ ] **Step 2–5:** Copy each section into its file
- [ ] **Step 6:** Replace `src/core/solvers.jl` with hub file:

```julia
"""
Solver implementations for Tarang.jl

Split into sub-files for readability:
- solver_types.jl: Solver definitions and constructors
- solver_stepping.jl: Time stepping, BVP/EVP solve
- solver_compiled_rhs.jl: RHS compilation and execution
- solver_utils.jl: Diagnostics and exports
"""

include("solvers/solver_types.jl")
include("solvers/solver_stepping.jl")
include("solvers/solver_compiled_rhs.jl")
include("solvers/solver_utils.jl")
```

- [ ] **Step 7:** Verify — run: `julia --project -e 'using Tarang; println("OK")'`

---

## Task 4: Flatten `expression_to_string` using dispatch

**Files:**
- Modify: `src/core/problems/problem_utils.jl` (after Task 2 split)

Replace the 16-branch if/elseif chain (lines 2956–3010 of original problems.jl) with Julia multiple dispatch:

- [ ] **Step 1:** Replace `expression_to_string` with dispatched methods:

```julia
# Base cases
expression_to_string(expr::String) = expr
expression_to_string(expr::Number) = string(expr)
expression_to_string(::ZeroOperator) = "0"
expression_to_string(expr::ConstantOperator) = string(expr.value)

# Binary operators
expression_to_string(expr::AddOperator) = "($(expression_to_string(expr.left)) + $(expression_to_string(expr.right)))"
expression_to_string(expr::SubtractOperator) = "($(expression_to_string(expr.left)) - $(expression_to_string(expr.right)))"
expression_to_string(expr::MultiplyOperator) = "($(expression_to_string(expr.left)) * $(expression_to_string(expr.right)))"
expression_to_string(expr::DivideOperator) = "($(expression_to_string(expr.left)) / $(expression_to_string(expr.right)))"
expression_to_string(expr::PowerOperator) = "($(expression_to_string(expr.left)) ^ $(expression_to_string(expr.right)))"

# Unary operators
expression_to_string(expr::NegateOperator) = "(-$(expression_to_string(expr.operand)))"
expression_to_string(expr::TimeDerivative) = "∂t($(expression_to_string(expr.operand)))"
expression_to_string(expr::Laplacian) = "Δ($(expression_to_string(expr.operand)))"
expression_to_string(expr::Gradient) = "∇($(expression_to_string(expr.operand)))"
expression_to_string(expr::Divergence) = "∇·($(expression_to_string(expr.operand)))"

function expression_to_string(expr::Differentiate)
    coord_name = hasfield(typeof(expr.coord), :name) ? expr.coord.name : string(expr.coord)
    return "d($(expression_to_string(expr.operand)), $coord_name)"
end

# Fallback
expression_to_string(expr) = hasfield(typeof(expr), :name) ? expr.name : string(expr)
```

- [ ] **Step 2:** Verify — run: `julia --project -e 'using Tarang; println("OK")'`

---

## Task 5: Replace duplicated `coerce_constant_value` inline patterns

**Files:**
- Modify: `src/core/problems/problem_matrices.jl` (after Task 2 split)

- [ ] **Step 1:** Replace 3 inline duplicates with the existing helper.

At line 2306 of original problems.jl (now in problem_matrices.jl), change:
```julia
# Before:
coeff = isa(expr.right, ConstantOperator) ? expr.right.value : expr.right
# After:
coeff = coerce_constant_value(expr.right)
```

At line 2310:
```julia
# Before:
coeff = isa(expr.left, ConstantOperator) ? expr.left.value : expr.left
# After:
coeff = coerce_constant_value(expr.left)
```

At line 2321:
```julia
# Before:
denom = isa(expr.right, ConstantOperator) ? expr.right.value : expr.right
# After:
denom = coerce_constant_value(expr.right)
```

- [ ] **Step 2:** Verify — run: `julia --project -e 'using Tarang; println("OK")'`

---

## Verification

After all tasks, run:
```bash
julia --project -e 'using Pkg; Pkg.test()'
```

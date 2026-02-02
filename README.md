# SobolSampling.jl

[![Code Style: Blue](https://img.shields.io/badge/code%20style-blue-4495d1.svg)](https://github.com/JuliaDiff/BlueStyle)
[![Aqua QA](https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg)](https://github.com/JuliaTesting/Aqua.jl)
[![Julia](https://img.shields.io/badge/julia-1.9%20%7C%201.10%20%7C%201.11-9558B2.svg)](https://julialang.org)

**SobolSampling.jl** provides a Sobol sequence-based hyperparameter tuning strategy for [MLJ](https://github.com/JuliaAI/MLJ.jl). It uses low-discrepancy quasi-random sampling to efficiently explore hyperparameter spaces with better coverage than pure random search.

## Overview

Sobol sequences are quasi-random low-discrepancy sequences that provide more uniform coverage of the search space compared to random sampling. This leads to more efficient hyperparameter exploration, especially in higher dimensions where Sobol sequences significantly outperform random search.

### Key Features

- **Low-discrepancy sampling**: Better space-filling properties than random search
- **Efficient exploration**: Particularly effective in moderate to high-dimensional spaces (d ≤ 20)
- **Deterministic by default**: Reproducible results with optional random shifting
- **Flexible range support**: Handles both numeric and nominal hyperparameters
- **Log-scale support**: Proper handling of log-scale ranges for parameters spanning orders of magnitude
- **MLJTuning integration**: Drop-in replacement for other tuning strategies (Grid, RandomSearch, LatinHypercube)

### When to Use Sobol Sequences

Sobol sequences are particularly valuable when:
- You have a moderate number of hyperparameters (2-20 dimensions)
- You want better coverage than random search without the curse of dimensionality from grid search
- You need deterministic, reproducible hyperparameter exploration
- Budget is limited and you want to maximise information per evaluation

### Sobol Sequence Properties

- **Skip parameter**: Controls initial point skipping (`:auto` uses 2^m - 1 for optimal properties)
- **Random shifts**: Optional Cranley-Patterson shifts for scrambling-like behaviour
- **Unbounded ranges**: Automatic finite-bounds heuristics using `origin` and `unit` parameters

## Installation

```julia
using Pkg
Pkg.add("SobolSampling")
```

Or for development:

```julia
using Pkg
Pkg.develop(path="/path/to/SobolSampling.jl")
```

## Quick Start

```julia
using MLJBase
using MLJTuning
using MLJDecisionTreeInterface
using StatisticalMeasures
using SobolSampling

# Load data
X, y = @load_iris

# Define model and hyperparameter ranges
model = DecisionTreeClassifier()
r1 = range(model, :max_depth; lower=1, upper=10)
r2 = range(model, :min_samples_split; lower=2, upper=50)
r3 = range(model, :min_samples_leaf; lower=1, upper=20)

# Create tuned model with Sobol sequence sampling
tuned_model = TunedModel(
    model=model,
    tuning=SobolSequence(skip=:auto, random_shift=false),
    range=[r1, r2, r3],
    resampling=CV(nfolds=5),
    measure=LogLoss(),
    n=128  # Evaluate 128 Sobol points
)

# Fit and select best hyperparameters
mach = machine(tuned_model, X, y)
fit!(mach)

# Inspect results
report(mach).best_model
report(mach).best_history_entry
```

## Licence

This software is distributed under the MIT Licence.

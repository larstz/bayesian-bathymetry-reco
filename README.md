# Bayesian Inverse Problems

This repository contains code and resources for solving inverse problems in a
statistical fashion using Bayes theorem. Bayesian inverse problems involve using statistical methods to infer unknown parameters in a system from observed data.

## Table of Contents
- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)


## Introduction

To study the functionality of bayesian inference techniques, in particular Markov chain monte carlo (MCMC),
we consider bathymetry reconstruction as a toy experiment.

## Installation
Needed:
- Working Julia
- Working Conda/Micromamba etc.

Setting up:
1. Clone the repository:
    ```bash
    git clone https://collaborating.tuhh.de/l_stz/bayesian-inverse-problems
    ```
2. Navigate to the project directory:
    ```bash
    cd bayesian-inverse-problems
    ```
3. micromamba create -f requirements.yml
4. Setup julia environment `julia -e 'using Pkg; Pkg.activate("."); Pkg.develop(path="BathymetryReco")'`
5. Setup PyCall to work with the correct python `julia -e 'using Pkg; Pkg.activate("."); using PyCall; ENV["PYTHON"]="path/to/env/bin/python"; Pkg.build("PyCall");'`
6. To setup dedalus correctly create file state at "myenv/conda-meta" and fill it with `"{"env_vars": {"OMP_NUM_THREADS": "1", "NUMEXPR_MAX_THREADS": "1"}}"`

## Usage

1. Create toy measurments by running the `julia toy_measurement.jl` script. The simulation parameters can be configured in simulation_config.toml
4. For the parameter inference run `julia mcmc_reconstruction.jl` with the setup defined by config.toml
5. After inference the inference data is stored (if save=true in config.toml)
6. To plot the results run `julia plots.jl path/to/exeperiment/data`


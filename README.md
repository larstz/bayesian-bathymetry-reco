# Bayesian Inverse Problems

This repository contains code and resources for solving inverse problems in a
statistical fashion using Bayes theorem. Bayesian inverse problems involve using statistical methods to infer unknown parameters in a system from observed data. In particular the inverse problem of bathymetry reconstruction is considered.

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

## Basic Usage

0. If you want to use the experimental measurements  download [Angel et al. (2024)](https://doi.org/10.15480/882.9403), transform the data into CSV format using `bash exp_files_to_csv.sh` and calculate the mean by using `julia analyse_experiment_data.jl`
1. Create toy measurments by running the `julia toy_measurement.jl {simulation_config.toml}` script. The simulation parameters can be configured in simulation_config.toml
4. For the parameter inference run `julia mcmc_reconstruction.jl {config.toml}` with the setup defined by config.toml for the parameterized bathymetry and `julia mcmc_reconstruction_serial.jl {config.toml}` for the discretized bathymetry.
5. After inference the inference data is stored (if save=true in config.toml)
6. To plot the results run `julia plots.jl path/to/exeperiment/data` for the parameterized bathymetry, `julia plot_discrete_bathymetry.jl path/to/exeperiment/data` for the discretized bathymetry.

## How to reproduce the data and figures

Follow step 0. from basic usage before starting

### Figure 2
1. Create the synthetic measurements by running `julia toy_measurement.jl paper_configs/simulation_configs/width_test/4_0_peak_0_01_width_simulation.toml`. Repeat this step for all other simulation configurations in `paper_configs/simulation_configs/width_test/`.
2. Run the reconstruction by calling `julia mcmc_reconstruction.jl paper/configs/parametrized/width_test/4_0_peak_0_01_width.toml`. Repeat this tesp for all other reconstruction configs in `paper/configs/parametrized/width_test/`.
3. After finishing the reconstruction of all possible widths, call `julia evaluate_chains.jl data/results/paper_results/width_test/toy_tests/sensor-2-3-4/prior-uniform-uniform/proposal-rw/stepsize-0.1-0.01/`

### Figure 3
Same as Figure 2 but use the configs in paper
`paper_configs/simulation_configs/peak_test/` for simulation and `paper/configs/parametrized/peak_test/` for reconstruction.

### Figure 4
1. Create the synthetic measurement by calling `julia toy_measurement.jl paper_configs/simulation_configs/simulation_config_toy.jl`
2. Create the landscape scan by calling `julia scan_lp.jl`
3. Run the reconstruction with uniform priors by calling `julia mcmc_reconstruction.jl paper_configs/parametrized/parameterized_uniform_prior_config.toml`
4. Create the figure with `julia plot_lp.jl data/results/lp_scan/lp_scan_{DATE_ID} data/results/paper_results/ data/results/paper_results/parametrized/toy_tests/sensor-2-3-4/prior-uniform-uniform/proposal-rw/stepsize-0.1-0.01/{DATE_ID}_waterchannel_exact_bathy`

### Figure 5
1. 1.-2. same as Figure 4 but no need to repeat if already done.
3. Run the reconstruction with normal prior by calling `julia mcmc_reconstruction.jl paper_configs/parametrized/parameterized_normal_prior_config.toml`
4. Create the figure with `julia plot_lp.jl data/results/lp_scan/lp_scan_{DATE_ID} data/results/paper_results/ data/results/paper_results/parametrized/toy_tests/sensor-2-3-4/prior-normal-uniform/proposal-rw/stepsize-0.1-0.01/{DATE_ID}_waterchannel_exact_bathy`

### Figure 6
1. If not created reconstruction for Figure 5, follow step 3. from Figure 5
2. To create the Figure call `julia plots.jl data/results/paper_results/ data/results/paper_results/parametrized/toy_tests/sensor-2-3-4/prior-normal-uniform/proposal-rw/stepsize-0.1-0.01/{DATE_ID}_waterchannel_exact_bathy`. The Figure corresponding to the one in the paper can then be found in the experiment directory under `plots/pdfs/mean_bathy_credible_interval_2_bi_200.pdf`.

### Figure 7
1. If not created synthetic measurement follow Figure 4 1. step
2. Run `julia mcmc_reconstruction_serial.jl paper_configs/discretized/toy_smooth_sparse_rw_smooth.toml`
3. Call `julia plot_discrete_experiment.jl data/results/paper_results/toy_tests/sensor-2-3-4/prior-smooth-sparse/proposal-rw-smooth/stepsize-0.002/{DATE_ID}_waterchannel_exact_bathy`. The plot can then be found in the experiment directory under `plots/mean_bathy_credible_interval_1000.pdf`

### Figure 8
1. If not done so follow step 0. from the Basic usage section.
2. Run `julia mcmc_reconstruction_serial.jl paper_configs/discretized/mean_heat_config.toml`
3. Call `julia plot_discrete_experiment.jl data/results/paper_results/heat_tests/mean_tests/sensor-2-3-4/prior-smooth-sparse/proposal-rw-smooth/stepsize-0.001/{DATE_ID}_mean_heat_wb`. The plot can then be found in the experiment directory under `plots/mean_bathy_credible_interval_1000.pdf`

### Table 1
1. Follow steps 1. & 2. from Figure 7 & Figure 8
2. Run `julia mcmc_reconstruction_serial.jl paper_configs/discretized/toy_smooth_sparse_rw_smooth.toml`
3. Run `julia plot_discrete_experiment.jl data/results/paper_results/toy_tests/sensor-2-3-4/prior-smooth-sparse/proposal-rw-/stepsize-0.002/{DATE_ID}_waterchannel_exact_bathy`
4. Run `julia mcmc_reconstruction_serial.jlpaper_configs/discretized/toy_sparse_rw_smooth.toml
5. Run `julia plot_discrete_experiment.jl data/results/paper_results/toy_tests/sensor-2-3-4/prior-sparse/proposal-rw-smooth/stepsize-0.002/{DATE_ID}_waterchannel_exact_bathy`
6. The values from the Table can be found in `metrics_1000.csv` in the corresponding experiment directory

### Reference

Angel, J., Behrens, J., Götschel, S., Hollm, M., Ruprecht, D., & Seifried, R. (2024). Data artefact: Bathymetry reconstruction from experimental data with PDE-constrained optimisation. https://doi.org/10.15480/882.9403
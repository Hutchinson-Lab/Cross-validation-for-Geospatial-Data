# Cross-validation-for-Geospatial-Data
This repository provides datasets and code for the paper _["Cross-validation for Geospatial Data: A Framework for Estimating Generalization Performance in Geostatistical Problems".](https://openreview.net/forum?id=VgJhYu7FmQ)_

## Datasets
We provided six simulation datasets and 15 real datasets.
The following abbreviations are used as `[dataset name]` in command line.
* Simulation: sim_sd, sim_si, sim_sdcs, sim_sics, sim_sirs, sim_sipcs
* HEWA1800: hewa1800_sd, hewa1800_si, hewa1800_sdcs, hewa1800_sics
* HEWA1000: hewa1000_sd, hewa1000_si, hewa1000_sdcs, hewa1000_sics
* WETA1800: weta1800_sd, weta1800_si, weta1800_sdcs, weta1800_sics
* Alaska: alaska
* Housing: house_bay, house_latitude

## Basic usage
Please read [`requirements`](./requirements.txt) file and install required packages first.

To get test errors and cross-validation (CV) estimates of five methods - standard K-Fold CV (KFCV), BLocking CV (BLCV), BuFfered CV (BFCV), Importance-Weighted CV (IWCV) and Importance-weighted Buffered CV (IBCV) on a specific dataset:
```
python run.py --dataset [dataset name]
```
For example, the command below saves the test errors and five CV estimates on the Simulation Scenario SD dataset as a csv file.
```
python run.py --dataset sim_sd
```

## Options
We also provide three scripts for those who are interested in exploring more.
* [`gen_sim`](./gen_sim.jl): It produces the simulation datasets. Users can generate simulations as many as they want by `sim`, and change the number of sampling points and sampling strategy as well.
* [`bcv`](./bcv.r): It splits the training set into blocks based on their geo coordinateusers, and then assign blocks into folds for cross-validation. Users can fine-tune the hyperparameters the number of folds by `k` and the block size by `bs`.
* [`cramer`](./cramer.r): It performs the statistical test on training and test features and reports the statistics and p value. Users can set a different significance level by `alpha`.
# Cross-validation-for-Geospatial-Data
This repository hosts datasets and code for the paper _["Cross-validation for Geospatial Data: A Framework for Estimating Generalization Performance in Geostatistical Problems".](https://openreview.net/forum?id=VgJhYu7FmQ)_
We compared the performance of five cross-validation (CV) methods - standard K-Fold CV (KFCV), BLocking CV (BLCV), BuFfered CV (BFCV), Importance-Weighted CV (IWCV) and our proposed Importance-weighted Buffered CV (IBCV) - in various geospatial scenarios.

## Datasets
We provided six simulation datasets and 15 real datasets.
The following abbreviations serve as `[dataset name]` in a command line.
* Simulation: sim_sd, sim_si, sim_sdcs, sim_sics, sim_sirs, sim_sipcs
* HEWA1800: hewa1800_sd, hewa1800_si, hewa1800_sdcs, hewa1800_sics
* HEWA1000: hewa1000_sd, hewa1000_si, hewa1000_sdcs, hewa1000_sics
* WETA1800: weta1800_sd, weta1800_si, weta1800_sdcs, weta1800_sics
* Alaska: alaska
* Housing: house_bay, house_latitude

## Environment Installation
To run the code, install the dependencies in [`requirements`](./requirements.txt).
```
python install -r requirements.txt
```

## Basic usage
To compute model errors and their estimates of five CV methods on a specific dataset:
```
python run.py --dataset [dataset name]
```
Take the Simulation Scenario SD (sim_sd) dataset for example:
```
python run.py --dataset sim_sd
```
The results will be saved in a csv file automatically.

## Options
To run any of the following scripts, please install the dependencies in [`requirements_extra`](./requirements_extra.txt) first.
* [`gen_sim`](./gen_sim.jl): It produces the simulation datasets. Users can generate simulations as many as they want by `sim`, and change the number of sampling points and sampling strategy as well.
* [`bcv`](./bcv.r): It splits the training set into blocks based on their geocoordinates, and then assign blocks into folds for cross-validation. Users can fine-tune the hyperparameters the number of folds by `k` and the block size by `bs`.
* [`cramer`](./cramer.r): It performs the statistical test on training and test features and reports the statistics and p value. Users can set the significance level by `alpha`.


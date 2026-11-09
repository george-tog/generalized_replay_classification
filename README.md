# generalized_replay_classification

> This is a fork of [replay_trajectory_classification](https://github.com/Eden-Kramer-Lab/replay_trajectory_classification) by Eric Denovellis and colleagues in the Eden and Frank labs. It adds `TCM_SR`, a continuous dynamic that models replay as movement along the routes the animal actually ran, and fits its parameters to replay by expectation maximization. The rest of the package is upstream's. The Python package is still imported as `replay_trajectory_classification`.
>
> If you use this code, please cite both [our paper](#this-fork) and [Denovellis et al. (2021)](#upstream).
>
> Report issues with the TCM-SR code in [this repository's issues](https://github.com/george-tog/generalized_replay_classification/issues), not upstream.
>
> Maintained by George Cai (<georgecai@g.harvard.edu>).

[**What this fork adds**](#what-this-fork-adds)
| [**Installation**](#installation)
| [**Tutorials**](#tutorials)
| [**References**](#references)
| [**Developer Installation**](#developer-installation)

## What this fork adds

- **`TCM_SR`**, a continuous transition built from the animal's empirical movement matrix `P`. It mixes a successor representation `M = (I - gamma * P)^-1 @ P` with a symmetric retrieved-context term `M.T @ M`. `gamma` sets how far ahead the kernel looks, and `asymmetry` how strongly it favors the direction of travel. It can replace `RandomWalk` in any classifier, and `RandomWalk` is the special case `TCM_SR(gamma=0, beta=1, asymmetry=1)`.
- **EM for continuous transitions.** `estimate_parameters(estimate_continuous_transition=True)` fits the parameters named in a transition's `learn` field. `score_continuous_transition` gives the data log likelihood at any parameter values, which is useful for checking a fit.
- **`continuous_transition_is_training`**, a new argument to `fit` on every classifier and decoder. The movement model can then include stops, such as at reward wells, while place fields are fit on running only.
- **Simulation helpers** in `sorted_spikes_simulation.py`: a directed-loop position coordinate and TCM-SR replay simulators.
- **Notebooks** in [notebooks/tcm_sr](#tcm-sr-notebooks).
- **Fixes to upstream code:**
  - `EmpiricalMovement` runs on NumPy 2.
  - The default `encoding_group` of `EmpiricalMovement` is now `0` instead of `None`. `None` selected no data.
  - `environments.py` works with pandas 3.

Apart from the `EmpiricalMovement` fix, the existing classes, defaults and tutorial notebooks behave as they do upstream.

## What is replay_trajectory_classification? (upstream)

`replay_trajectory_classification` is a python package for decoding spatial position represented by neural activity and categorizing the type of trajectory.

<p align="center">
  <img src="docs/_static/fra_11_04_0001.gif" height="400"></img>
</p>

### Advantages over other algorithms

It has several advantages over decoders typically used to characterize hippocampal data:

1. It allows for moment-by-moment estimation of position using small temporal time bins which allow for rapid movement of neural position and makes fewer assumptions about what downstream cells can integrate.
2. The decoded trajectories can change direction and are not restricted to constant velocity trajectories.
3. The decoder can use spikes from spike-sorted cells or use clusterless spikes and their associated waveform features to decode .
4. The decoder can categorize the type of neural trajectory and give an estimate of the confidence of the model in the type of trajectory.
5. Proper handling of complex 1D linearized environments
6. Ability to extract and decode 2D environments
7. Easily installable, documented code with tutorials on how to use the code (see below)
8. Fast computation using GPUs. (Note: must install `cupy` to use)

## Installation

This fork is not on PyPI or conda. Install it from this repository:

```bash
git clone https://github.com/george-tog/generalized_replay_classification.git
cd generalized_replay_classification
conda env create -f environment.yml
conda activate replay_trajectory_classification
pip install -e .
```

The `conda install -c edeno replay_trajectory_classification` and `pip install replay_trajectory_classification` packages are upstream's and do not include `TCM_SR`.

**Requirements**: Python 3.10+ with scientific computing stack (NumPy, SciPy, pandas, etc.)

### Documentation

The upstream documentation covers everything except the additions above: <https://replay-trajectory-classification.readthedocs.io/en/latest/>. The additions are documented in their docstrings and in the TCM-SR notebooks.

## Tutorials

There are five jupyter notebooks introducing the package:

1. [01-Introduction_and_Data_Format](notebooks/tutorial/01-Introduction_and_Data_Format.ipynb): How to get your data in the correct format to use with the decoder.
2. [02-Decoding_with_Sorted_Spikes](notebooks/tutorial/02-Decoding_with_Sorted_Spikes.ipynb): How to decode using a single movement model using sorted spikes.
3. [03-Decoding_with_Clusterless_Spikes](notebooks/tutorial/03-Decoding_with_Clusterless_Spikes.ipynb): How to decode using a single movement model using the "clusterless" approach --- which does not require spike sorting.
4. [04-Classifying_with_Sorted_Spikes](notebooks/tutorial/04-Classifying_with_Sorted_Spikes.ipynb): Using multiple movement models to classify the movement dynamics and decode the trajectory using sorted spikes.
5. [05-Classifying_with_Clusterless_Spikes](notebooks/tutorial/05-Classifying_with_Clusterless_Spikes.ipynb): Using multiple movement models to classify the movement dynamics and decode the trajectory using clusterless spikes.

### TCM-SR notebooks

In [notebooks/tcm_sr](notebooks/tcm_sr):

1. [Classifying_with_Sorted_Spikes_TCM_SR](notebooks/tcm_sr/Classifying_with_Sorted_Spikes_TCM_SR.ipynb): Simulated replay on a directed linear track. Classifies sweeps, TCM-SR sequences, random jumps and stationary events, and compares `TCM_SR` with `RandomWalk`.
2. [Fitting_TCM_SR_with_EM](notebooks/tcm_sr/Fitting_TCM_SR_with_EM.ipynb): Recovers `gamma` and `asymmetry` by EM from simulated replay, and checks the fit against a grid of the marginal likelihood.
3. [Fitting_TCM_SR_on_Pfeiffer_Rat1_Linear1](notebooks/tcm_sr/Fitting_TCM_SR_on_Pfeiffer_Rat1_Linear1.ipynb): Fits `TCM_SR` by EM to population bursts recorded on a linear track (Pfeiffer & Foster, 2015), and compares it with `RandomWalk` on held-out events. All the dataset-specific loading is in one cell, so another linear-track recording can be plugged in by replacing it.

The simulation notebooks share their settings and helpers through `tcmsr_simulation_config.py` and `tcmsr_simulation.py` in the same folder.

## References

### This fork

> Preprint coming soon.

The real-data notebook uses data from:

> Pfeiffer, B.E., and Foster, D.J. (2015). Autoassociative dynamics in the generation of sequences of hippocampal place cells. Science 349, 180–183.

### Upstream

For further details, please see our [eLife paper](https://doi.org/10.7554/eLife.64505):
> Denovellis, E.L., Gillespie, A.K., Coulter, M.E., Sosa, M., Chung, J.E., Eden, U.T., and Frank, L.M. (2021). Hippocampal replay of experience at real-world speeds. ELife 10, e64505.

or our [conference paper](https://doi.org/10.1109/IEEECONF44664.2019.9048688):
> Denovellis, E.L., Frank, L.M., and Eden, U.T. (2019). Characterizing hippocampal replay using hybrid point process state space models. In 2019 53rd Asilomar Conference on Signals, Systems, and Computers, (Pacific Grove, CA, USA: IEEE), pp. 245–249.

Also see other work using this code:
> Gillespie, A.K., Astudillo Maya, D.A., Denovellis, E.L., Liu, D.F., Kastner, D.B., Coulter, M.E., Roumis, D.K., Eden, U.T., and Frank, L.M. (2021). Hippocampal replay reflects specific past experiences rather than a plan for subsequent choice. Neuron S0896627321005730. <https://doi.org/10.1016/j.neuron.2021.07.029>.

> Joshi, A., Denovellis, E.L., Mankili, A., Meneksedag, Y., Davidson, T., Gillespie, K., Guidera, J.A., Roumis, D., and Frank, L.M. (2022). Dynamic Synchronization between Hippocampal Spatial Representations and the Stepping Rhythm. bioRxiv, 30. <https://doi.org/10.1101/2022.02.23.481357>.

> Gillespie, A.K., Astudillo Maya, D.A., Denovellis, E.L., Desse, S., and Frank, L.M. (2022). Neurofeedback training can modulate task-relevant memory replay in rats. bioRxiv, 2022.10.13.512183. <https://doi.org/10.1101/2022.10.13.512183>.

Upstream: [![DOI](https://zenodo.org/badge/177004334.svg)](https://zenodo.org/badge/latestdoi/177004334) [GitHub](https://github.com/Eden-Kramer-Lab/replay_trajectory_classification)

## Developer Installation

For people who want to expand upon the code for their own use:

#### Prerequisites

Install miniconda (or anaconda) if it isn't already installed:

```bash
# Linux/macOS
wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
bash miniconda.sh -b -p $HOME/miniconda
export PATH="$HOME/miniconda/bin:$PATH"
hash -r
```

#### Environment Setup

```bash
# Update conda
conda update -n base conda

# Create environment from environment.yml
conda env create -f environment.yml

# Activate environment
conda activate replay_trajectory_classification
```

#### Development Installation

```bash
# Modern development installation (recommended)
pip install -e .

# With optional development tools
pip install -e '.[dev]'      # Includes ruff, jupyter, testing tools
pip install -e '.[test]'     # Testing dependencies only
pip install -e '.[docs]'     # Documentation building tools
```

#### Verification

Test your installation:
```bash
python -c "import replay_trajectory_classification; print('✓ Installation successful')"
ruff check replay_trajectory_classification/  # Code quality check
```

#### Building Distribution Packages

```bash
pip install build
python -m build --wheel      # Creates wheel in dist/
python -m build --sdist      # Creates source distribution
```

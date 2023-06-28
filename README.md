# LLSVP_composition_python

[![Build status](https://github.com/KennyVilella/LLSVP_composition_python/workflows/CI/badge.svg)](https://github.com/KennyVilella/LLSVP_composition_python/actions)
[![](https://img.shields.io/badge/docs-main-blue.svg)][docs-main]

This repository implements the model described in:

“Constraints on the composition and temperature of LLSVPs from seismic properties of lower mantle minerals” by K. Vilella, T. Bodin, C.-E. Boukare, F. Deschamps, J. Badro, M. D. Ballmer, and Y. Li

The aim of this module is to provide some constraints on the composition and temperature of the Large Low Shear Velocity Provinces (LLSVPs) observed in the lowermost mantle.
To do so, this simulator calculates the seismic anomalies associated with a wide range of potential LLSVPs compositions under various temperature conditions.

## Installation

First, clone this repository by running the following command in your terminal:
```
   git clone https://github.com/KennyVilella/LLSVP_composition_python
```
Go to the repository directory and install the Python package:
```
   cd LLSVP_composition_python
   pip3 install -e .
```
This will install the necessary dependencies if they are not already installed.

## To-do list

There are several features that are yet to be implemented.
These include, in order of priority:
- Verification: Provide results in the repository to check at first order the validity of the calculation.
- Unit testing and integration testing: Conduct thorough unit and integration tests to ensure the functionality of the simulator.
- Code optimization: Enhance the overall performance and efficiency of the codebase.

## Running the simulator

An example script to run the simulator is provided in the `example` folder.
To execute the script, use the following command:
```
   python3 <path_to_repositpry>/example/sample_script.py
```
This will run a quick simulation and save the results in the `results` folder.

[docs-main]: https://kennyvilella.github.io/LLSVP_composition_python/

# Numerical Weather Prediction (NWP): Fundamentals and Stability

This repository contains numerical experiments and projects focused on the mathematical foundations of Numerical Weather Prediction. The projects investigate discretization methods, error propagation, and the numerical stability of atmospheric solvers.

## Project 1: Numerical Differentiation and Error Analysis

This project explores the sensitivity of numerical derivatives to horizontal grid spacing ($\Delta x$).
- **Objective**: Compare analytical derivatives of trigonometric and exponential functions with numerical approximations at different resolutions.
- **Process**: Implementation of finite difference schemes and systematic evaluation of truncation errors as $\Delta x$ decreases from 100km to 20km.
- **Key Analysis**: Identification of the convergence rates and the "double penalty" effect in high-resolution differentiation.

## Project 2: Stability and Time-Steering

Focuses on the temporal evolution and stability of numerical solutions.
- **Objective**: Visualize and analyze the stability criteria (e.g., Courant-Friedrichs-Lewy / CFL condition) for atmospheric equations.
- **Visualization**: Output includes `.gif` animations showing the evolution of various functions and their associated error propagation over time.
- **Key Analysis**: Investigation of how time-step selection ($\Delta t$) impacts the fidelity of the numerical prediction.

## Requirements

- Python 3.x
- NumPy
- Matplotlib
- SciPy

## Key Scripts
- `numerical_weather_prediction_project_1.py`: Finite difference and error scaling analysis.
- `numerical_weather_prediction_project_2.py`: Stability and time-series evolution.

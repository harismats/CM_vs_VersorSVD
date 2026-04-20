# Code for Paper "Absolute Orientation with Geometric Algebra: A Comparative Study of Characteristic Multivector and Versor-SVD Solvers"

This repository contains the MATLAB code used for the experiments in the paper:

**Absolute Orientation with Geometric Algebra: A Comparative Study of Characteristic Multivector and Versor-SVD Solvers**

The code compares two Geometric Algebra based solvers for the 3D Absolute Orientation problem:

- **CM (Characteristic Multivector) method**
- **Versor–SVD method**

Both methods estimate the rotation aligning two corresponding 3D point sets and are evaluated under a common experimental pipeline.

## Overview

The main script runs three experiments:

### Experiment A — Benchmark on a real point cloud
A real 3D point cloud is transformed by 200 random rigid motions.  
For each trial, the script estimates the rotation with both methods and records:

- runtime
- Frobenius rotation error
- geodesic angle error
- RMS point residual
- determinant deviation

### Experiment B — Noise robustness
Gaussian noise is added to the transformed point cloud at several relative levels.  
The script evaluates how both solvers behave as the noise level increases.

### Experiment C — Geometric conditioning
Synthetic isotropic, planar, and line-like point clouds are generated with controlled anisotropy.  
The script studies the relation between rotation error and the condition number of the cross-covariance matrix.

## Requirements

To run the full code, you will need:

- **MATLAB** (R2016b or newer recommended)
- a **3D point cloud file** named `dolphins.ply`
- a **Clifford algebra toolbox** providing:
  - `clifford_signature`
  - `wedge`
  - `reverse`
  - `unit`
  - `coefficients`
  - basis vectors such as `e1`, `e2`, `e3`

## Notes on dependencies

- The **CM method** depends on the Clifford algebra toolbox.
- The **Versor–SVD implementation** is purely numerical and does **not** require the Clifford toolbox by itself.
- Since the main script compares both methods, the toolbox is required to run the full set of experiments.

## Input data

Put `dolphins.ply` in the same directory as the script before running the code.

The script loads the point cloud with:

```matlab
ptCloud = pcread('dolphins.ply');


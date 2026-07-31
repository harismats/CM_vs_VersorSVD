# Code for “Absolute Orientation with Geometric Algebra: A Comparative Study of Characteristic Multivector and Versor-SVD Solvers”

This repository contains the MATLAB code used for the experiments in the paper

**Absolute Orientation with Geometric Algebra: A Comparative Study of Characteristic Multivector and Versor-SVD Solvers**

The script compares two Geometric Algebra based solvers for the 3D Absolute Orientation problem

- **CM**, the Characteristic Multivector method
- **Versor-SVD**, a numerical solver based on the versor equation and singular value decomposition

Both methods estimate the rotation between corresponding centred 3D point sets.

## Main script

Run the supplied MATLAB `.m` file from the same directory as `dolphins.ply`.

The script creates

```text
output_AO_CM_vs_VSVD_combined
```

and saves all figures, raw `.mat` data, summary tables, trial-level `.csv` files, statistical results and convergence diagnostics in that folder.

## Experiments

### Experiment A

A real dolphin point cloud is transformed by 200 random rigid motions. The script records

- runtime
- Frobenius rotation error
- geodesic angle error
- RMS point residual
- determinant deviation
- orthogonality defect

It also produces paired runtime statistics and a cumulative trial-count analysis.

### Experiment B

Gaussian noise is added at 0, 0.1, 0.5, 1.0, and 1.5 percent of the point cloud bounding-box diagonal. Each level uses 200 trials.

The outputs include

- angle error and RMS-residual plots
- paired bootstrap confidence intervals
- paired sign-flip randomisation tests
- Holm adjusted p-values
- trial count convergence results

### Experiment C

Synthetic isotropic, planar and line-like clouds are generated with anisotropy parameter

```text
ε = 10^-1, 10^-2, 10^-3
```

Each cloud contains 300 points and each configuration uses 100 trials. The script records rotation errors, `cond(H)`, rotational information diagnostics, Versor-SVD singular values, null-space information and convergence results.

## Requirements

- MATLAB R2019b or newer
- Computer Vision Toolbox for `pcread`
- Statistics and Machine Learning Toolbox for `boxplot`
- `dolphins.ply`
- a Clifford algebra toolbox providing
  - `clifford_signature`
  - `wedge`
  - `reverse`
  - `unit`
  - `coefficients`
  - `e1`, `e2`, and `e3`

The CM implementation requires the Clifford toolbox. The Versor-SVD implementation is numerical and does not use the toolbox internally. The complete script requires both components.

## Running the code

1. Add the Clifford toolbox to the MATLAB path.
2. Place the MATLAB script and `dolphins.ply` in the same folder.
3. Set the three experiment switches at the top of the script as required.
4. Run the script.

All computations use double precision and fixed random seeds for reproducibility.

## Timing note

The runtime values reproduce the timing convention used in the paper. CM timing covers the characteristic-multivector solver stage after covariance and reciprocal frame construction. Versor-SVD timing covers the complete `versorSVD_from_pairs_GA` call. The reported values should therefore be interpreted as implementation stage timings rather than equal boundary end-to-end pipeline timings.

## Key outputs

```text
01_Accuracy_Boxplots.png
02_Runtime_Box_and_Scatter.png
B_Noise_Clouds_Overview.png
B_Noise_Metrics_vs_Sigma.png
B_Paired_Angle_Effects.png
B_Trial_Count_Convergence.png
C_Cond_Angle_vs_CondH.png
C_Conditioning_Diagnostics.png
A_runtime_paired_statistics.csv
B_noise_paired_statistics.csv
C_conditioning_summary.csv
metrics_CM_vs_VSVD.mat
B_noise_robustness.mat
C_conditioning_synthetic.mat
```

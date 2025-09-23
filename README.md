# Rates-of-Change-for-Deeptime-data

## Project Overview
This repository contains Python scripts developed for the analysis of **Rates of Change (RoC)** in Cenozoic climate records.  
The methods are designed to:
- Quantify temporal variability in long-term paleoclimate time series (e.g., benthic δ¹⁸O).
- Compare multiple RoC estimation approaches (derivative-based, time-bin methods, robust statistics).
- Evaluate the sensitivity of results to time-window size, original data resolution, and interpolation methods.
- Detect paleoclimate events, stage boundaries, and assess robustness under downsampling.

These codes support the reproducibility of the analyses presented in our manuscript.

The repository includes 16 scripts, organized into five categories:
1. RoC Calculation
- 1.1_smoothline_derivative.py – LOWESS smoothing, bootstrap CI, derivative calculation.
- 1.2_timebin_theil_sen.py – RoC via Theil–Sen slope (absolute value).
- 1.3_timebin_iqr.py – RoC via interquartile range (IQR).
- 1.4_timebin_mean.py – Mean value calculation within each time-bin.
- 1.5_weighted_interpolation.py – Distance- and count-weighted interpolation for missing bins.
- 1.6_timebin_mean_rate.py – Bin-to-bin rate of change based on mean values.
2. RoC Evaluation
- 2.1_rate_evaluation_metrics.py – Evaluation metrics: normalized total variation (nTV) and uniformity index (UVar).
3. Time-window Effects
- 3.1_merge_timebin_results.py – Merge results across multiple time-bins.
- 3.2_log_rate_interval.py – Log-transformed regression across timescales.
- 3.3_multiscale_curves.py – Plot RoC curves at multiple timescales.
- 3.4_heatmap_multiscale.py – Heatmap visualization of RoC across timescales.
4. Data Resolution Effects
- 4.1_downsampling.py – Monte Carlo downsampling, mean curves and CI.
- 4.2_downsampling&rate.py – Downsampling combined with RoC estimation.
5. Statistical Analyses
- 5.1_event_peakrate_detection.py – Detect event peaks within predefined time windows.
- 5.2_segmented_fit_dbscan.py – Segmented regression + DBSCAN clustering to detect breakpoints.
- 5.3_stage.py – Stage division with summary statistics (mean, SD, CI, IQR).

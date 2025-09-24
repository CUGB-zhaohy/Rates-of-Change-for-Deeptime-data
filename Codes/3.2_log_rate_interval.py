import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

# Filter outliers based on standard deviation
def filter_outliers(age_list, value_list, threshold=2):
    ages = np.array(age_list)
    values = np.array(value_list)
    mean_value = np.mean(values)
    std_value = np.std(values)
    mask = np.abs(values - mean_value) <= threshold * std_value
    return ages[mask], values[mask]

# Main function: read Excel file path from user input and generate the plot
def process_and_plot():
    file_path = input("Enter the full path to the Excel file (including file name): ")

    # Try reading the Excel file
    try:
        data = pd.read_excel(file_path, header=None)  # first row has ages, no header
    except FileNotFoundError:
        print("File not found. Please check the path and try again.")
        return
    except Exception as e:
        print(f"Error reading file: {e}")
        return

    # Extract data: first row = ages; remaining rows = values
    ages = data.iloc[0, :]
    values = data.iloc[1:, :].values

    # Flatten into two lists: one for ages, one for corresponding values
    age_list = []
    value_list = []
    for i, age in enumerate(ages):
        for value in values[:, i]:
            if not pd.isna(value):
                age_list.append(age)
                value_list.append(value)

    # Remove outliers
    threshold = 5  # standard deviation multiplier
    filtered_ages, filtered_values = filter_outliers(age_list, value_list, threshold)

    # Convert to numeric and drop NaNs
    filtered_ages = pd.to_numeric(filtered_ages, errors='coerce')
    filtered_values = pd.to_numeric(filtered_values, errors='coerce')
    mask_valid = ~np.isnan(filtered_ages) & ~np.isnan(filtered_values)
    filtered_ages = filtered_ages[mask_valid]
    filtered_values = filtered_values[mask_valid]

    # Ensure numeric arrays
    filtered_ages = np.array(filtered_ages, dtype=float)
    filtered_values = np.array(filtered_values, dtype=float)

    # Prepare arrays for regression (all points)
    age_array = filtered_ages.reshape(-1, 1)
    value_array = filtered_values

    if len(age_array) == 0 or len(value_array) == 0:
        print("Insufficient valid data for linear regression. Please check the input data.")
        return

    # Regression: all points
    reg_all = LinearRegression()
    reg_all.fit(age_array, value_array)
    value_pred_all = reg_all.predict(age_array)
    r2_all = r2_score(value_array, value_pred_all)

    # Compute per-age statistics: mean, Q1, median (Q2), Q3
    unique_ages = np.unique(age_array)
    mean_values, q1_values, q2_values, q3_values = [], [], [], []
    for age_val in unique_ages:
        subset = value_array[age_array.flatten() == age_val]
        mean_values.append(np.mean(subset))
        q1_values.append(np.percentile(subset, 25))
        q2_values.append(np.percentile(subset, 50))
        q3_values.append(np.percentile(subset, 75))

    unique_ages_2d = unique_ages.reshape(-1, 1)

    # Regressions for summary statistics
    reg_mean = LinearRegression().fit(unique_ages_2d, mean_values)
    pred_mean = reg_mean.predict(unique_ages_2d)
    r2_mean = r2_score(mean_values, pred_mean)

    reg_q1 = LinearRegression().fit(unique_ages_2d, q1_values)
    pred_q1 = reg_q1.predict(unique_ages_2d)
    r2_q1 = r2_score(q1_values, pred_q1)

    reg_q2 = LinearRegression().fit(unique_ages_2d, q2_values)
    pred_q2 = reg_q2.predict(unique_ages_2d)
    r2_q2 = r2_score(q2_values, pred_q2)

    reg_q3 = LinearRegression().fit(unique_ages_2d, q3_values)
    pred_q3 = reg_q3.predict(unique_ages_2d)
    r2_q3 = r2_score(q3_values, pred_q3)

    # Plot
    plt.figure(figsize=(10, 6))

    # Scatter: all filtered points
    plt.scatter(filtered_ages, filtered_values, label="Filtered data points", alpha=0.6, color='gray')

    # Regression line: all points
    plt.plot(age_array, value_pred_all, color="green",
             label=f"Regression (all points): y={reg_all.coef_[0]:.2f}x+{reg_all.intercept_: .2f}, R^2={r2_all:.2f}")

    # Mean points and regression
    plt.scatter(unique_ages, mean_values, color="red", label="Mean by age", zorder=5)
    plt.plot(unique_ages, pred_mean, color="red", linestyle='--',
             label=f"Regression (mean): y={reg_mean.coef_[0]:.2f}x+{reg_mean.intercept_: .2f}, R^2={r2_mean:.2f}")

    # Q1 points and regression
    plt.scatter(unique_ages, q1_values, color="blue", label="Q1 (25th percentile)", zorder=5)
    plt.plot(unique_ages, pred_q1, color="blue", linestyle='--',
             label=f"Regression (Q1): y={reg_q1.coef_[0]:.2f}x+{reg_q1.intercept_: .2f}, R^2={r2_q1:.2f}")

    # Q2 (median) points and regression
    plt.scatter(unique_ages, q2_values, color="orange", label="Q2 (median)", zorder=5)
    plt.plot(unique_ages, pred_q2, color="orange", linestyle='--',
             label=f"Regression (Q2): y={reg_q2.coef_[0]:.2f}x+{reg_q2.intercept_: .2f}, R^2={r2_q2:.2f}")

    # Q3 points and regression
    plt.scatter(unique_ages, q3_values, color="purple", label="Q3 (75th percentile)", zorder=5)
    plt.plot(unique_ages, pred_q3, color="purple", linestyle='--',
             label=f"Regression (Q3): y={reg_q3.coef_[0]:.2f}x+{reg_q3.intercept_: .2f}, R^2={r2_q3:.2f}")

    # Labels and style
    plt.xlabel("Age")
    plt.ylabel("Value")
    plt.title("Scatter with regression lines (all points, mean, Q1, median, Q3)")
    plt.legend()
    plt.grid(True)
    plt.show()

# Run
process_and_plot()

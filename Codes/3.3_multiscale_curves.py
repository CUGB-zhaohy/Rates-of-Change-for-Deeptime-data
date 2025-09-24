import os
import re
import pandas as pd
import matplotlib.pyplot as plt

def main():
    # —— Read file ——
    file_path = input("Enter Excel file path: ").strip()
    if not os.path.exists(file_path):
        print("File does not exist. Please check the path.")
        return

    try:
        data = pd.read_excel(file_path)
    except Exception as e:
        print(f"Failed to read: {e}")
        return

    # —— Clean column names and age column ——
    data.columns = data.columns.astype(str).str.strip()   # strip leading/trailing spaces in column names
    age_col = data.columns[0]                              # first column is the age column
    data[age_col] = pd.to_numeric(data[age_col], errors='coerce')  # convert age to numeric
    data = data.dropna(subset=[age_col]).sort_values(by=age_col)   # drop invalid ages and sort by age

    # —— Extract timescale columns and labels (regex for robustness) ——
    experiment_columns = list(data.columns[1:])  # remaining columns are timescale/measurement columns
    keep_cols, new_labels = [], []
    for col in experiment_columns:
        # skip columns that are entirely NaN
        if data[col].notna().sum() == 0:
            continue
        # extract numeric part from the column name (e.g., Timescale_50 / Timescale-050 / Timescale 100 -> 50/100)
        m = re.search(r'(\d+(\.\d+)?)', str(col))
        new_labels.append(m.group(1) if m else str(col))
        keep_cols.append(col)

    if not keep_cols:
        print("No usable (non-empty) timescale columns found.")
        return

    ages = data[age_col]

    # —— Plot ——
    plt.figure(figsize=(30, 6))
    for col, lab in zip(keep_cols, new_labels):
        plt.plot(
            ages, data[col],
            marker='.', linestyle='-', linewidth=1, markersize=2,
            label=f"TimeScale: {lab}"
        )

    plt.title("RoC Trends Across TimeScales", fontsize=16)
    plt.xlabel("Age(kyr)", fontsize=14)
    plt.ylabel("RoC Value", fontsize=14)
    plt.legend(title="TimeScale", loc='upper right', fontsize=10)
    plt.gca().invert_xaxis()  # geologic time: old (left) to young (right)
    plt.grid(True, alpha=0.3)
    plt.gca().set_ylim(0, 20)
    plt.tight_layout(rect=(0.02, 0.02, 0.98, 0.9))
    plt.show()

if __name__ == "__main__":
    main()
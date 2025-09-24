import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as sts  # 不再使用 theilslopes，但保留不影响

# ====================== User-configurable parameters ======================
# Input data file (must include 'Age' and 'Value' columns)
INFILE = r"C:\Users\22457\Desktop\test\O.xlsx"

# Output filename prefix (without extension; the script will append width and .xlsx/.png)
OUT_PREFIX = r"C:\Users\22457\Desktop\test\O_IQR_"

# TimeBin window widths: explicitly listed or generated via range
TIMEBIN_LIST = list(range(50, 1001, 50))

# Age-axis parameters
START_AGE = 67000        # Start age (must be > END_AGE)
END_AGE = 0              # End age
AGE_INTERVAL = 10        # Age node step

# Plot title prefix
TITLE_PREFIX = "O_IQR_"
# ========================================================================

def calc_iqr(y):
    """Return IQR (Q3 - Q1) for a 1-D array-like; if n < 2 return np.nan."""
    arr = np.asarray(y)
    if arr.size < 2:
        return np.nan
    # Compatible with different versions of numpy parameter naming
    try:
        q1, q3 = np.percentile(arr, [25, 75], method="linear")
    except TypeError:
        q1, q3 = np.percentile(arr, [25, 75], interpolation="linear")
    return float(q3 - q1)

def run_one_timebin(tb_width, data, age_nodes):
    """Process a single TimeBin width: generate the result table, save Excel + PNG."""
    half_w = tb_width / 2
    time_bins = [(p - half_w, p + half_w) for p in age_nodes]

    rows = []
    for i, (s, e) in enumerate(time_bins):
        df_bin = data[(data["Age"] >= s) & (data["Age"] < e)]
        uniq_n = df_bin["Age"].nunique()
        iqr = calc_iqr(df_bin["Value"])
        rows.append({
            "TimeBin":   f"TimeBin{i+1}",
            "Age_node":  (s + e) / 2,
            "Counts":    len(df_bin),
            "Age_unique": uniq_n,
            "IQR":       iqr
        })

    df_out = pd.DataFrame(rows)

    # ----------- Save Excel -----------
    excel_path = f"{OUT_PREFIX}{tb_width}.xlsx"
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        df_out.to_excel(writer, index=False, sheet_name="TimeBin_IQR")
    print(f"Saved Excel: {excel_path}")

    # ----------- Plot and save PNG -----------
    df_plot = df_out.sort_values("Age_node", ascending=False)
    plt.figure(figsize=(30, 6))
    plt.plot(df_plot["Age_node"], df_plot["IQR"],
             marker=".", linestyle="-", linewidth=1, markersize=1,
             label="IQR (Q3 - Q1)")
    plt.xlabel("Age_node")
    plt.ylabel("IQR")
    plt.title(f"{TITLE_PREFIX}{tb_width}")
    plt.gca().invert_xaxis()
    plt.legend()
    plt.tight_layout()
    png_path = f"{OUT_PREFIX}{tb_width}.png"
    plt.savefig(png_path, dpi=300)
    plt.close()
    print(f"Saved PNG: {png_path}")

def main():
    if not os.path.isfile(INFILE):
        raise FileNotFoundError(f"Input file not found: {INFILE}")

    # Read and preprocess
    data = pd.read_excel(INFILE, usecols=["Age", "Value"])
    data = data.apply(pd.to_numeric, errors="coerce")
    data = data.dropna(subset=["Age", "Value"])
    # The same Age multi-value is averaged first to avoid the influence of dense repetition points on quantile estimation.
    data = data.groupby("Age", as_index=False, sort=False)["Value"].mean()
    print(f"Data preprocessed: {len(data)} unique Age records")

    # Generate common Age axis
    age_pts = list(range(END_AGE, START_AGE + 1, AGE_INTERVAL))

    # Batch process TimeBins
    print(f"Start batch processing TimeBins: {TIMEBIN_LIST}")
    for tb in TIMEBIN_LIST:
        run_one_timebin(tb, data, age_pts)
    print("All TimeBins processed!")

if __name__ == "__main__":
    main()

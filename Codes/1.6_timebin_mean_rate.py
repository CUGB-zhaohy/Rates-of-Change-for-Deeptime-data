import pandas as pd
import matplotlib.pyplot as plt
import os

# ---------- Read and preprocess ----------
def process_excel(file_path):
    """
    Read a single O_Mean_{TimeBin}.xlsx, keep columns Age_node and Mean,
    drop rows where Mean is NaN, and sort by Age_node in ascending order.
    """
    df = pd.read_excel(file_path)
    # Keep only required columns (in case the file contains others)
    if not {'Age_node', 'Mean'}.issubset(df.columns):
        raise ValueError(f"Missing required columns in {file_path}. Need ['Age_node', 'Mean'].")
    df = df[['Age_node', 'Mean']].copy()
    df = df.dropna(subset=['Mean']).sort_values('Age_node', ascending=True).reset_index(drop=True)
    return df

# ---------- Compute two-point rate separated by time_bin_value ----------
def compute_timebin_rate(df, time_bin_value):
    """
    Find pairs of ages (age, age + time_bin_value) in df,
    compute rate = |Mean(age + Δ) - Mean(age)| / Δ,
    and define Age-point = (age + age + Δ) / 2.
    Return a DataFrame with columns ['Age-point', 'rate'] only.
    """
    # Map Age_node → Mean via index for exact matching by age
    s = df.set_index('Age_node')['Mean']

    base = pd.DataFrame({'Age_node': s.index})
    base['Age_node_next'] = base['Age_node'] + time_bin_value

    # Get corresponding Mean values (NaN if not found)
    base['Mean'] = s.reindex(base['Age_node']).values
    base['Mean_next'] = s.reindex(base['Age_node_next']).values

    # Keep only rows where both points exist
    base = base.dropna(subset=['Mean', 'Mean_next'])

    # Compute Age-point and rate
    base['Age-point'] = (base['Age_node'] + base['Age_node_next']) / 2.0
    base['rate'] = (base['Mean_next'] - base['Mean']).abs() / float(time_bin_value)

    # Output two columns only
    return base[['Age-point', 'rate']].reset_index(drop=True)

# ---------- Plotting ----------
def plot_chart(result_df, time_bin_value, out_png_path):
    """
    Plot the curve in the requested style:
    - figure size (30, 6)
    - line plot (blue, marker '.', linestyle '-', linewidth=1, markersize=1)
    - X axis: Age-point (inverted)
    - Y axis: [0, 1.2 * max]
    - Axis labels and title consistent with previous style
    """
    if result_df.empty:
        print(f"[WARN] No valid pairs for timebin={time_bin_value} kyr. Skipping plot.")
        return

    plt.figure(figsize=(30, 6))
    plt.plot(
        result_df['Age-point'],
        result_df['rate'],
        marker='.',
        linestyle='-',
        color='blue',
        linewidth=1,
        markersize=1,
        label='Rate of change'
    )

    plt.xlabel('Age(kyr)')
    plt.ylabel('Rate of change')
    plt.title(f'Rate of Change —— δ18O [{time_bin_value}kyr]')

    max_rate = result_df['rate'].max()
    plt.ylim(0, max_rate * 1.2 if pd.notna(max_rate) else 1)
    plt.gca().invert_xaxis()

    # Margins and layout
    plt.subplots_adjust(left=0.01)
    plt.tight_layout()

    # Ensure output directory exists
    os.makedirs(os.path.dirname(out_png_path), exist_ok=True)
    plt.savefig(out_png_path, dpi=300)
    plt.close()
    print(f"Figure saved to {out_png_path}")

# ---------- Main ----------
if __name__ == "__main__":
    # Batch list of time bins
    time_bin_values = list(range(50, 1001, 50))

    for tb in time_bin_values:
        in_path  = fr"C:\Users\22457\Desktop\test\Mean\O_Mean_{tb}.xlsx"
        out_xlsx = fr"C:\Users\22457\Desktop\test\O_timebinrate_{tb}.xlsx"
        out_png  = fr"C:\Users\22457\Desktop\test\O_timebinrate_{tb}.png"

        # Read input
        df = process_excel(in_path)

        # Compute rate based on pairs separated by tb
        result = compute_timebin_rate(df, tb)

        # Export results (Age-point and rate only)
        os.makedirs(os.path.dirname(out_xlsx), exist_ok=True)
        result.to_excel(out_xlsx, index=False)
        print(f"Results saved to {out_xlsx} (rows: {len(result)})")

        # Plot
        plot_chart(result, tb, out_png)

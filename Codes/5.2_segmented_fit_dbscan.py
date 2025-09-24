import os
import re
import numpy as np
import pandas as pd
import pwlf
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

AGE_MIN, AGE_MAX = 0, 67100  # Age range

# ----------------- Utilities -----------------
def compute_segment_r2(x, y, slope, intercept, left_bp, right_bp, last_segment=False):
    """Compute local R^2 for a single segment."""
    if last_segment:
        mask = (x >= left_bp) & (x <= right_bp)
    else:
        mask = (x >= left_bp) & (x < right_bp)

    x_seg = x[mask]
    y_seg = y[mask]
    if len(x_seg) == 0:
        return np.nan

    y_pred = slope * x_seg + intercept
    y_mean = np.mean(y_seg)
    ssr = np.sum((y_seg - y_pred) ** 2)
    sst = np.sum((y_seg - y_mean) ** 2)
    if sst == 0:
        return np.nan
    return 1 - ssr / sst

def alpha_10(value):
    """Clip value to [0,1] and discretize into 10 bins for alpha."""
    val = 0.0 if value is None or not np.isfinite(value) else min(max(float(value), 0.0), 1.0)
    return round(val * 10) / 10.0

# ----------------- Main -----------------
def main():
    # ====== Inputs ======
    in_path = input("Enter Excel path: ").strip()
    if not os.path.exists(in_path):
        print("Input file does not exist.")
        return

    try:
        num_segments = int(input("Enter desired number of segments (>=2): ").strip())
    except Exception:
        print("Invalid number of segments.")
        return
    if num_segments < 2:
        print("Number of segments must be >= 2.")
        return

    out_path = input("Enter output Excel path for breakpoints/confidence (e.g., D:\\result.xlsx): ").strip()
    if not out_path.lower().endswith(".xlsx"):
        out_path += ".xlsx"

    # ====== Read and clean input (Age + multiple timescale columns) ======
    try:
        data = pd.read_excel(in_path)
    except Exception as e:
        print(f"Failed to read: {e}")
        return

    data.columns = data.columns.astype(str).str.strip()
    age_col = data.columns[0]
    data[age_col] = pd.to_numeric(data[age_col], errors='coerce')
    data = data.dropna(subset=[age_col]).sort_values(by=age_col)

    # Extract usable timescale columns (skip all-NaN) and parse numeric labels from column names
    experiment_columns = list(data.columns[1:])
    keep_cols, labels = [], []
    for col in experiment_columns:
        if data[col].notna().sum() == 0:
            continue
        m = re.search(r'(\d+(\.\d+)?)', str(col))
        lab = m.group(1) if m else str(col)
        keep_cols.append(col)
        labels.append(lab)

    if not keep_cols:
        print("No usable (non-empty) timescale columns found.")
        return

    # ====== Piecewise fit and build result table (breakpoints/slopes/R2/confidence) ======
    columns = []
    for i in range(1, num_segments + 1):
        columns += [f"Breakpoint{i}", f"Slope{i}", f"R2_{i}", f"Confidence{i}"]
    columns += [f"Breakpoint{num_segments + 1}", f"Confidence{num_segments + 1}"]
    df_results = pd.DataFrame(columns=columns)
    df_results.index.name = "Timescale"

    ages_all = data[age_col].values  # ascending
    for col, lab in zip(keep_cols, labels):
        y_raw = pd.to_numeric(data[col], errors='coerce').values
        mask = (~np.isnan(y_raw)) & (ages_all >= AGE_MIN) & (ages_all <= AGE_MAX)
        x = ages_all[mask]
        y = y_raw[mask]

        if len(x) < (num_segments + 1):
            print(f"Warning: timescale {lab} has insufficient valid points ({len(x)}); skipping.")
            continue

        sum_y = np.nansum(y)
        if not np.isfinite(sum_y) or sum_y <= 0:
            print(f"Warning: timescale {lab} has non-positive or invalid sum of Rate; skipping.")
            continue

        # Normalize + cumulative sum (for reverse cumsum use: np.cumsum(y_norm[::-1])[::-1])
        y_norm = y / sum_y
        cumsum_rate = np.cumsum(y_norm)

        # pwlf piecewise linear fit
        try:
            model = pwlf.PiecewiseLinFit(x, cumsum_rate)
            breakpoints = model.fit(num_segments)  # num_segments+1 breakpoints
            slopes = model.slopes                 # num_segments slopes
            intercepts = model.intercepts         # num_segments intercepts
        except Exception as e:
            print(f"Timescale {lab} fit failed: {e}")
            continue

        # per-segment R^2
        r2_list = []
        for i in range(num_segments):
            r2_local = compute_segment_r2(
                x, cumsum_rate, slopes[i], intercepts[i],
                breakpoints[i], breakpoints[i + 1],
                last_segment=(i == num_segments - 1)
            )
            r2_list.append(r2_local)

        # breakpoint confidence: ends=1; middle = (|Δslope|/max|Δslope|) * mean R^2
        slope_diffs = [abs(slopes[i + 1] - slopes[i]) for i in range(num_segments - 1)]
        max_slope_diff = max(slope_diffs) if slope_diffs else 1e-9
        conf_list = [0.0] * (num_segments + 1)
        conf_list[0] = 1.0
        conf_list[-1] = 1.0
        for k in range(1, num_segments):
            slope_left, slope_right = slopes[k - 1], slopes[k]
            r2_left, r2_right = r2_list[k - 1], r2_list[k]
            slope_factor = abs(slope_right - slope_left) / max_slope_diff if max_slope_diff != 0 else 0.0
            avg_r2 = 0.0 if (np.isnan(r2_left) or np.isnan(r2_right)) else (r2_left + r2_right) / 2.0
            conf_list[k] = max(0.0, slope_factor * avg_r2)

        # write result row
        row_name = f"Timescale_{lab}"
        for i in range(num_segments):
            df_results.loc[row_name, f"Breakpoint{i + 1}"] = breakpoints[i]
            df_results.loc[row_name, f"Slope{i + 1}"] = slopes[i]
            df_results.loc[row_name, f"R2_{i + 1}"] = r2_list[i]
            df_results.loc[row_name, f"Confidence{i + 1}"] = conf_list[i]
        df_results.loc[row_name, f"Breakpoint{num_segments + 1}"] = breakpoints[num_segments]
        df_results.loc[row_name, f"Confidence{num_segments + 1}"] = conf_list[num_segments]

    # R^2 <= 0 check
    r2_cols = [c for c in df_results.columns if c.startswith("R2_")]
    bad = []
    for idx, row in df_results.iterrows():
        for c in r2_cols:
            v = row[c]
            if pd.notnull(v) and v <= 0:
                bad.append((idx, c, v))
    if bad:
        print("Warning: some R^2 <= 0 detected:")
        for (idx, c, v) in bad:
            print(f"  - row '{idx}', col '{c}', R^2 = {float(v):.4f}")
    else:
        print("Check complete: all R^2 > 0.")

    # Write output Excel
    try:
        df_results.to_excel(out_path)
        print(f"[OK] Results written to: {out_path}")
    except Exception as e:
        print(f"Failed to write: {e}")
        return

    # ====== Read the output back and run DBSCAN clustering & plotting ======
    try:
        df = pd.read_excel(out_path, index_col=0)
    except Exception as e:
        print(f"Failed to read output file: {e}")
        return

    # Identify number of segments from columns
    breakpoint_cols = [c for c in df.columns if c.startswith("Breakpoint")]
    confidence_cols = [c for c in df.columns if c.startswith("Confidence")]
    N_plus_1 = len(breakpoint_cols)
    N = N_plus_1 - 1

    # Parse timescale from index (expects 'Timescale_50', etc.)
    def parse_resolution(idx_name):
        return int(idx_name.split("_")[1])

    try:
        resolutions = [parse_resolution(name) for name in df.index]
    except Exception as e:
        print(f"Failed to parse timescale index: {e}")
        return
    num_scales = len(resolutions)

    # vertical sub-offsets to separate rows
    substeps = np.linspace(0, 1, num_scales)
    line_colors = plt.cm.tab20(np.linspace(0, 1, num_scales))

    # ====== Plot (no boxplots) ======
    fig, ax = plt.subplots(figsize=(10, 6))

    all_points = []
    all_confidence = []

    # per-timescale polylines and blue scatter (alpha from confidence)
    for row_idx, (idx_name, row_data) in enumerate(df.iterrows()):
        x_vals = [row_data[f"Breakpoint{i+1}"] for i in range(N_plus_1)]
        conf_vals = [row_data[f"Confidence{i+1}"] for i in range(N_plus_1)]
        y_vals = [i + substeps[row_idx] for i in range(N_plus_1)]

        for i in range(N_plus_1):
            all_points.append((x_vals[i], y_vals[i]))
            all_confidence.append(conf_vals[i])

        ax.plot(x_vals, y_vals, color=line_colors[row_idx], alpha=1.0, label=f"{idx_name}", zorder=1)
        for i in range(N_plus_1):
            ax.scatter(x_vals[i], y_vals[i], color='blue', alpha=alpha_10(conf_vals[i]), zorder=2)

    # timescale legend
    legend_time_scales = ax.legend(loc='upper right', title="Timescale")
    ax.add_artist(legend_time_scales)

    # axes
    ax.invert_xaxis()
    ax.set_ylim(-0.5, N + 1.5)
    ax.set_xlabel("Age (kyr)")
    ax.set_ylabel("Breakpoint index (with per-timescale sub-offset)")
    ax.set_title(f"Breakpoints across timescales (N = {N} segments) + DBSCAN clustering")

    # ====== DBSCAN clustering and reorder by weighted mean X ======
    all_points = np.array(all_points)
    all_confidence = np.array(all_confidence)

    dbscan = DBSCAN(eps=1000, min_samples=5)
    cluster_labels = dbscan.fit_predict(all_points)
    unique_labels = np.unique(cluster_labels)

    # compute weighted mean X and mean confidence per cluster
    cluster_mean_x_old = {}
    cluster_mean_conf_old = {}
    for lbl in unique_labels:
        idxs = np.where(cluster_labels == lbl)[0]
        x_sel = all_points[idxs, 0]
        c_sel = all_confidence[idxs]
        if idxs.size == 0:
            wavg_x, avg_conf = np.nan, np.nan
        else:
            sum_c = c_sel.sum()
            wavg_x = (x_sel * c_sel).sum() / sum_c if sum_c > 0 else np.nan
            avg_conf = c_sel.mean() if c_sel.size > 0 else np.nan
        cluster_mean_x_old[lbl] = wavg_x
        cluster_mean_conf_old[lbl] = avg_conf

    # reorder non-noise clusters (lbl != -1) by weighted mean X
    non_noise_labels = [lbl for lbl in unique_labels if lbl != -1]
    non_noise_labels.sort(key=lambda L: (cluster_mean_x_old[L], L))
    new_label_map = {old_lbl: i for i, old_lbl in enumerate(non_noise_labels)}
    if -1 in unique_labels:
        new_label_map[-1] = -1

    new_cluster_labels = np.array([new_label_map[old_lbl] for old_lbl in cluster_labels])

    # stats under new labels
    cluster_mean_x_new = {}
    cluster_mean_conf_new = {}
    for old_lbl in unique_labels:
        new_lbl = new_label_map[old_lbl]
        cluster_mean_x_new[new_lbl] = cluster_mean_x_old[old_lbl]
        cluster_mean_conf_new[new_lbl] = cluster_mean_conf_old[old_lbl]

    # colors: evenly spaced for non-noise, gray for noise
    cmap = plt.cm.rainbow
    final_labels_sorted = sorted(np.unique(new_cluster_labels))
    if -1 in final_labels_sorted:
        final_labels_sorted.remove(-1)
        final_labels_sorted.append(-1)
    n_non_noise = len([lbl for lbl in final_labels_sorted if lbl != -1])

    color_map_dict = {}
    count_for_color = 0
    for lbl in final_labels_sorted:
        if lbl == -1:
            color_map_dict[lbl] = (0.5, 0.5, 0.5, 1.0)
        else:
            ratio = (count_for_color / (n_non_noise - 1)) if n_non_noise > 1 else 0.0
            color_map_dict[lbl] = cmap(ratio)
            count_for_color += 1

    # overlay colored scatter with black edges
    for i, (x, y) in enumerate(all_points):
        lbl = new_cluster_labels[i]
        plt.scatter(x, y,
                    color=color_map_dict[lbl],
                    alpha=alpha_10(all_confidence[i]),
                    edgecolors='black',
                    linewidths=0.5,
                    zorder=3)

    # cluster legend
    cluster_handles, cluster_labels_for_legend = [], []
    for lbl in final_labels_sorted:
        c = color_map_dict[lbl]
        wavg_x = cluster_mean_x_new.get(lbl, np.nan)
        avg_conf = cluster_mean_conf_new.get(lbl, np.nan)
        wavg_x_str = "Invalid" if np.isnan(wavg_x) else f"{wavg_x:.2f}"
        avg_conf_str = "Invalid" if np.isnan(avg_conf) else f"{avg_conf:.2f}"
        label_str = (f"Noise (-1) | Weighted mean X={wavg_x_str} | Mean confidence={avg_conf_str}"
                     if lbl == -1 else
                     f"Cluster {lbl} | Weighted mean X={wavg_x_str} | Mean confidence={avg_conf_str}")
        sc = plt.scatter([], [], color=c, edgecolors='black', linewidths=0.5)
        cluster_handles.append(sc)
        cluster_labels_for_legend.append(label_str)

    legend_clusters = plt.legend(handles=cluster_handles,
                                 labels=cluster_labels_for_legend,
                                 loc='lower left',
                                 title="DBSCAN clustering (reordered)")
    ax.add_artist(legend_clusters)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()

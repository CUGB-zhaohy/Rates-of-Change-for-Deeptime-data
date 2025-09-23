import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.nonparametric.smoothers_lowess import lowess

# --------------------------
# 1) Read data
# --------------------------
in_path = input("Enter the Excel file path (with columns 'Age' and 'Value'): ").strip()
if not os.path.exists(in_path):
    raise FileNotFoundError(f"File not found: {in_path}")

# LOWESS span & bootstrap
try:
    frac_str = input("LOWESS span frac (0.05-0.5, press Enter for 0.2): ").strip()
    FRAC = float(frac_str) if frac_str else 0.2
except Exception:
    FRAC = 0.2

try:
    b_str = input("Bootstrap replicates (press Enter for 200): ").strip()
    N_BOOT = int(b_str) if b_str else 200
except Exception:
    N_BOOT = 200

# Read & clean
df = pd.read_excel(in_path)
if not {"Age", "Value"}.issubset(df.columns):
    raise ValueError("The Excel must contain columns: 'Age' and 'Value'.")

df = df[["Age", "Value"]].dropna().sort_values("Age", ascending=True).reset_index(drop=True)
x = df["Age"].to_numpy()
y = df["Value"].to_numpy()

# --------------------------
# 2) LOWESS fit + CI on a grid
# --------------------------
def lowess_on_grid(x, y, grid, frac=0.2, it=1):
    """
    Run LOWESS on (x,y) then interpolate the fitted values onto 'grid'.
    """
    fit_xy = lowess(endog=y, exog=x, frac=frac, it=it, return_sorted=True)
    x_fit = fit_xy[:, 0]
    y_fit = fit_xy[:, 1]
    y_grid = np.interp(grid, x_fit, y_fit)
    return y_grid

# Grid
x_min, x_max = float(np.min(x)), float(np.max(x))
GRID_N = 1500
grid = np.linspace(x_min, x_max, GRID_N)

# Main fit
IT = 0  # robust reweighting iterations; set 0 for "tighter" fit if needed
y_fit = lowess_on_grid(x, y, grid, frac=FRAC, it=IT)

# Bootstrap CI
rng = np.random.default_rng(20250731)
boot_preds = np.empty((N_BOOT, GRID_N), dtype=float)
n = len(x)
for b in range(N_BOOT):
    idx = rng.integers(0, n, size=n)  # resample with replacement
    xb = x[idx]; yb = y[idx]
    order = np.argsort(xb)
    xb = xb[order]; yb = yb[order]
    boot_preds[b, :] = lowess_on_grid(xb, yb, grid, frac=FRAC, it=IT)

ci_low = np.percentile(boot_preds, 2.5, axis=0)
ci_high = np.percentile(boot_preds, 97.5, axis=0)

# --------------------------
# 3) Derivative on the grid
# --------------------------
dy_dx = np.abs(np.gradient(y_fit, grid))  # grid is uniform

# ---- Map derivative to left-axis with linear transform y_left = a*dy_dx + b ----
# Scale factor 'a' keeps the derivative band visually moderate;
# Offset 'b' pushes it above the fitted curve to avoid overlap.
y_range = float(np.max(y_fit) - np.min(y_fit))
y_range = y_range if y_range > 0 else 1.0
amp_deriv = np.percentile(np.abs(dy_dx), 95) if np.all(np.isfinite(dy_dx)) else 1.0
amp_deriv = amp_deriv if amp_deriv > 0 else 1.0

SCALE_FRACTION = 0.25  # fraction of y_fit vertical span
OFFSET_FRACTION = 0.10 # additional gap above the top of y_fit

a = SCALE_FRACTION * y_range / amp_deriv
b = np.max(y_fit) + OFFSET_FRACTION * y_range
y_deriv_left = a * dy_dx + b  # this is what we actually plot on the left axis

# Define transform functions for a secondary right axis that displays true derivative units:
def left_to_deriv(y_left):
    return (y_left - b) / a

def deriv_to_left(y_deriv):
    return a * y_deriv + b

# --------------------------
# 4) Plot (left axis + secondary right axis)
# --------------------------
fig, ax = plt.subplots(figsize=(30, 6))

# Colors
scatter_color = 'tab:blue'
fit_color     = 'tab:orange'
ci_alpha      = 0.25
deriv_color   = 'tab:green'

# Scatter (left y-axis)
ax.scatter(x, y, s=4, alpha=0.8, color=scatter_color, label="Data")

# Fit (left y-axis)
ax.plot(grid, y_fit, linestyle='-', linewidth=2, color=fit_color, label=f"LOWESS fit (frac={FRAC})")

# CI (left y-axis)
ax.fill_between(grid, ci_low, ci_high, alpha=ci_alpha, color=fit_color, label="95% CI")

# Derivative drawn on the left axis (shifted upward), but labeled on the right axis:
line_deriv, = ax.plot(grid, y_deriv_left, linestyle='--', linewidth=1.5, color=deriv_color, label="d(Value)/dAge (shifted)")

# Secondary y-axis that shows the true derivative scale
secax = ax.secondary_yaxis('right', functions=(left_to_deriv, deriv_to_left))
secax.set_ylabel("|d(Value)/dAge|")

# Axes labels & title
ax.set_xlabel("Age (kyr)")
ax.set_ylabel("Value")
ax.set_title("Scatter with LOWESS fit and 95% CI; derivative shifted with true scale on right axis")

# Y-limits for left axis (include data/CI and the shifted derivative band)
y_all_min = np.nanmin([np.min(y), np.min(ci_low)])
y_all_max = np.nanmax([np.max(y), np.max(ci_high), np.max(y_deriv_left)])
ax.set_ylim(y_all_min - 0.05 * (y_all_max - y_all_min), y_all_max * 1.02)

# Invert X to match your style
ax.invert_xaxis()

# Merge legend (the derivative line is plotted on ax, so legend picks it up)
ax.legend(loc="best")

# Layout
plt.subplots_adjust(left=0.01)
plt.tight_layout()

# Save
out_dir = os.path.dirname(in_path)
base = os.path.splitext(os.path.basename(in_path))[0]
out_png = os.path.join(out_dir, f"{base}_fit_with_deriv_dual_axis_shifted.png")
plt.savefig(out_png, dpi=300)
plt.close()

print(f"Figure saved to: {out_png}")

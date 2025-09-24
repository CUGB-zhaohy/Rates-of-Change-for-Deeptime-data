import pandas as pd
import math


# ========== 1) Inverse Distance Weighting (two-point) interpolation ========== #
def interpolate(df, a_w_dis: float = 1.0, a_w: float = 1.0, edge: str = "0"):
    """
    Interpolate rows where Counts == 0 (missing) along the Age_node axis.

    Creates two new columns:
        • Value_OnlyDistance     – distance-weighted average, exponent a_w_dis (linear if 1)
        • Value_Distance&Counts  – distance weight × neighbor Counts, exponent a_w

    Parameters
    ----------
    df : DataFrame
        Must contain ['Age_node', 'Mean_origin', 'Counts'].
    a_w_dis : float
        Distance exponent for Value_OnlyDistance.
    a_w : float
        Distance exponent for Value_Distance&Counts.
    edge : {"0", "nearest"}, default "0"
        How to handle positions with only one neighbor or none:
            "0"        → leave as 0 (still considered missing)
            "nearest"  → copy the nearest neighbor's Mean_origin
    """

    def _interp_one_pass(df_sorted: pd.DataFrame, a: float, use_counts: bool):
        out = []
        # Known points have Counts != 0
        known = df_sorted[df_sorted["Counts"] != 0]
        use_counts = use_counts and ("Counts" in df_sorted.columns)

        for _, row in df_sorted.iterrows():
            # Keep original value if this row is known
            if row["Counts"] != 0:
                out.append(row["Mean_origin"])
                continue

            age = row["Age_node"]
            left = known[known["Age_node"] < age]
            right = known[known["Age_node"] > age]

            # Handle cases with zero or one neighbor
            if left.empty and right.empty:
                out.append(0)
                continue
            if left.empty or right.empty:
                if edge == "nearest":
                    p = right.iloc[0] if left.empty else left.iloc[-1]
                    out.append(p["Mean_origin"])
                else:
                    out.append(0)
                continue

            # Two nearest neighbors
            pL = left.iloc[-1]
            pR = right.iloc[0]

            d1 = abs(age - pL["Age_node"])
            d2 = abs(age - pR["Age_node"])

            # Same Age_node (rare) – copy value directly
            if d1 == 0:
                out.append(pL["Mean_origin"])
                continue
            if d2 == 0:
                out.append(pR["Mean_origin"])
                continue

            w1 = 1.0 / (d1 ** a)
            w2 = 1.0 / (d2 ** a)
            if use_counts:
                w1 *= pL["Counts"]
                w2 *= pR["Counts"]

            out.append((w1 * pL["Mean_origin"] + w2 * pR["Mean_origin"]) / (w1 + w2))
        return out

    # Sort by Age_node for reliable neighbor detection
    df_sorted = df.sort_values("Age_node").reset_index()
    df_sorted["Value_OnlyDistance"] = _interp_one_pass(df_sorted, a_w_dis, use_counts=False)
    df_sorted["Value_Distance&Counts"] = _interp_one_pass(df_sorted, a_w, use_counts=True)

    # Merge back to original order
    df_out = df.copy()
    df_out = df_out.merge(
        df_sorted[["index", "Value_OnlyDistance", "Value_Distance&Counts"]],
        left_index=True,
        right_on="index",
        how="left",
    ).drop(columns=["index"])
    return df_out


# ========== 2) Batch processing ========== #
if __name__ == "__main__":
    print(
        "Distance exponent a controls decay speed:\n"
        "  larger a  → higher weight on closer points;\n"
        "  smaller a → smoother weighting.\n"
        "Linear interpolation corresponds to a = 1 for distance-only weighting.\n"
    )

    a_str = input(
        "Enter the distance exponent a for the Distance&Counts case (press Enter for 1): "
    ).strip()
    a_counts = float(a_str) if a_str else 1.0
    a_only_distance = 1.0  # linear weighting for Value_OnlyDistance

    # Example list; change to range(10, 100, 10) for 10–90, or range(50, 501, 50) for 50–500 kyr
    time_bin_values = list(range(10, 19, 10))

    for TimeBin_Value in time_bin_values:
        in_path = fr"C:\Users\22457\Desktop\test\O_Mean_{TimeBin_Value}.xlsx"
        out_path = fr"C:\Users\22457\Desktop\test\O_Mean_{TimeBin_Value}_interp.xlsx"

        df = pd.read_excel(in_path)

        # Create Mean_origin if not present
        if "Mean_origin" not in df.columns:
            df["Mean_origin"] = df["Mean"]
            df = df.drop(columns=["Mean"])

        # Interpolate rows where Counts == 0
        df = interpolate(df, a_w_dis=a_only_distance, a_w=a_counts, edge="0")

        df.to_excel(out_path, index=False)
        print(f"Interpolated data saved to {out_path}")

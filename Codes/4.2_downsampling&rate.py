import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ====================== User-configurable parameters ======================
INFILE = r"C:\Users\22457\Desktop\test\O.xlsx"
OUT_PREFIX = r"C:\Users\22457\Desktop\test\O_DownSample_"

# 降密参数
DOWNSAMPLE_WIDTH = 30   # 降密区间宽度 (kyr)
N_ITER = 20             # 迭代次数

# 变率计算参数
TIMEBIN_WIDTH = 200      # 变率计算的时间窗
START_AGE = 67000
END_AGE = 0
AGE_INTERVAL = 10
# ========================================================================

# ========== Step 1: 降密抽样 ========== #
def downsample_once(data, width):
    """对原始数据进行一次降密（每个 width 窗口随机取一个点）"""
    bins = np.arange(END_AGE, START_AGE + width, width)
    sampled = []
    for i in range(len(bins) - 1):
        df_bin = data[(data["Age"] >= bins[i]) & (data["Age"] < bins[i + 1])]
        if len(df_bin) > 0:
            sampled.append(df_bin.sample(n=1))
    if sampled:
        return pd.concat(sampled, ignore_index=True)
    else:
        return pd.DataFrame(columns=["Age", "Value"])

# ========== Step 2: TimeBin 求均值 ========== #
def run_timebin(tb_width, data, age_nodes):
    half_w = tb_width / 2
    time_bins = [(p - half_w, p + half_w) for p in age_nodes]

    rows = []
    for (s, e) in time_bins:
        df_bin = data[(data["Age"] >= s) & (data["Age"] < e)]
        rows.append({
            "Age_node": (s + e) / 2,
            "Mean_origin": df_bin["Value"].mean() if len(df_bin) > 0 else np.nan,
            "Counts": len(df_bin)
        })
    return pd.DataFrame(rows)

# ========== Step 3: 插值 ========== #
def interpolate(df, a_w_dis=1.0, a_w=1.0, edge="0"):
    def _interp_one_pass(df_sorted, a, use_counts):
        out = []
        known = df_sorted[df_sorted["Counts"] != 0]
        for _, row in df_sorted.iterrows():
            if row["Counts"] != 0:  # 已知点
                out.append(row["Mean_origin"])
                continue
            age = row["Age_node"]
            left = known[known["Age_node"] < age]
            right = known[known["Age_node"] > age]

            if left.empty and right.empty:
                out.append(0); continue
            if left.empty or right.empty:
                if edge == "nearest":
                    p = right.iloc[0] if left.empty else left.iloc[-1]
                    out.append(p["Mean_origin"])
                else:
                    out.append(0)
                continue

            pL, pR = left.iloc[-1], right.iloc[0]
            d1, d2 = abs(age - pL["Age_node"]), abs(age - pR["Age_node"])
            if d1 == 0: out.append(pL["Mean_origin"]); continue
            if d2 == 0: out.append(pR["Mean_origin"]); continue
            w1, w2 = 1.0 / (d1 ** a), 1.0 / (d2 ** a)
            if use_counts:
                w1 *= pL["Counts"]; w2 *= pR["Counts"]
            out.append((w1 * pL["Mean_origin"] + w2 * pR["Mean_origin"]) / (w1 + w2))
        return out

    df_sorted = df.sort_values("Age_node").reset_index()
    df_sorted["Mean"] = _interp_one_pass(df_sorted, a_w, use_counts=True)
    return df_sorted[["Age_node", "Mean"]]

# ========== Step 4: Rate 计算 ========== #
def compute_rate(df, time_bin_value):
    s = df.set_index("Age_node")["Mean"]
    base = pd.DataFrame({"Age_node": s.index})
    base["Age_node_next"] = base["Age_node"] + time_bin_value
    base["Mean"] = s.reindex(base["Age_node"]).values
    base["Mean_next"] = s.reindex(base["Age_node_next"]).values
    base = base.dropna(subset=["Mean", "Mean_next"])
    base["Age-point"] = (base["Age_node"] + base["Age_node_next"]) / 2.0
    base["rate"] = (base["Mean_next"] - base["Mean"]).abs() / float(time_bin_value)
    return base[["Age-point", "rate"]].reset_index(drop=True)

# ========== Step 5: 主流程 ========== #
def main():
    if not os.path.isfile(INFILE):
        raise FileNotFoundError(f"Input file not found: {INFILE}")
    data = pd.read_excel(INFILE, usecols=["Age", "Value"])
    data = data.dropna(subset=["Age", "Value"])
    data = data.groupby("Age", as_index=False)["Value"].mean()
    print(f"原始数据：{len(data)} 个点")

    age_pts = list(range(END_AGE, START_AGE + 1, AGE_INTERVAL))
    all_iters = []

    for it in range(N_ITER):
        down_data = downsample_once(data, DOWNSAMPLE_WIDTH)
        df_tb = run_timebin(TIMEBIN_WIDTH, down_data, age_pts)
        df_interp = interpolate(df_tb, a_w=1.0)
        df_rate = compute_rate(df_interp, TIMEBIN_WIDTH)
        all_iters.append(df_rate.set_index("Age-point")["rate"])

    all_df = pd.concat(all_iters, axis=1)
    mean_curve = all_df.mean(axis=1)
    lower = all_df.quantile(0.025, axis=1)
    upper = all_df.quantile(0.975, axis=1)

    # 绘制置信区间图
    plt.figure(figsize=(30, 6))
    plt.plot(mean_curve.index, mean_curve.values, color="blue", label="Mean rate")
    plt.fill_between(mean_curve.index, lower, upper, color="blue", alpha=0.3, label="95% CI")
    plt.xlabel("Age(kyr)")
    plt.ylabel("Rate of change")
    plt.title(f"Rate of Change —— δ18O [{TIMEBIN_WIDTH} kyr], DownSample {DOWNSAMPLE_WIDTH} kyr")
    plt.gca().invert_xaxis()
    plt.legend()
    plt.tight_layout()
    out_png = f"{OUT_PREFIX}{DOWNSAMPLE_WIDTH}_{TIMEBIN_WIDTH}_Rate_Meanrate.png"
    plt.savefig(out_png, dpi=300)
    plt.close()
    print(f"结果图保存: {out_png}")

if __name__ == "__main__":
    main()

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ====================== User-configurable parameters ======================
INFILE = r"C:\Users\22457\Desktop\test\O.xlsx"   # 输入文件
OUT_PREFIX = r"C:\Users\22457\Desktop\test\O_DownSample_"  # 输出文件前缀

# 降密参数
DOWNSAMPLE_WIDTH = 50    # 降密的分辨率（kyr）
N_ITER = 200              # 重复次数，用于蒙特卡洛置信区间

# 变率计算参数
TIMEBIN_LIST = [100]      # 只计算 100 kyr
START_AGE = 67000
END_AGE = 0
AGE_INTERVAL = 10

TITLE_PREFIX = "O_Rate_"
# ========================================================================

def calc_mean(y):
    arr = np.asarray(y)
    if arr.size == 0:
        return np.nan
    return float(np.mean(arr))

def run_one_timebin(tb_width, data, age_nodes):
    """在降密数据上跑一次变率计算"""
    half_w = tb_width / 2
    time_bins = [(p - half_w, p + half_w) for p in age_nodes]

    rows = []
    for i, (s, e) in enumerate(time_bins):
        df_bin = data[(data["Age"] >= s) & (data["Age"] < e)]
        mean_v = calc_mean(df_bin["Value"])
        rows.append({
            "Age_node": (s + e) / 2,
            "Mean": mean_v
        })

    return pd.DataFrame(rows)

def downsample_once(data, width):
    """对原始数据进行一次降密（每个宽度区间随机取一点）"""
    bins = np.arange(END_AGE, START_AGE + width, width)
    sampled = []
    for i in range(len(bins) - 1):
        df_bin = data[(data["Age"] >= bins[i]) & (data["Age"] < bins[i+1])]
        if len(df_bin) > 0:
            sampled.append(df_bin.sample(n=1))
    if sampled:
        return pd.concat(sampled, ignore_index=True)
    else:
        return pd.DataFrame(columns=["Age", "Value"])

def main():
    if not os.path.isfile(INFILE):
        raise FileNotFoundError(f"Input file not found: {INFILE}")

    # 读取并预处理原始数据
    data = pd.read_excel(INFILE, usecols=["Age", "Value"])
    data = data.apply(pd.to_numeric, errors="coerce")
    data = data.dropna(subset=["Age", "Value"])
    data = data.groupby("Age", as_index=False, sort=False)["Value"].mean()
    print(f"原始数据：{len(data)} 个 Age 点")

    # 公共 Age 节点
    age_pts = list(range(END_AGE, START_AGE + 1, AGE_INTERVAL))

    # 存储多次迭代结果
    all_iters = []

    for it in range(N_ITER):
        down_data = downsample_once(data, DOWNSAMPLE_WIDTH)
        df_rate = run_one_timebin(100, down_data, age_pts)
        all_iters.append(df_rate["Mean"].values)

    all_iters = np.array(all_iters)   # shape = (N_ITER, n_age_nodes)

    # 计算统计量
    mean_curve = np.nanmean(all_iters, axis=0)
    lower = np.nanpercentile(all_iters, 2.5, axis=0)
    upper = np.nanpercentile(all_iters, 97.5, axis=0)

    # 绘图
    plt.figure(figsize=(30, 6))
    plt.plot(age_pts, mean_curve, color="blue", label="Mean (Monte Carlo)")
    plt.fill_between(age_pts, lower, upper, color="blue", alpha=0.3, label="95% CI")
    plt.xlabel("Age (kyr)")
    plt.ylabel("Mean Value (100 kyr)")
    plt.title(f"{TITLE_PREFIX}DownSample{DOWNSAMPLE_WIDTH}kyr")
    plt.gca().invert_xaxis()
    plt.legend()
    plt.tight_layout()
    png_path = f"{OUT_PREFIX}{DOWNSAMPLE_WIDTH}_MC.png"
    plt.savefig(png_path, dpi=300)
    plt.close()
    print(f"结果图已保存: {png_path}")

if __name__ == "__main__":
    main()
import os
import glob
import numpy as np
import pandas as pd

# ====================== 用户可配置参数 ======================
INPUT_DIR   = r"C:\Users\22457\Desktop\test\Mean-Rate"   # 结果表文件夹
FILE_GLOB   = "*.xlsx"                             # 匹配文件
AGE_COL     = "Age-point"                           # ← 指定年龄列名（例如 "Age_node"、"Age"）
RATE_COL    = "rate"                                # ← 指定作为“变率”的列名（例如 "IQR"）
AGE_WINDOW  = 1000.0                               # 滚动方差窗口（与 AGE_COL 同单位）
SAVE_CSV    = True
OUT_CSV     = os.path.join(INPUT_DIR, "nTV_UVar_summary.csv")
AGE_TOL_REL = 1e-3                                 # 等间隔相对容忍阈值
# ==========================================================

def zscore(x):
    x = np.asarray(x, dtype=float)
    mu = np.nanmean(x); sd = np.nanstd(x, ddof=0)
    if not np.isfinite(sd) or sd == 0: return np.zeros_like(x)
    return (x - mu) / sd

def estimate_dt(age_arr):
    diffs = np.diff(age_arr); diffs = np.abs(diffs); diffs = diffs[np.isfinite(diffs)]
    if diffs.size == 0: return np.nan, np.nan
    dt = float(np.median(diffs))
    if dt == 0: return 0.0, np.nan
    cv = float(np.std(diffs, ddof=0) / dt)
    return dt, cv

def nTV_time(z, dt):
    z = np.asarray(z, dtype=float)
    if z.size < 2 or not np.isfinite(dt) or dt <= 0: return np.nan
    dz = np.diff(z)
    return float(np.nansum(np.abs(dz)) / ((z.size - 1) * dt))

def rolling_var(z, L):
    if L < 2: return np.array([], dtype=float)
    s = pd.Series(z)
    v = s.rolling(window=L, min_periods=L, center=False).var(ddof=0).to_numpy()
    return v[np.isfinite(v)]

def gini_nonneg(x):
    x = np.asarray(x, dtype=float)
    if x.size == 0: return np.nan
    x = x[x >= 0]
    if x.size == 0: return 0.0
    s = x.sum()
    if s == 0: return 0.0
    x_sorted = np.sort(x); n = x_sorted.size
    idx = np.arange(1, n + 1, dtype=float)
    g = (2.0 * np.sum(idx * x_sorted) / (n * s)) - (n + 1.0) / n
    return float(g)

def process_one_file(path, age_col=AGE_COL, rate_col=RATE_COL, age_window=AGE_WINDOW, age_tol_rel=AGE_TOL_REL):
    df = pd.read_excel(path, engine="openpyxl")

    # 列校验
    if age_col not in df.columns:
        raise ValueError(f"{os.path.basename(path)} 缺少年龄列 {age_col}；现有列: {list(df.columns)}")
    if rate_col not in df.columns:
        raise ValueError(f"{os.path.basename(path)} 缺少指定的变率列 {rate_col}；现有列: {list(df.columns)}")

    # 清理 & 排序（按年龄升序）
    df = df[[age_col, rate_col]].dropna()
    df = df.sort_values(age_col, ascending=True).reset_index(drop=True)

    ages = df[age_col].to_numpy(dtype=float)
    vals = df[rate_col].to_numpy(dtype=float)

    # Z-score 标准化
    z = zscore(vals)

    # 估计 Δt 并检查等间隔性
    dt, cv = estimate_dt(ages)
    if not np.isfinite(dt) or dt <= 0:
        print(f"[WARN] {os.path.basename(path)}: 无法估计有效 Δt（点数过少或 {age_col} 异常）")
        return os.path.basename(path), np.nan, np.nan
    if np.isfinite(cv) and cv > age_tol_rel:
        print(f"[INFO]  {os.path.basename(path)}: 相邻 {age_col} 差 CV={cv:.3g}，可能并非严格等间隔（将继续计算）")

    # nTV：对时间归一
    ntv = nTV_time(z, dt)

    # UVar：滚动方差 -> Gini -> UVar
    L = int(round(age_window / dt))
    L = max(2, min(L, z.size))
    v = rolling_var(z, L)
    if v.size == 0:
        uvar = np.nan
    else:
        g = gini_nonneg(v)
        uvar = np.nan if not np.isfinite(g) else 1.0 - max(0.0, min(1.0, g))

    return os.path.basename(path), ntv, uvar

def main():
    files = sorted(glob.glob(os.path.join(INPUT_DIR, FILE_GLOB)))
    if not files:
        raise FileNotFoundError(f"未在 {INPUT_DIR} 下找到匹配 {FILE_GLOB} 的文件")

    rows = []
    for fp in files:
        try:
            name, ntv, uvar = process_one_file(fp)
        except Exception as e:
            print(f"[ERROR] 处理 {os.path.basename(fp)} 出错：{e}")
            name, ntv, uvar = os.path.basename(fp), np.nan, np.nan
        rows.append({"File": name, "nTV": ntv, "UVar": uvar})

    out = pd.DataFrame(rows)
    pd.set_option("display.float_format", lambda x: f"{x:.6f}")
    print("\n=== nTV 与 UVar 汇总 ===")
    print(out.to_string(index=False))

    if SAVE_CSV:
        out.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
        print(f"\n已保存：{OUT_CSV}")

if __name__ == "__main__":
    main()

import warnings
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

def main():
    # ========== 1. 读取 ==========
    file_path = input("请输入数据文件的路径：").strip()
    df = pd.read_excel(file_path)

    age_col = df.columns[0]
    rate_cols = list(df.columns[1:])

    # >10 截断
    df[rate_cols] = df[rate_cols].clip(lower=0, upper=10)

    # ========== 2. 排序 ==========
    # 先按年龄降序，保证插值沿着单调的 Age 方向进行
    df_sorted = df.sort_values(by=age_col, ascending=False)

    # 仅对“内部空洞”做线性插值（不外推端点）
    def interpolate_inside(s: pd.Series) -> pd.Series:
        # 若有效点不足 2 个，无法线性插值，直接返回
        if s.notna().sum() < 2:
            return s
        # 优先使用 pandas 的 limit_area='inside'
        try:
            return s.interpolate(method='linear', limit_area='inside')
        except TypeError:
            # 兼容旧版 pandas：先两端方向插值，再把头尾原本的 NaN 段复原回 NaN
            out = s.interpolate(method='linear', limit_direction='both')

            # 计算前导与尾部连续 NaN 的长度
            vals = s.to_numpy()
            lead = 0
            for v in vals:
                if pd.isna(v):
                    lead += 1
                else:
                    break
            trail = 0
            for v in vals[::-1]:
                if pd.isna(v):
                    trail += 1
                else:
                    break
            if lead > 0:
                out.iloc[:lead] = pd.NA
            if trail > 0:
                out.iloc[-trail:] = pd.NA
            return out

    # 对每个 rate 列做插值（仅内部 NaN）
    df_sorted[rate_cols] = df_sorted[rate_cols].apply(interpolate_inside, axis=0)

    # Timescale 列名排序（按后缀数字）
    def suffix_numeric(name):
        try:
            return float(str(name).split('_')[-1])
        except ValueError:
            return name
    rate_cols_sorted = sorted(rate_cols, key=suffix_numeric)

    heat_data = df_sorted[rate_cols_sorted].T

    # ========== 3. 画图 ==========

    fig, ax = plt.subplots(figsize=(30, 6))

    # -- 颜色映射：深红(低)→黄(高) --
    cmap = mpl.colormaps['turbo']

    im = ax.imshow(
        heat_data,
        aspect='auto',
        cmap=cmap,
        vmin=0, vmax=10,
        interpolation='nearest'
    )

    # === X 轴：只显示 0、10000、20000 … ===
    import numpy as np
    ages = df_sorted[age_col].to_numpy()  # 已降序排列
    max_age = int(ages.max())

    tick_vals = list(range(0, max_age + 1, 10000))
    tick_pos = [int(np.argmin(np.abs(ages - tv))) for tv in tick_vals]

    # 去重并按位置排序
    tick_pos_unique, tick_vals_unique = zip(
        *sorted({p: v for p, v in zip(tick_pos, tick_vals)}.items())
    )

    ax.set_xticks(tick_pos_unique)
    ax.set_xticklabels(tick_vals_unique, rotation=0)
    ax.set_xlabel('Age(kyr)')

    # Y 轴
    ax.set_yticks(range(len(rate_cols_sorted)))
    ax.set_yticklabels(rate_cols_sorted)
    ax.set_ylabel('RoC of different TimeScales')

    # 颜色条
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('RoC Value')

    plt.tight_layout(rect=(0.06, 0.02, 0.99, 0.98))
    plt.show()

    # （可选）屏蔽特定类型警告
    # warnings.filterwarnings("ignore", category=UserWarning, message="Glyph")
    # warnings.filterwarnings("ignore", category=mpl.MatplotlibDeprecationWarning)

if __name__ == "__main__":
    main()

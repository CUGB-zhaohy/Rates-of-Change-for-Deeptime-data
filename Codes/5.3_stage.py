import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm  # 用于获取渐变色


def read_and_plot_hist(file_path):
    # 读取 Excel 文件
    data = pd.read_excel(file_path)

    # 给定 bins，用于分组（9 段）
    bins = [0, 3687, 13136, 24252, 32777, 42163, 50900, 56767, 67000]

    # 划分分段，labels=False 直接得到分段索引(0~8)
    data['bin_index'] = pd.cut(
        data['Age-point'],
        bins=bins,
        labels=False,
        include_lowest=True
    )

    # 获取所有数值型列（排除 AgePoint 本身）
    numeric_cols = data.select_dtypes(include='number').columns.drop(['Age-point'])

    # 分段统计结果
    group_means = []
    stats_list = []

    for i in range(len(bins) - 1):
        # 取出当前分段的数据
        subset = data[data['bin_index'] == i]

        # 将所有数值列扁平化并去除 NaN
        vals = subset[numeric_cols].values.flatten()
        vals = vals[~np.isnan(vals)]

        if len(vals) == 0:
            mean_val = 0
            sd_val = 0
            se_val = 0
            ci_lower = 0
            ci_upper = 0
            val_min = 0
            val_max = 0
            iqr_val = 0
        else:
            mean_val = np.mean(vals)
            # 样本标准差 (ddof=1)
            sd_val = np.std(vals, ddof=1)
            n = len(vals)  # 样本容量
            # 标准误
            se_val = sd_val / np.sqrt(n) if n > 1 else 0

            # 95% CI （假设正态分布，使用1.96倍SE）
            z = 1.96
            ci_lower = mean_val - z * se_val
            ci_upper = mean_val + z * se_val

            # 极差
            val_min = np.min(vals)
            val_max = np.max(vals)

            # IQR
            q1, q3 = np.percentile(vals, [25, 75])
            iqr_val = q3 - q1

        group_means.append(mean_val)
        stats_info = {
            'sd': sd_val,
            'se': se_val,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'min': val_min,
            'max': val_max,
            'iqr': iqr_val,
        }
        stats_list.append(stats_info)

    # ---------------------------------------------------------
    # 根据 group_means 的高低来分配渐变色，并设定透明度下限 0.5
    # ---------------------------------------------------------
    min_val = min(group_means)
    # 避免分母为 0 或所有值都相同导致异常
    max_val = max(group_means) if max(group_means) != 0 else 1

    cmap = cm.Blues  # 这里使用 Blues，可换成 Reds, Greens, viridis 等
    colors_manual = []

    for mean_val in group_means:
        if max_val == min_val:
            # 如果全部平均值相同，则统一给一个中间值
            norm_val = 0.5
        else:
            # 归一化到 [0, 1]
            norm_val = (mean_val - min_val) / (max_val - min_val)

        # 从 cmap 获取对应的 (R,G,B,A) 四元组
        rgba = list(cmap(norm_val))
        # 调整 alpha：最浅不低于 0.5，最大不超过 1.0
        rgba[3] = 0.5 + 0.5 * norm_val
        colors_manual.append(tuple(rgba))

    # 用户可输入图表标题
    chart_title = input("请输入图表的标题：")

    # 绘制直方图风格的柱状图
    plt.figure(figsize=(10, 6))
    bar_container = plt.bar(
        x=bins[:-1],          # 左边界
        height=group_means,   # 柱的高度（均值）
        width=np.diff(bins),  # 柱的宽度
        align='edge',         # 柱左边缘与 bins[i] 对齐
        color=colors_manual   # 设置每个柱的颜色
    )

    # 翻转 x 轴，让左侧是 67100，右侧是 0
    plt.xlim(bins[-1], bins[0])

    plt.title(chart_title)
    plt.xlabel('Age')
    plt.ylabel('Mean Value')
    plt.grid(True)

    # 在每个柱子上方标注统计信息
    for i, rect in enumerate(bar_container):
        left = rect.get_x()
        width = rect.get_width()
        height = rect.get_height()

        x_center = left + width / 2.0
        y_top = height

        sd_val = stats_list[i]['sd']
        se_val = stats_list[i]['se']
        ci_l = stats_list[i]['ci_lower']
        ci_u = stats_list[i]['ci_upper']
        val_min = stats_list[i]['min']
        val_max = stats_list[i]['max']
        iqr_val = stats_list[i]['iqr']

        text_str = (
            f"SD={sd_val:.2f}\n"
            f"SE={se_val:.2f}\n"
            f"95%CI=[{ci_l:.2f},{ci_u:.2f}]\n"
            f"Range=[{val_min:.2f},{val_max:.2f}]\n"
            f"IQR={iqr_val:.2f}"
        )

        # 这里 0.05 * max(group_means) 只是一个示例偏移量，可自行调节
        plt.text(
            x_center,
            y_top + 0.05 * max(group_means),
            text_str,
            ha='center',
            va='bottom',
            fontsize=9
        )

    plt.show()


if __name__ == "__main__":
    file_path = input("请输入文件地址: ")
    read_and_plot_hist(file_path)

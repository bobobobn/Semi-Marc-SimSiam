import matplotlib.pyplot as plt
import numpy as np

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 设置默认字体为微软雅黑
plt.rcParams['axes.unicode_minus'] = False  # 防止负号显示问题

# 模型名称
models = ['SimSiam', 'SimCLR', 'BYOL', 'CNN']

# Semi-Marc 前后的准确率
before_semi_marc = [0.9198, 0.9131, 0.9120, 0.7420]  # 添加 CNN 的准确率
after_semi_marc = [0.9310, 0.9264, 0.9297, 0.7783]  # 更新 CNN 的准确率为 0.7783

# 设置柱状图的位置
x = np.arange(len(models))  # 模型的数量
width = 0.35  # 柱子的宽度

# 创建图表
fig, ax = plt.subplots(figsize=(10, 6))

# 绘制 Semi-Marc 前后的柱状图
bars1 = ax.bar(x - width/2, before_semi_marc, width, label='Semi-Marc 前', color='b')
bars2 = ax.bar(x + width/2, after_semi_marc, width, label='Semi-Marc 后', color='g')

# 添加标签
ax.set_xlabel('模型', fontsize=12)
ax.set_ylabel('准确率', fontsize=12)
ax.set_title('Semi-Marc 前后模型准确率比较', fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(models)

# 将图例放置到右侧，避免遮挡
ax.legend(loc='lower right', bbox_to_anchor=(1, 1), fontsize=12)

# 添加数据标签
def add_labels(bars):
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, yval + 0.001, round(yval, 4), ha='center', va='bottom', fontsize=10)

# 显示数据标签
add_labels(bars1)
add_labels(bars2)

# 调整图表布局，避免图例被遮挡
plt.tight_layout()

# 展示图表
plt.show()

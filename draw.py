# -*- coding = utf-8 -*-
# @Time : 2024/5/9 10:58
# @Author : bobobobn
# @File : draw.py
# @Software: PyCharm
# from matplotlib import pyplot as plt
# plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置字体为SimHei
# plt.rcParams['axes.unicode_minus'] = False  # 设置正确显示负号
# y = [59.93,63.73,54.62,62.26,61.71]
# x = ["DCNN", "DCNN+先验知识SSV", "DCNN+线性AE-SSV", "DCNN+DCAE-SSV", "DCNN+先验知识-DCAE-SSV"]
# plt.bar(x, y)
# plt.ylabel("ACC")
# # 显示数值标签在柱子上方
# for i in range(len(y)):
#     plt.text(i, y[i], str(y[i]), ha='center', va='bottom')
# plt.show()


from collections import deque


def min_cost_to_transform(a, y, c1, c2, c3):
    queue = deque([(a, 0)])  # 队列初始化为 (当前值, 累计花费)
    visited = set()  # 存储已访问过的状态

    while queue:
        current_value, current_cost = queue.popleft()

        # 如果当前值已经达到目标值，返回累计的花费
        if current_value == y:
            return current_cost

        # 如果当前状态已经访问过，跳过
        if current_value in visited:
            continue

        # 标记当前状态为已访问
        visited.add(current_value)

        # 三种操作，加入新的状态和对应的花费到队列中
        queue.append((current_value + 2, current_cost + c1))
        queue.append((current_value - 3, current_cost + c2))
        queue.append((current_value + 5, current_cost + c3))

    return -1  # 如果无法达到目标值，返回 -1


# 示例
a = 2
y = 10
c1 = 100
c2 = 2
c3 = 2

print(min_cost_to_transform(a, y, c1, c2, c3))  # 输出最少花费

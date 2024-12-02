# -*- coding = utf-8 -*-
# @Time : 2023/11/20 21:04
# @Author : bobobobn
# @File : train_qt.py
# @Software: PyCharm
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton,QLabel,QLineEdit
import sys

if __name__ == '__main__':
    # 2.创建一个 QApplication 对象，指向QApplication ，接受命令行参数

    app = QApplication(sys.argv)
    #  3. 创建一个  QWidget对象
    w = QWidget()

    # 4. 设置窗口标题
    w.setWindowTitle("模型训练程序")
    w.resize(500, 500)

    btn = QPushButton("按钮")
    btn.setParent(w)

    # 5. 展示窗口
    label = QLabel("账号: ", w)
    # 显示位置与大小 ： x, y , w, h
    label.setGeometry(20, 20, 30, 30)

    edit = QLineEdit(w)
    edit.setPlaceholderText("请输入账号")
    edit.setGeometry(55, 20, 200, 20)

    w.show()

    # 程序进行循环等待状态
    app.exec()
# -*- coding = utf-8 -*-
# @Time : 2024/10/11 15:56
# @Author : bobobobn
# @File : test.py
# @Software: PyCharm
class BaseClass:
    def make_ssv_label(self):
        raise NotImplementedError("Subclass must implement this abstract method")

    def get_ssv(self):
        print(self.make_ssv_label())

class SubClass(BaseClass):
    def __init__(self, ssv_set, signals_tr_ssv, labels_tr_ssv):
        self.ssv_set = ssv_set
        self.signals_tr_ssv = signals_tr_ssv
        self.labels_tr_ssv = labels_tr_ssv

    # 重写make_ssv_label方法
    def make_ssv_label(self):
        # 这里是子类自定义的逻辑，可以根据需要实现
        return "SSV label from subclass"

# 示例调用
ssv_set = "some ssv data"
signals_tr_ssv = "training signals"
labels_tr_ssv = "training labels"

subclass_instance = SubClass(ssv_set, signals_tr_ssv, labels_tr_ssv)

# 调用get_ssv，会自动调用子类的make_ssv_label方法
ssv_data = subclass_instance.get_ssv()
print(ssv_data)

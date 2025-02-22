# -*- coding = utf-8 -*-
# @Time : 2024/10/4 15:34
# @Author : bobobobn
# @File : data_base.py.py
# @Software: PyCharm

from data.data_preprocess import normalize, get_mean_data, train_set_split_ssv, get_imb_data, CWRUdata, create_base_dataset,\
    create_knowledge_label, standardize, get_imb_data_beta
import numpy as np


class DataBase():
    def __init__(self, train_frac=0.8, ssv_size=100, normal_size=100, excep_size=100, ssv_size_max=200, beta=None):
        signals_tr_np, labels_tr_np, signals_tt_np, labels_tt_np = create_base_dataset(file_name='data/annotations.txt',\
                                                                                       train_frac=train_frac, dim=1024)
        # signals_tr_np, signals_tt_np = normalize(signals_tr_np, signals_tt_np)
        signals_tr_np, signals_tt_np = standardize(signals_tr_np, signals_tt_np)
        self.signals_tt_np, self.labels_tt_np = get_mean_data(signals_tt_np,labels_tt_np)
        signals_tr_np, labels_tr_np, self.ssv_set = train_set_split_ssv(signals_tr_np, labels_tr_np, ssv_size, ssv_size_max)
        if beta is not None:
            self.signals_tr_ssv, self.labels_tr_ssv = get_imb_data_beta(signals_tr_np, labels_tr_np, beta=beta, total_sample=100)
        else:
            self.signals_tr_ssv, self.labels_tr_ssv = get_imb_data(signals_tr_np, labels_tr_np, normal_num=normal_size, excep_num=excep_size)


    def make_ssv_label(self):
        raise NotImplementedError("Subclass must implement this abstract method")

    def get_train(self):
        return CWRUdata(self.signals_tr_ssv, self.labels_tr_ssv)

    def get_ssv(self):
        return CWRUdata(self.ssv_set, self.make_ssv_label())

    def get_test(self):
        return CWRUdata(self.signals_tt_np, self.labels_tt_np)


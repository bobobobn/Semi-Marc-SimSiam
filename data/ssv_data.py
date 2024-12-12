# -*- coding = utf-8 -*-
# @Time : 2024/10/4 16:19
# @Author : bobobobn
# @File : ssv_data.py
# @Software: PyCharm

from data.data_base import *

class DCAESSVData(DataBase):
    def make_ssv_label(self):
        dcae_label = create_dcae_label(
            CWRUdata(self.ssv_set, np.zeros((self.ssv_set.shape[0], 1))))
        mean = np.mean(dcae_label, axis=0)
        std_dev = np.std(dcae_label, axis=0)
        mean = mean.reshape((1, -1))
        std_dev = std_dev.reshape((1, -1))
        dcae_label = (dcae_label - mean) / std_dev
        return dcae_label


class KnowledgeSSVData(DataBase):
    def make_ssv_label(self):
        return create_knowledge_label(self.ssv_set)


class DCAEKnowledgeSSVData(DataBase):
    def make_ssv_label(self):
        dcae_label = create_dcae_label(
            CWRUdata(self.ssv_set, np.zeros((self.ssv_set.shape[0], 1))))
        mean = np.mean(dcae_label, axis=0)
        std_dev = np.std(dcae_label, axis=0)
        mean = mean.reshape((1, -1))
        std_dev = std_dev.reshape((1, -1))
        # 标准化矩阵
        dcae_label = (dcae_label - mean) / std_dev
        knowledge_label = create_knowledge_label(self.ssv_set)
        proxy_label = np.hstack((dcae_label, knowledge_label))
        return proxy_label




class KmeansSSVData(DataBase):
    def make_ssv_label(self):
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=6, random_state=42)
        kmeans.fit(self.ssv_set)
        y_kmeans = kmeans.predict(self.ssv_set)
        l = {}
        for v in y_kmeans:
            if v not in l.keys():
                l[v] = 1
            else:
                l[v] += 1
        return y_kmeans

class NonLabelSSVData(DataBase):

    def make_ssv_label(self):

        return np.zeros((len(self.ssv_set), 1))

    def get_train(self, transforms = None):
        return CWRUdata(self.signals_tr_ssv, self.labels_tr_ssv, transforms)

    def get_ssv(self, transforms = None):
        return CWRUdata(self.ssv_set , self.make_ssv_label(), transforms)

    def get_test(self, transforms = None):
        return CWRUdata(self.signals_tt_np, self.labels_tt_np, transforms)

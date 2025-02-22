# Semi-Marc SimSiam
 
****
运行simsiam，模型保存在checkpoints/simsiam

````
cd simsiam
python main_simsiam.py
````
运行BYOL，模型保存在checkpoints/byol
````
cd byol
python main_byol.py
````
运行SimCLR，模型保存在checkpoints/simclr
````
cd SimCLR
python main_simclr.py
````
微调阶段，基于Semi-Marc算法，微调分为三个阶段，阶段一直接用原始长尾数据训练分类层，阶段二用阶段一的模型生成无标签数据集的伪标签重新训练分类层，阶段三用Marc调整决策面
````
python train_semi_marc.py --pretrained_model "path_to_pretrained_model"
````

数据集封装为NonLabelSSVData类，继承自DataBase类，DataBase类在初始化时处理原始数据，从data/annotations.txt读取data/raw_data中各个文件对应的类别label，并分割成自监督数据集、训练集、测试集。由beta参数或指定正常类与故障类的样本个数构造长尾分布训练集。
****
参考文献

SimSiam: Chen X, He K. Exploring simple siamese representation learning[C]. Proceedings of the
 IEEE/CVF conference on computer vision and pattern recognition, 2021: 15750-15758.

SimCLR: Chen T, Kornblith S, Norouzi M, et al. A simple framework for contrastive learning of visual
 representations[C]. International conference on machine learning, 2020: 1597-1607.
 
BYOL: Grill J-B, Strub F, Altché F, et al. Bootstrap your own latent-a new approach to self-supervised
 learning[J]. Advances in neural information processing systems, 2020, 33: 21271-21284.

Semi: Yang Y, Xu Z. Rethinking the value of labels for improving class-imbalanced learning[J]. Ad
vances in neural information processing systems, 2020, 33: 19290-19301.

MARC:  Wang Y, Zhang B, Hou W, et al. Margin calibration for long-tailed visual recognition[C]. Asian
 Conference on Machine Learning, 2023: 1101-1116.
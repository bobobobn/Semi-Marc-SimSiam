# -*- coding = utf-8 -*-
# @Time : 2023/11/15 23:02
# @Author : bobobobn
# @File : gen_new_sample.py
# @Software: PyCharm
import torch

gen = torch.load('models/gen.pth')
device = 'cuda' if torch.cuda.is_available() else 'cpu'
fake_imgs = torch.randn(size = (1000 , 100), device = device)
gen_img = (gen(fake_imgs)+1)/2
print(gen_img)
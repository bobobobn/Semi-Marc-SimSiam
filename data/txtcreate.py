# -*- coding = utf-8 -*-
# @Time : 2023/5/24 11:02
# @Author : bobobobn
# @File : txtcreate.py
# @Software: PyCharm
import pandas as pd
frame = pd.read_excel(r'C:\Users\bobobob\Desktop\1.xlsx')
with open('annotations.txt', 'w') as f:
    for i in range(frame.shape[0]):
        if frame['Inner Race'][i]!='*':
            f.write(frame['Inner Race'][i]+'\t1'+ '\n')
        if frame['Ball'][i] != '*':
            f.write(frame['Ball'][i]+ '\t2'+'\n')
        if frame['Centered@6:00'][i] != '*':
            f.write(frame['Centered@6:00'][i]+ '\t3'+ '\n')
        if frame['Centered@3:00'][i] != '*':
            f.write(frame['Centered@3:00'][i]+ '\t4'+ '\n')
        if frame['Centered@12:00'][i] != '*':
            f.write(frame['Centered@12:00'][i]+ '\t5'+ '\n')
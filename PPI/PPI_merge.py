# 导入数据
import pandas as pd
import numpy as np

BioPlex_293T = pd.read_csv('D:\\jiazhuangxian\\Network_JZX\\PPI\\BioPlex+HuRI+innatedb+INSIDER-1.tsv', sep='\t', header = None,
                           names = ['Gene1', 'Gene2', 'sheet1'])
BioPlex_HCT116 = pd.read_csv('D:\\jiazhuangxian\\Network_JZX\\PPI\\Interactome3D+PhosphositePlus.tsv', sep='\t', header = None,
                           names = ['Gene3', 'Gene4', 'sheet2'])

# 数据框转字符串型
BioPlex_293T = BioPlex_293T.applymap(str) 


BioPlex_HCT116 = BioPlex_HCT116.applymap(str)


# 因PPI关系的无序性，将第二个数据框的两列变换顺序，分别求两数据框交集
df1 = pd.merge(BioPlex_293T, BioPlex_HCT116, left_on=['Gene1','Gene2'], right_on=['Gene3','Gene4'], how = 'inner', indicator = True)
df2 = pd.merge(BioPlex_293T, BioPlex_HCT116, left_on=['Gene1','Gene2'], right_on=['Gene4','Gene3'], how = 'inner', indicator = True)


# 将第二个数据框的两列变换顺序，分别求两数据框并集
df3 = pd.merge(BioPlex_293T, BioPlex_HCT116, left_on=['Gene1','Gene2'],right_on=['Gene3','Gene4'], how = 'outer', indicator = True)
df4 = pd.merge(BioPlex_293T, BioPlex_HCT116, left_on=['Gene1','Gene2'],right_on=['Gene4','Gene3'], how = 'outer', indicator = True)


# 分别求两并集与交集之间的差集，以完全去除重复数据
df5 = df1.append(df2)
df6 = df3.append(df5).drop_duplicates(subset=['Gene1','Gene2'],keep=False)
df7 = df4.append(df5).drop_duplicates(subset=['Gene3','Gene4'],keep=False)

# 删除NA列
df8 = df6.drop(['Gene3','Gene4','sheet2'], axis=1) 


df9 = df7.drop(['Gene1','Gene2','sheet1'], axis=1) 


# 修改第二个数据框索引，使其与第一个数据框匹配
df9 = df9.rename(columns = {'Gene3':'Gene1', 'Gene4':'Gene2', 'sheet2':'sheet1'})


# 将两数据框纵向合并
df10 = pd.concat([df8, df9], axis=0, join='outer', ignore_index=True, sort=False)


# 将交集部分批注好两个数据来源
df5['sheet1'] = df5['sheet1'] + '|' + df5['sheet2']

# 纵向拼接差集总数据与交集总数据
df11 = pd.concat([df10, df5], axis=0, join='inner', ignore_index=True, sort=False)

print(df11[:6])

# 将PPI总表写出为.tsv格式
df11.to_csv('D:\\jiazhuangxian\\Network_JZX\\PPI\\BioPlex+HuRI+innatedb+INSIDER+Interactome3D+PhosphositePlus.tsv',
            sep='\t',index = False,header = None)

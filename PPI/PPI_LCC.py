# libraries
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

# Build your graph

df = pd.read_csv('C:\\Users\\GaoKai\\Desktop\\PPI.csv', sep=',', header = None)
df = df.applymap(str)

G = nx.Graph()

G.add_edges_from(zip(df[0], df[1]))

# 连通分量的节点列表，包含每个连通图的节点列表
nx.connected_components(G)

#获取最大的连通子图
largest_cc = max(nx.connected_components(G), key=len)

# 返回的是列表，但是元素是图，这些分量按照节点数目从大到小排列，所以第一个就是最大的连通分量
H = list(G.subgraph(c) for c in nx.connected_components(G))[0]

# 取最大连通子图并作图 21.08.07
G_LCC = max((G.subgraph(c) for c in nx.connected_components(G)),key=len)

# plt.figure()
# nx.draw(G_LCC, with_labels= False, node_color="green", edge_color="grey", node_size=6)  # True
# plt.savefig("G_LCC.pdf")

nx.write_edgelist(G_LCC, "C:\\Users\\GaoKai\\Desktop\\PPI_LCC.csv", delimiter=',', data=False)

# ENTREZID转换为SYMBOL后，去掉SYMBOL为缺失值的PPI边
import pandas as pd

df = pd.read_csv(r'C:\Users\GaoKai\Desktop\network pharmacology\data\Union+PPI_LCC_geneSymbol.tsv', sep='\t', header = None)

# 去除indication列有空值的行
df_nonNA = df.dropna(axis=0)

df_nonNA.to_csv(r'C:\Users\GaoKai\Desktop\network pharmacology\data\Union+PPI_LCC_geneSymbol_nonNA.tsv', sep='\t', index=False, header = None)
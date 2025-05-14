import numpy as np
from typing import Dict, List, Tuple
import networkx as nx
import pandas as pd

def compute_transition_matrix(multiplex_het: MultiplexHet) -> np.ndarray:
    """
    计算异构多重网络的转移矩阵
    参考 RandomWalkRestartMH 包中的实现逻辑
    """
    # 1. 从第一个多重网络构建邻接矩阵
    adj_matrix1 = {}
    for layer_name, graph in multiplex_het.multiplex1.graphs.items():
        adj = nx.adjacency_matrix(graph).toarray()
        # 计算度矩阵的逆
        degrees = np.sum(adj, axis=1)
        deg_inv = np.zeros_like(adj)
        np.fill_diagonal(deg_inv, 1 / degrees)
        # 计算归一化的邻接矩阵
        adj_matrix1[layer_name] = np.dot(deg_inv, adj)

    # 2. 从第二个多重网络构建邻接矩阵
    adj_matrix2 = {}
    for layer_name, graph in multiplex_het.multiplex2.graphs.items():
        adj = nx.adjacency_matrix(graph).toarray()
        degrees = np.sum(adj, axis=1)
        deg_inv = np.zeros_like(adj)
        np.fill_diagonal(deg_inv, 1 / degrees)
        adj_matrix2[layer_name] = np.dot(deg_inv, adj)

    # 3. 构建两个网络之间的偶联矩阵
    relations_matrix = np.zeros((
        len(multiplex_het.multiplex1.Pool_of_Nodes),
        len(multiplex_het.multiplex2.Pool_of_Nodes)
    ))
    
    # 根据关系数据填充偶联矩阵
    for _, row in multiplex_het.relations.iterrows():
        i = multiplex_het.multiplex1.Pool_of_Nodes.index(row['Source'])
        j = multiplex_het.multiplex2.Pool_of_Nodes.index(row['Target'])
        relations_matrix[i, j] = 1

    # 归一化偶联矩阵
    row_sums = np.sum(relations_matrix, axis=1)
    col_sums = np.sum(relations_matrix, axis=0)
    
    relations_matrix_norm1 = np.zeros_like(relations_matrix)
    relations_matrix_norm2 = np.zeros_like(relations_matrix.T)
    
    for i in range(relations_matrix.shape[0]):
        if row_sums[i] > 0:
            relations_matrix_norm1[i, :] = relations_matrix[i, :] / row_sums[i]
            
    for i in range(relations_matrix.shape[1]):
        if col_sums[i] > 0:
            relations_matrix_norm2[i, :] = relations_matrix.T[i, :] / col_sums[i]

    # 4. 构建完整的转移矩阵
    n1 = len(multiplex_het.multiplex1.Pool_of_Nodes)
    n2 = len(multiplex_het.multiplex2.Pool_of_Nodes)
    total_size = n1 + n2
    
    transition_matrix = np.zeros((total_size, total_size))
    
    # 填充第一个多重网络的转移概率
    for adj in adj_matrix1.values():
        transition_matrix[:n1, :n1] += adj / len(adj_matrix1)
    
    # 填充第二个多重网络的转移概率
    for adj in adj_matrix2.values():
        transition_matrix[n1:, n1:] += adj / len(adj_matrix2)
    
    # 填充网络间的转移概率
    delta = 0.5  # 网络间跳转概率
    transition_matrix[:n1, n1:] = relations_matrix_norm1 * delta
    transition_matrix[n1:, :n1] = relations_matrix_norm2 * delta
    
    return transition_matrix

def random_walk_restart_multiplexHet(
    transition_matrix: np.ndarray,
    multiplex_het: MultiplexHet,
    seed_disease: List[str],
    seed_nodes: List[str],
    restart_prob: float = 0.7,
    epsilon: float = 1e-6,
    max_iter: int = 100
) -> Dict[str, pd.DataFrame]:
    """
    实现随机游走重启算法
    参考 RandomWalkRestartMH 包中的实现逻辑
    """
    # 1. 构建初始概率向量
    n1 = len(multiplex_het.multiplex1.Pool_of_Nodes)
    n2 = len(multiplex_het.multiplex2.Pool_of_Nodes)
    total_size = n1 + n2
    
    p0 = np.zeros(total_size)
    
    # 设置种子节点的初始概率
    for node in seed_disease:
        if node in multiplex_het.multiplex1.Pool_of_Nodes:
            idx = multiplex_het.multiplex1.Pool_of_Nodes.index(node)
            p0[idx] = 1
    
    for node in seed_nodes:
        if node in multiplex_het.multiplex2.Pool_of_Nodes:
            idx = n1 + multiplex_het.multiplex2.Pool_of_Nodes.index(node)
            p0[idx] = 1
    
    # 归一化初始概率向量
    if np.sum(p0) > 0:
        p0 = p0 / np.sum(p0)
    
    # 2. 迭代计算直到收敛
    pt = p0.copy()
    for _ in range(max_iter):
        pt_next = (1 - restart_prob) * np.dot(transition_matrix, pt) + restart_prob * p0
        
        # 检查收敛
        if np.sum(np.abs(pt_next - pt)) < epsilon:
            break
            
        pt = pt_next
    
    # 3. 提取结果
    results = {
        'RWRMH_Multiplex1': pd.DataFrame({
            'Node': multiplex_het.multiplex1.Pool_of_Nodes,
            'Score': pt[:n1]
        }),
        'RWRMH_GlobalResults': pd.DataFrame({
            'Node': (multiplex_het.multiplex1.Pool_of_Nodes + 
                    multiplex_het.multiplex2.Pool_of_Nodes),
            'Score': pt
        })
    }
    
    # 对结果进行排序
    for key in results:
        results[key] = results[key].sort_values('Score', ascending=False)
    
    return results
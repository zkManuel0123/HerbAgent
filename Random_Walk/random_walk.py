# 输入文件是由Herb_target_search.py生成的Formula_1_herb_ingredient.csv和Syndrome_target_search.py生成的Formula_1_syndrome_ingredient.csv

import os
import pandas as pd
import networkx as nx
import numpy as np
from typing import List, Dict, Union, Tuple
import scipy as sp
from scipy import sparse
import random
from pathlib import Path

# 设置工作路径
wkdir = "D:\\HerbAgent\\Random_Walk\\"
os.chdir(wkdir)

class MultiplexObject:
    """多重网络对象（完全对齐R代码实现）"""
    def __init__(self, graphs: Dict[str, nx.Graph], nodes: List[str] = None):
        # 验证输入
        if not isinstance(graphs, dict):
            raise ValueError("graphs参数必须是字典类型")
        
        # 验证每个图
        for name, graph in graphs.items():
            validate_graph(graph)
        
        self.graphs = {}
        # 处理每一层网络
        for layer_name, graph in graphs.items():
            # 确保节点有名称
            if not all(isinstance(n, str) for n in graph.nodes()):
                graph = nx.convert_node_labels_to_integers(graph, label_attribute='name')
                graph = nx.relabel_nodes(graph, {n: str(n) for n in graph.nodes()})
            
            # 简化网络
            graph = simplify_graph(graph)
            self.graphs[layer_name] = graph
        
        # 获取或使用提供的节点列表
        if nodes is None:
            self.Pool_of_Nodes = sorted(set().union(*[set(g.nodes()) 
                                                    for g in self.graphs.values()]))
        else:
            self.Pool_of_Nodes = sorted(set(nodes))
        
        self.Number_of_Nodes = len(self.Pool_of_Nodes)
        self.Number_of_Layers = len(self.graphs)
        
        # 为每层添加缺失的节点
        for graph in self.graphs.values():
            add_missing_nodes(graph, self.Pool_of_Nodes)
            
        # 设置边的类型属性
        for layer_name, graph in self.graphs.items():
            nx.set_edge_attributes(graph, layer_name, 'type')

    def get_adjacency_matrices(self) -> Dict[str, sp.sparse.spmatrix]:
        """获取每层的邻接矩阵"""
        matrices = {}
        for layer_name, graph in self.graphs.items():
            # 获取邻接矩阵
            adj = nx.adjacency_matrix(graph)
            # 归一化
            adj = normalize_adjacency_matrix(adj)
            matrices[layer_name] = adj
        return matrices

    def __str__(self):
        """打印对象信息（对应R代码中的print.Multiplex）"""
        output = [
            f"Number of Layers: {self.Number_of_Layers}",
            f"Number of Nodes: {self.Number_of_Nodes}"
        ]
        for name, graph in self.graphs.items():
            output.append(f"\nLayer {name}:")
            output.append(f"Nodes: {graph.number_of_nodes()}")
            output.append(f"Edges: {graph.number_of_edges()}")
        return "\n".join(output)

class MultiplexHet:
    """异构多重网络对象（完全对齐R代码实现）"""
    def __init__(self, multiplex1: MultiplexObject, multiplex2: MultiplexObject, 
                 relations: pd.DataFrame):
        # 验证输入
        if not isinstance(multiplex1, MultiplexObject):
            raise ValueError("multiplex1必须是MultiplexObject类型")
        if not isinstance(multiplex2, MultiplexObject):
            raise ValueError("multiplex2必须是MultiplexObject类型")
        
        # 验证关系数据
        validate_relations(relations, multiplex1, multiplex2)
        
        self.multiplex1 = multiplex1
        self.multiplex2 = multiplex2
        
        # 处理权重
        if 'weight' not in relations.columns:
            relations['weight'] = 1.0
        else:
            relations['weight'] = normalize_weights(relations['weight'].values)
        
        self.relations = relations
        
        # 构建二分图矩阵
        print("构建二分图矩阵...")
        self.bipartite_matrix = get_bipartite_matrix(
            self.multiplex1.Pool_of_Nodes,
            self.multiplex2.Pool_of_Nodes,
            self.relations
        )
        
        # 扩展到多重网络尺寸
        print("扩展二分图矩阵...")
        self.supra_bipartite_matrix = expand_bipartite_matrix(
            self.bipartite_matrix,
            self.multiplex1.Number_of_Nodes,
            self.multiplex1.Number_of_Layers,
            self.multiplex2.Number_of_Nodes,
            self.multiplex2.Number_of_Layers
        )

    def __str__(self):
        """打印对象信息（对应R代码中的print.MultiplexHet）"""
        output = [
            f"Number of Layers Multiplex 1: {self.multiplex1.Number_of_Layers}",
            f"Number of Nodes Multiplex 1: {self.multiplex1.Number_of_Nodes}",
            "\nMultiplex 1 Layers:"
        ]
        for name, graph in self.multiplex1.graphs.items():
            output.append(f"\nLayer {name}:")
            output.append(f"Nodes: {graph.number_of_nodes()}")
            output.append(f"Edges: {graph.number_of_edges()}")
            
        output.extend([
            f"\nNumber of Layers Multiplex 2: {self.multiplex2.Number_of_Layers}",
            f"Number of Nodes Multiplex 2: {self.multiplex2.Number_of_Nodes}",
            "\nMultiplex 2 Layers:"
        ])
        for name, graph in self.multiplex2.graphs.items():
            output.append(f"\nLayer {name}:")
            output.append(f"Nodes: {graph.number_of_nodes()}")
            output.append(f"Edges: {graph.number_of_edges()}")
            
        return "\n".join(output)

def create_multiplex(graph_dict: Dict[str, nx.Graph]) -> MultiplexObject:
    """创建多重网络对象（完全对齐R代码实现）"""
    try:
        return MultiplexObject(graph_dict)
    except Exception as e:
        print(f"创建多重网络对象时出错: {str(e)}")
        raise

def create_multiplexHet(multiplex1: MultiplexObject, multiplex2: MultiplexObject, 
                       relations: pd.DataFrame) -> MultiplexHet:
    """创建异构多重网络对象（完全对齐R代码实现）"""
    try:
        return MultiplexHet(multiplex1, multiplex2, relations)
    except Exception as e:
        print(f"创建异构多重网络对象时出错: {str(e)}")
        raise

def simplify_graph(graph: nx.Graph) -> nx.Graph:
    """简化图（对应R代码中的simplify.layers函数）"""
    # 确保是无向图
    graph = graph.to_undirected()
    
    # 处理权重
    edges_data = list(graph.edges(data=True))
    if edges_data:  # 确保图中有边
        if 'weight' in edges_data[0][2]:
            weights = [d['weight'] for _, _, d in edges_data]
            if min(weights) != max(weights):
                # 归一化权重到[0,1]区间
                a = min(weights) / max(weights)
                b = 1
                weights_norm = [(b-a)*(w-min(weights))/(max(weights)-min(weights)) + a 
                              for w in weights]
                for (u, v), w in zip(graph.edges(), weights_norm):
                    graph[u][v]['weight'] = w
            else:
                nx.set_edge_attributes(graph, 1, 'weight')
        else:
            nx.set_edge_attributes(graph, 1, 'weight')
    
    # 移除自环和重复边
    graph = nx.Graph(graph)
    return graph

def normalize_adjacency_matrix(adj_matrix: sp.sparse.spmatrix) -> sp.sparse.spmatrix:
    """列归一化邻接矩阵（与R代码一致）"""
    # 计算每列的和
    col_sums = np.array(adj_matrix.sum(axis=0)).flatten()
    # 避免除零
    col_sums[col_sums == 0] = 1
    # 创建归一化矩阵
    norm_matrix = sp.sparse.diags(1 / col_sums)
    # 返回归一化后的矩阵
    return adj_matrix.dot(norm_matrix)

def get_bipartite_matrix(
    names_mul1: List[str],
    names_mul2: List[str],
    relations: pd.DataFrame
) -> sp.sparse.spmatrix:
    """构建二分图矩阵（对应R代码中的get.bipartite.graph函数）"""
    n1 = len(names_mul1)
    n2 = len(names_mul2)
    
    # 创建稀疏矩阵
    bipartite_matrix = sp.sparse.lil_matrix((n1, n2))
    
    # 获取排序后的节点名称
    names_mul1_order = sorted(names_mul1)
    names_mul2_order = sorted(names_mul2)
    
    # 创建节点名称到索引的映射
    mul1_idx = {name: i for i, name in enumerate(names_mul1_order)}
    mul2_idx = {name: i for i, name in enumerate(names_mul2_order)}
    
    # 填充矩阵
    for _, row in relations.iterrows():
        i = mul1_idx.get(row['Source'])
        j = mul2_idx.get(row['Target'])
        if i is not None and j is not None:
            bipartite_matrix[i, j] = 1
    
    return bipartite_matrix.tocsr()

def expand_bipartite_matrix(
    bipartite_matrix: sp.sparse.spmatrix,
    n1: int,
    l1: int,
    n2: int,
    l2: int
) -> sp.sparse.spmatrix:
    """扩展二分图矩阵到多重网络尺寸"""
    # 创建扩展后的矩阵
    expanded_matrix = sp.sparse.lil_matrix((n1 * l1, n2 * l2))
    
    # 复制原始矩阵到每一层
    for i in range(l1):
        for j in range(l2):
            expanded_matrix[i*n1:(i+1)*n1, j*n2:(j+1)*n2] = bipartite_matrix
    
    return expanded_matrix.tocsr()

def geometric_mean(scores: np.ndarray, num_layers: int, num_nodes: int) -> np.ndarray:
    """完全按照R代码实现的几何平均值计算"""
    final_score = np.zeros(num_nodes)
    for i in range(num_nodes):
        # 使用与R代码相同的索引方式
        layer_scores = scores[np.arange(i, num_nodes * num_layers, num_nodes)]
        # 避免log(0)
        layer_scores = np.maximum(layer_scores, 1e-10)
        final_score[i] = np.exp(np.mean(np.log(layer_scores)))
    return final_score

def regular_mean(scores: np.ndarray, num_layers: int, num_nodes: int) -> np.ndarray:
    """算术平均值计算"""
    final_score = np.zeros(num_nodes)
    for i in range(num_nodes):
        layer_scores = scores[np.arange(i, num_nodes * num_layers, num_nodes)]
        final_score[i] = np.mean(layer_scores)
    return final_score

def sum_values(scores: np.ndarray, num_layers: int, num_nodes: int) -> np.ndarray:
    """求和计算"""
    final_score = np.zeros(num_nodes)
    for i in range(num_nodes):
        layer_scores = scores[np.arange(i, num_nodes * num_layers, num_nodes)]
        final_score[i] = np.sum(layer_scores)
    return final_score

def get_seed_scores_multHet(
    multiplex1_seeds: List[str],
    multiplex2_seeds: List[str],
    eta: float,
    l1: int,
    l2: int,
    tau1: List[float],
    tau2: List[float]
) -> pd.DataFrame:
    """计算种子节点的初始分数（与R代码对齐）"""
    n = len(multiplex1_seeds)
    m = len(multiplex2_seeds)
    
    if n != 0 and m != 0:
        # 两个网络都有种子节点
        seed_multiplex1_layer = [f"{node}_{layer}" 
                               for layer in range(1, l1+1) 
                               for node in multiplex1_seeds]
        seed_multiplex2_layer = [f"{node}_{layer}" 
                               for layer in range(1, l2+1) 
                               for node in multiplex2_seeds]
        
        seeds_multiplex1_scores = np.repeat((1-eta) * np.array(tau1) / n, n)
        seeds_multiplex2_scores = np.repeat(eta * np.array(tau2) / m, m)
    else:
        eta = 1
        if n == 0:
            seed_multiplex1_layer = []
            seeds_multiplex1_scores = np.array([])
            seed_multiplex2_layer = [f"{node}_{layer}" 
                                   for layer in range(1, l2+1) 
                                   for node in multiplex2_seeds]
            seeds_multiplex2_scores = np.repeat(np.array(tau2) / m, m)
        else:
            seed_multiplex1_layer = [f"{node}_{layer}" 
                                   for layer in range(1, l1+1) 
                                   for node in multiplex1_seeds]
            seeds_multiplex1_scores = np.repeat(np.array(tau1) / n, n)
            seed_multiplex2_layer = []
            seeds_multiplex2_scores = np.array([])
    
    return pd.DataFrame({
        'Seeds_ID': seed_multiplex1_layer + seed_multiplex2_layer,
        'Score': np.concatenate([seeds_multiplex1_scores, seeds_multiplex2_scores])
    })

def compute_transition_matrix(multiplex_het: MultiplexHet, delta: float = 0.5) -> sp.sparse.spmatrix:
    """计算转移矩阵（与R代码完全对齐）"""
    n1 = multiplex_het.multiplex1.Number_of_Nodes
    n2 = multiplex_het.multiplex2.Number_of_Nodes
    l1 = multiplex_het.multiplex1.Number_of_Layers
    l2 = multiplex_het.multiplex2.Number_of_Layers
    
    print(f"构建转移矩阵 (大小: {n1*l1 + n2*l2} x {n1*l1 + n2*l2})")
    
    # 构建第一个多重网络的邻接矩阵
    adj_matrices1 = []
    for graph in multiplex_het.multiplex1.graphs.values():
        adj = nx.adjacency_matrix(graph)
        # 归一化
        adj = normalize_adjacency_matrix(adj)
        adj_matrices1.append(adj)
    
    # 构建第二个多重网络的邻接矩阵
    adj_matrices2 = []
    for graph in multiplex_het.multiplex2.graphs.values():
        adj = nx.adjacency_matrix(graph)
        # 归一化
        adj = normalize_adjacency_matrix(adj)
        adj_matrices2.append(adj)
    
    # 构建完整的转移矩阵
    total_size = n1 * l1 + n2 * l2
    transition_matrix = sp.sparse.lil_matrix((total_size, total_size))
    
    # 填充层内转移
    for i in range(l1):
        start = i * n1
        transition_matrix[start:start+n1, start:start+n1] = (1 - delta) * adj_matrices1[i]
    
    offset = n1 * l1
    for i in range(l2):
        start = offset + i * n2
        transition_matrix[start:start+n2, start:start+n2] = (1 - delta) * adj_matrices2[i]
    
    # 填充层间转移
    layer_jump_prob = delta / (l2 - 1) if l2 > 1 else delta
    
    # 获取二分图关系矩阵
    relations_matrix = multiplex_het.bipartite_matrix
    
    # 确保关系矩阵被归一化
    relations_matrix_norm1 = normalize_adjacency_matrix(relations_matrix)
    relations_matrix_norm2 = normalize_adjacency_matrix(relations_matrix.T)
    
    # 填充层间转移
    for i in range(l1):
        for j in range(l2):
            start1 = i * n1
            start2 = offset + j * n2
            
            # 从网络1到网络2的转移
            block12 = layer_jump_prob * relations_matrix_norm1
            transition_matrix[start1:start1+n1, start2:start2+n2] = block12
            
            # 从网络2到网络1的转移
            block21 = layer_jump_prob * relations_matrix_norm2
            transition_matrix[start2:start2+n2, start1:start1+n1] = block21
    
    print("转移矩阵建完成")
    return transition_matrix.tocsr()

def random_walk_restart_multiplexHet(
    transition_matrix: sp.sparse.spmatrix,
    multiplex_het: MultiplexHet,
    seed_disease: List[str],
    seed_nodes: List[str],
    restart_prob: float = 0.7,
    epsilon: float = 1e-10,
    max_iter: int = 1000,
    eta: float = 0.5,
    tau1: List[float] = None,
    tau2: List[float] = None,
    mean_type: str = "Geometric"
) -> Dict[str, pd.DataFrame]:
    """实现随机游走重启算法（完全对齐R代码）"""
    n1 = multiplex_het.multiplex1.Number_of_Nodes
    n2 = multiplex_het.multiplex2.Number_of_Nodes
    l1 = multiplex_het.multiplex1.Number_of_Layers
    l2 = multiplex_het.multiplex2.Number_of_Layers
    
    # 置默认的tau值
    if tau1 is None:
        tau1 = [1/l1] * l1
    if tau2 is None:
        tau2 = [1/l2] * l2
    
    # 获取种子节点分数
    seed_scores = get_seed_scores_multHet(
        seed_disease, seed_nodes, eta, l1, l2, tau1, tau2
    )
    
    # 初始化概率向量
    p0 = np.zeros(n1 * l1 + n2 * l2)
    for _, row in seed_scores.iterrows():
        node, layer = row['Seeds_ID'].split('_')
        layer = int(layer) - 1
        if node in multiplex_het.multiplex1.Pool_of_Nodes:
            idx = multiplex_het.multiplex1.Pool_of_Nodes.index(node)
            p0[layer * n1 + idx] = row['Score']
        elif node in multiplex_het.multiplex2.Pool_of_Nodes:
            idx = multiplex_het.multiplex2.Pool_of_Nodes.index(node)
            p0[n1 * l1 + layer * n2 + idx] = row['Score']
    
    # 归一化初始向量
    if np.sum(p0) > 0:
        p0 = p0 / np.sum(p0)
    
    # 迭代计算
    pt = p0.copy()
    for _ in range(max_iter):
        pt_old = pt.copy()
        pt = (1 - restart_prob) * transition_matrix.dot(pt) + restart_prob * p0
        
        # 检查收敛
        residue = np.sqrt(np.sum((pt - pt_old) ** 2))
        if residue < epsilon:
            break
    
    # 计算最终分数
    if mean_type == "Geometric":
        scores1 = geometric_mean(pt[:n1 * l1], l1, n1)
        scores2 = geometric_mean(pt[n1 * l1:], l2, n2)
    elif mean_type == "Arithmetic":
        scores1 = regular_mean(pt[:n1 * l1], l1, n1)
        scores2 = regular_mean(pt[n1 * l1:], l2, n2)
    else:
        scores1 = sum_values(pt[:n1 * l1], l1, n1)
        scores2 = sum_values(pt[n1 * l1:], l2, n2)
    
    # 创建结果数据框
    results = {
        'RWRMH_Multiplex1': pd.DataFrame({
            'Node': multiplex_het.multiplex1.Pool_of_Nodes,
            'Score': scores1
        }),
        'RWRMH_GlobalResults': pd.DataFrame({
            'Node': (multiplex_het.multiplex1.Pool_of_Nodes + 
                    multiplex_het.multiplex2.Pool_of_Nodes),
            'Score': np.concatenate([scores1, scores2])
        })
    }
    
    # 排序并移除种子节点
    for key in results:
        results[key] = results[key].sort_values('Score', ascending=False)
        results[key] = results[key][~results[key]['Node'].isin(seed_disease + seed_nodes)]
    
    return results

def validate_graph(graph: nx.Graph) -> bool:
    """验证图对象的有效性"""
    if not isinstance(graph, nx.Graph):
        raise ValueError("输入必须是networkx.Graph类型")
    if graph.number_of_nodes() == 0:
        raise ValueError("图不能为空")
    return True

def validate_relations(relations: pd.DataFrame, multiplex1: MultiplexObject, 
                      multiplex2: MultiplexObject) -> bool:
    """验证关系数据的有效性"""
    if not isinstance(relations, pd.DataFrame):
        raise ValueError("relations必须是pandas.DataFrame类型")
    
    required_cols = {'Source', 'Target'}
    if not set(relations.columns) >= required_cols:
        raise ValueError(f"relations必须包含列: {required_cols}")
    
    source_nodes = set(relations['Source'])
    target_nodes = set(relations['Target'])
    
    missing_source = source_nodes - set(multiplex1.Pool_of_Nodes)
    if missing_source:
        raise ValueError(f"以下Source节点在multiplex1中不存在: {missing_source}")
    
    missing_target = target_nodes - set(multiplex2.Pool_of_Nodes)
    if missing_target:
        raise ValueError(f"以下Target节点在multiplex2中不存在: {missing_target}")
    
    return True

def normalize_weights(weights: np.ndarray) -> np.ndarray:
    """归一化权重（与R代码一致）"""
    if len(weights) == 0:
        return weights
    
    if min(weights) != max(weights):
        a = min(weights) / max(weights)
        b = 1
        return (b-a) * (weights - min(weights)) / (max(weights) - min(weights)) + a
    return np.ones_like(weights)

def add_missing_nodes(graph: nx.Graph, all_nodes: List[str]) -> nx.Graph:
    """添加缺失的节点（对应R代码中的add.missing.nodes函数）"""
    missing = set(all_nodes) - set(graph.nodes())
    if missing:
        graph.add_nodes_from(missing)
    return graph

class RandomWalkAnalyzer:
    """随机游走分析器"""
    def __init__(self):
        # 定义基础路径
        self.base_path = Path("D:/HerbAgent/Random_Walk")
        self.data_path = self.base_path / "randomwalk_data"
        self.results_path = self.base_path / "results"
        
        # 确保结果目录存在
        self.results_path.mkdir(parents=True, exist_ok=True)
        
        # 定义子目录
        self.formula_herb_path = self.data_path / "formula_herb_ingredient"
        self.formula_target_path = self.data_path / "formula_ingredient_target"
        self.syndrome_path = self.data_path / "syndrome"

    def get_formula_files(self):
        """获取所有方剂相关文件"""
        formula_files = {}
        for herb_file in self.formula_herb_path.glob("*.csv"):
            formula_name = herb_file.stem.split('_')[0]  # 提取方剂名称
            target_file = self.formula_target_path / f"{formula_name}_ingredient_target.csv"
            
            if target_file.exists():
                formula_files[formula_name] = {
                    'herb_ingredient': herb_file,
                    'ingredient_target': target_file
                }
        return formula_files

    def get_random_syndrome_file(self):
        """随机获取一个证候文件"""
        syndrome_files = list(self.syndrome_path.glob("*.csv"))
        return random.choice(syndrome_files) if syndrome_files else None

    def analyze_single_formula(self, formula_name: str, herb_file: Path, 
                             target_file: Path, syndrome_file: Path):
        """分析单个方剂"""
        try:
            print(f"开始分析方剂: {formula_name}")
            
            # 读取数据文件
            formula_ingredient = pd.read_csv(herb_file)
            ppi = pd.read_csv(self.base_path / "PPI_LCC_edges.csv", header=None)
            ingredient_target = pd.read_csv(target_file)
            syndrome = pd.read_csv(syndrome_file)
            
            print("创建并简化图...")
            # 创建并简化图
            formula_ingredient_graph = nx.from_pandas_edgelist(
                formula_ingredient,
                source='Herb_id',
                target='Ingredient_id'
            )
            formula_ingredient_graph = simplify_graph(formula_ingredient_graph)
            
            ppi_graph = nx.from_pandas_edgelist(
                ppi,
                source=0,
                target=1
            )
            ppi_graph = simplify_graph(ppi_graph)
            
            print("创建多重网络对象...")
            # 创建多重网络对象
            formula_ingredient_multiplex = create_multiplex({
                "Formula_ingredient": formula_ingredient_graph
            })
            print(f"Formula-Ingredient网络节点数: {formula_ingredient_multiplex.Number_of_Nodes}")
            print(f"Formula-Ingredient网络层数: {formula_ingredient_multiplex.Number_of_Layers}")
            
            ppi_multiplex = create_multiplex({
                "PPI": ppi_graph
            })
            print(f"PPI网络节点数: {ppi_multiplex.Number_of_Nodes}")
            print(f"PPI网络层数: {ppi_multiplex.Number_of_Layers}")
            
            print("处理关系数据...")
            # 过滤和处理关系数据
            gene_ingredient_relations = ingredient_target[
                ingredient_target['Target_name'].isin(ppi_multiplex.Pool_of_Nodes)
            ].rename(columns={
                'Ingredient_id': 'Source',
                'Target_name': 'Target'
            })
            print(f"过滤后的关系数量: {len(gene_ingredient_relations)}")
            
            print("创建异构多重网络...")
            # 创建异构多重网络
            multiplex_het = create_multiplexHet(
                formula_ingredient_multiplex,
                ppi_multiplex,
                gene_ingredient_relations
            )
            print("异构多重网络创建完成")
            print(multiplex_het)
            
            print("计算转移矩阵...")
            # 计算转移矩阵
            transition_matrix = compute_transition_matrix(multiplex_het, delta=0.5)
            print(f"转移矩阵形状: {transition_matrix.shape}")
            
            print("准备种子节点...")
            # 准备种子节点
            seed_nodes = syndrome['SYMBOL'].tolist()
            seed_disease = []
            print(f"种子节点数量: {len(seed_nodes)}")
            
            print("运行随机游走重启算法...")
            # 运行随机游走重启算法
            results = random_walk_restart_multiplexHet(
                transition_matrix=transition_matrix,
                multiplex_het=multiplex_het,
                seed_disease=seed_disease,
                seed_nodes=seed_nodes,
                restart_prob=0.7,
                epsilon=1e-10,
                max_iter=1000,
                eta=0.5,
                mean_type="Geometric"
            )
            
            print("保存结果...")
            # 保存结果
            result_file = self.results_path / f"{formula_name}_ingredient_score.csv"
            results['RWRMH_Multiplex1'].to_csv(result_file, index=False)
            
            print(f"方剂 {formula_name} 分析完成")
            return True
            
        except Exception as e:
            print(f"分析方剂 {formula_name} 时出错: {str(e)}")
            return False

    def run_analysis(self):
        """运行所有分析"""
        formula_files = self.get_formula_files()
        
        for formula_name, files in formula_files.items():
            syndrome_file = self.get_random_syndrome_file()
            if syndrome_file:
                self.analyze_single_formula(
                    formula_name,
                    files['herb_ingredient'],
                    files['ingredient_target'],
                    syndrome_file
                )
            else:
                print("未找到证候文件")

def run_random_walk_analysis():
    """供外部模块调用的主函数"""
    try:
        analyzer = RandomWalkAnalyzer()
        analyzer.run_analysis()
        return True  # 成功完成返回True
    except Exception as e:
        print(f"随机游走分析过程中发生错误: {str(e)}")
        return False  # 发生错误返回False

if __name__ == "__main__":
    run_random_walk_analysis()

# from Random_Walk.random_walk import run_random_walk_analysis

# # 运行分析
# run_random_walk_analysis()
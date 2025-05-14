#######################################
### Project: Network Medicine Framework for Identifying Drug Repurposing Opportunities for COVID-19.
### Description: Pipeline for Proximity: P1-P3
### Author: Xiao Gan
### email: jack dot xiao dot gan at gmail dot com 
### date: 1st March 2021
#######################################

import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import os
import re
from tqdm import tqdm 
import time

import sys
sys.path.append(r'D:\HerbAgent\PPI\utils')

import os
if os.path.exists(r'D:\HerbAgent\PPI\utils\guney_code\wrappers.py'):
    print("wrappers.py found.")
else:
    print("wrappers.py not found. Check the path.")

from guney_code import wrappers
from guney_code import network_utilities
from guney_code import network_utils
#import testdisease

import separation as tools
from multiprocessing import Pool
import multiprocessing



"""
Neccesary files:
guney_code(folder)
genes_annotated_0311.csv
separation.py
disease genes file: interactome.tsv
disease genes file: COVID19_Human_Targets.csv
drug target file: drug_targets_test.txt

"""


def convert(list1,entry_from='GeneID',entry_to='Symbol' ,dataset = 'data/interactome_2019_merged_protAnnots.csv'):
    # convert a list of entry1 to another list of entry2
    df = pd.read_csv(dataset)
    listx1 = list(df[entry_from].astype(str))
    listx2 = list(df[entry_to].astype(str))
    result_list =[]
    for i in list1:
        try:
            result_list.append(listx2[listx1.index(i)])
        except:
            raise Exception('Error occured')
    return result_list

def parse_drug_target(file1, output_file=None):
    # 修改函数以正确处理中文
    drug_targets = {}
    if output_file is not None:
        f = open(output_file, "a+", encoding='utf-8')
    
    # 使用 utf-8 编码读取文件
    with open(file1, 'r', encoding='utf-8') as file:
        for line in file:
            words = line.strip().split("\t")
            set1 = set()
            for word1 in words[1:]:
                if word1 != '' and word1 != '\n':
                    set1.add(word1.strip())
            drug_targets[words[0]] = set1
            if output_file is not None:
                f.write(f"{words[0]}\t{set1}\n")
    
    if output_file is not None:
        f.close()
    
    print(f'\n> Done parsing drug targets: read {len(drug_targets)} total drugs')
    # 打印检查第一个药物名称的编码
    if drug_targets:
        first_drug = next(iter(drug_targets))
        print(f"First drug name: {first_drug}")
        print(f"Encoding of first drug name: {first_drug.encode()}")
    
    return drug_targets

def single_proximity(sample):
    # --------------------------------------------------------
    #
    # LOADING NETWORK and DISEASE GENES
    #
    # --------------------------------------------------------
    drug_key = sample[0]
    disease_key = sample[1]
    drug_targets =sample[2]
    disease_genes =sample[3]
    network = sample [4]
    nsims = sample [5]

    disease_save = os.path.basename(disease_key)
    # 移除文件扩展名
    disease_save = os.path.splitext(disease_save)[0]
    
    # 修改文件名处理部分
    def clean_filename(filename):
        # 确保字符串是 UTF-8 编码
        if isinstance(filename, bytes):
            filename = filename.decode('utf-8')
        elif not isinstance(filename, str):
            filename = str(filename)
            
        # 移除非法字符，但保留中文字符
        illegal_chars = r'[<>:"/\\|?*]'
        cleaned = re.sub(illegal_chars, '_', filename)
        return cleaned

    # 确保 drug_key 是正确的 UTF-8 字符串
    if isinstance(drug_key, bytes):
        drug_key = drug_key.decode('utf-8')
    
    filename = f"{drug_key}_{disease_save}.txt"
    filename = clean_filename(filename)
    
    # multiple proximity loop. Call each key combination from the two dictionaries

    nodes_from =  set(drug_targets[drug_key]) & set(network.nodes())
    nodes_to =  set(disease_genes[disease_key]) & set(network.nodes())
    print (('drug=%s, disease=%s')%(drug_key, disease_key))

    if len(nodes_from) == 0 or len(nodes_to) == 0: # if no applicable target, stop
        return

    # computing proximity. Please set the parameters to proper values.
    d, (mean, sd), (z, pval) = wrappers.calculate_proximity(network, nodes_from, nodes_to, n_random=nsims, min_bin_size = 100, seed=None)
    #print (('d=%s, z=%s, (mean, sd)=%s')%(d, z, (mean, sd)))

    # write in file
    current_directory = os.getcwd()
    final_directory = os.path.join(current_directory, r'output', 'proximity')
    os.makedirs(final_directory, exist_ok=True)  # 如果目录不存在就创建


    output_path = os.path.join(final_directory, filename)
    print(f"正在写入输出文件: {output_path}")
        
    # 使用 utf-8 编码写入文件
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(f"{drug_key}\t{disease_save}\t{d}\t{mean}\t{sd}\t{z}\t{pval}\n")

def get_single_file_from_dir(directory):
    """
    从指定目录获取单个文件
    """
    files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
    if len(files) == 1:
        return os.path.join(directory, files[0])
    else:
        raise Exception(f"目录 {directory} 中应该只包含一个文件，但找到了 {len(files)} 个文件")

def get_all_files_from_dir(directory):
    """
    从指定目录获取所有文件
    """
    return [os.path.join(directory, f) for f in os.listdir(directory) 
            if os.path.isfile(os.path.join(directory, f))]

def run_proximity_analysis(nsims=500):
    """
    运行药物-疾病近邻性分析
    
    参数:
    nsims: int, 随机化次数，默认1000
    """
    # 定义基础路径
    base_path = r'D:\HerbAgent\data\PPI_test_data'
    
    # 定义各个目录
    drug_dir = os.path.join(base_path, 'Drug_targets')
    herb_dir = os.path.join(base_path, 'Herb_targets')
    disease_dir = os.path.join(base_path, 'Disease_targets')
    syndrome_dir = os.path.join(base_path, 'Syndrome_targets')
    
    # 定义结果目录
    results_base = os.path.join(base_path, 'PPI_results')
    disease_drug_dir = os.path.join(results_base, 'disease_drug')
    disease_herb_dir = os.path.join(results_base, 'disease_herb')
    syndrome_herb_dir = os.path.join(results_base, 'syndrome_herb')
    
    # 创建结果目录
    for dir_path in [disease_drug_dir, disease_herb_dir, syndrome_herb_dir]:
        os.makedirs(dir_path, exist_ok=True)
    
    # 获取各类文件
    drug_file = get_single_file_from_dir(drug_dir)
    herb_file = get_single_file_from_dir(herb_dir)
    disease_file = get_single_file_from_dir(disease_dir)
    syndrome_files = get_all_files_from_dir(syndrome_dir)
    
    # 固定网络文件路径
    network_file1 = os.path.join(base_path, 'PPI.csv')
    
    # 读取网络
    G = tools.read_network(network_file1)
    components = nx.connected_components(G)
    lcclist = sorted(list(components), key=len, reverse=True)
    G1 = nx.subgraph(G, lcclist[0])
    
    # 处理三种情况
    # 1. Disease + Drug
    disease_genes = pd.read_csv(disease_file)
    disease_gene_set = set(disease_genes['EntrezID'].astype(str))
    drug_targets = parse_drug_target(drug_file)
    
    # 修改 single_proximity 函数的输出路径
    original_output_dir = os.path.join(os.getcwd(), r'output', 'proximity')
    os.makedirs(disease_drug_dir, exist_ok=True)
    samples = []
    for i in drug_targets.keys():
        if len(drug_targets[i]) >= 2:
            samples.append([i, disease_file, drug_targets, 
                          {disease_file: disease_gene_set}, G1, nsims])
    
    # 设置并行计算
    ncpus = multiprocessing.cpu_count() - 2
    with Pool(ncpus) as p:
        res = list(tqdm(p.imap(single_proximity, samples), total=len(samples)))
    
    # 2. Disease + Herb
    herb_targets = parse_drug_target(herb_file)
    os.makedirs(disease_herb_dir, exist_ok=True)
    samples = []
    for i in herb_targets.keys():
        if len(herb_targets[i]) >= 2:
            samples.append([i, disease_file, herb_targets, 
                          {disease_file: disease_gene_set}, G1, nsims])
    
    with Pool(ncpus) as p:
        res = list(tqdm(p.imap(single_proximity, samples), total=len(samples)))
    
    # 3. Syndrome + Herb
    os.makedirs(syndrome_herb_dir, exist_ok=True)
    for syndrome_file in syndrome_files:
        syndrome_genes = pd.read_csv(syndrome_file)
        syndrome_gene_set = set(syndrome_genes['EntrezID'].astype(str))
        
        samples = []
        for i in herb_targets.keys():
            if len(herb_targets[i]) >= 2:
                samples.append([i, syndrome_file, herb_targets, 
                              {syndrome_file: syndrome_gene_set}, G1, nsims])
        
        with Pool(ncpus) as p:
            res = list(tqdm(p.imap(single_proximity, samples), total=len(samples)))
    
    return True

if __name__ == "__main__":
    run_proximity_analysis()


# from PPI.Proximity import run_proximity_analysis

# # 调用分析函数
# drug_file = "path/to/drug_targets.txt"
# disease_file = "path/to/disease_genes.csv"
# result = run_proximity_analysis(drug_file, disease_file, nsims=1000)
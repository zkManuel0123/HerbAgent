import pandas as pd
import os
from typing import List, Dict, Union
from target_downloader_forloop import download_syndrome_targets
from Entrez_ID import convert_symbols_to_entrez

class SyndromeTargetSearch:
    def __init__(self):
        self.syndrome_map_path = r"D:\HerbAgent\data\SymMap_syndrome_name.xlsx"
        self.symmap_all_path = r"D:\HerbAgent\data\SymMap_all.xlsx"
        self.gene_symbol_path = r"D:\HerbAgent\data\symmap_genesymbol.csv"
        self.output_dir = r"D:\HerbAgent\data\PPI_test_data\Syndrome_targets"
        
        # 读取所需的数据表
        self.syndrome_map_df = pd.read_excel(self.syndrome_map_path)
        self.symmap_all_df = pd.read_excel(self.symmap_all_path)
        self.gene_symbol_df = pd.read_csv(self.gene_symbol_path)

    def save_syndrome_targets(self, result_dict: Dict[str, List[int]]) -> None:
        """
        将每个证候的Entrez ID列表保存为单独的CSV文件，使用syndrome_id作为文件名
        
        Args:
            result_dict: 字典，键为syndrome_id，值为Entrez ID列表
        """
        # 确保输出目录存在
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 直接使用syndrome_id作为文件名
        for syndrome_id, entrez_ids in result_dict.items():
            file_name = f"syndrome_{syndrome_id}.csv"
            file_path = os.path.join(self.output_dir, file_name)
            
            # 创建DataFrame并保存
            df = pd.DataFrame(entrez_ids, columns=['EntrezID'])
            df.to_csv(file_path, index=False)
            print(f"已保存证候 {syndrome_id} 的靶点到文件: {file_name}")
    
    def find_syndrome_ids(self, syndrome_names: List[str]) -> List[str]:
        """
        查找给定证候名称对应的syndrome_id
        
        Args:
            syndrome_names: 中医证候名称列表(中文或英文)
        
        Returns:
            对应的syndrome_id列表
        """
        found_ids = []
        for name in syndrome_names:
            # 在中文名和英文名中都查找
            mask = (self.syndrome_map_df['Syndrome_name'] == name) | \
                  (self.syndrome_map_df['Syndrome_English'] == name)
            matches = self.syndrome_map_df[mask]['Syndrome_id'].tolist()
            
            if matches:
                found_ids.extend([str(id) for id in matches])  # 不再填充为3位数
            else:
                print(f"Warning: No syndrome ID found for '{name}'")
        
        return list(set(found_ids))  # 去重
    
    def get_gene_symbols(self, syndrome_id: str) -> List[str]:
        """
        通过syndrome_id获取对应的gene symbols并保存中间结果
        
        Args:
            syndrome_id: 证候ID
        
        Returns:
            对应的gene symbols列表
        """
        # 从SymMap_all.xlsx中获取Gene_id
        gene_ids = self.symmap_all_df[self.symmap_all_df['Syndrome_id'] == int(syndrome_id)]['Gene_id'].tolist()
        
        # 从symmap_genesymbol.csv中获取Gene_symbol
        gene_symbols = self.gene_symbol_df[self.gene_symbol_df['Gene_id'].isin(gene_ids)]['Gene_symbol'].tolist()
        
        # 添加保存中间结果的代码
        output_dir = r"D:\HerbAgent\Random_Walk\randomwalk_data\syndrome"
        os.makedirs(output_dir, exist_ok=True)
        
        # 直接使用syndrome_id作为文件名
        output_file = os.path.join(output_dir, f'{syndrome_id}.csv')
        
        # 创建包含Syndrome_id和SYMBOL的DataFrame
        result_df = pd.DataFrame({
            'Syndrome_id': [syndrome_id] * len(gene_symbols),
            'SYMBOL': gene_symbols
        })
        
        # 保存为CSV文件
        result_df.to_csv(output_file, index=False)
        
        return gene_symbols
    
    def get_targets_for_syndromes(self, syndrome_names: List[str]) -> Dict[str, List[int]]:
        """
        获取给定证候名称对应��所有基因靶点，并转换为Entrez ID
        
        Args:
            syndrome_names: 中医证候名称列表(中文或英文)
        
        Returns:
            字典 {syndrome_id: [entrez_ids]}
        """
        # 1. 查找syndrome_ids
        syndrome_ids = self.find_syndrome_ids(syndrome_names)
        if not syndrome_ids:
            return {}
            
        # 2. 获取每个syndrome对应的gene symbols
        result = {}
        for sid in syndrome_ids:
            gene_symbols = self.get_gene_symbols(sid)
            result[sid] = gene_symbols
        
        # 3. 转换为Entrez ID
        result_entrez = convert_symbols_to_entrez(result)
        
        # 4. 保存结果到CSV文件
        self.save_syndrome_targets(result_entrez)
                
        return result_entrez

def search_targets(syndrome_names: List[str]) -> Dict[str, List[int]]:
    """
    主函数：查找证候对应的基因靶点
    
    Args:
        syndrome_names: 中医证候名称列表(中文或英文)
    
    Returns:
        字典 {syndrome_id: [entrez_ids]}
    """
    searcher = SyndromeTargetSearch()
    return searcher.get_targets_for_syndromes(syndrome_names)

if __name__ == "__main__":
    # 使用示例
    # test_syndromes = ["亡阳", "外感"]
    test_syndromes = ["yang exhaustion", "externally infected disease"]
    results = search_targets(test_syndromes)
    
    # 打印结果统计信息
    for sid, targets in results.items():
        print(f"\nSyndrome ID: {sid}")
        print(f"Number of Entrez IDs: {len(targets)}")
        print(f"First 5 Entrez IDs: {targets[:5] if targets else 'No targets found'}")
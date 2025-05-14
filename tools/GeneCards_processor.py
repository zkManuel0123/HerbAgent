import pandas as pd
import os
import glob
from typing import List, Dict, Union

class GeneCardsProcessor:
    def __init__(self, data_folder: str = r"D:\HerbAgent\GeneCards_data"):
        """
        初始化处理器
        
        Args:
            data_folder: GeneCards数据文件所在的文件夹路径
        """
        self.data_folder = data_folder
        self.gene_symbols_list = []
        
    def get_csv_files(self) -> List[str]:
        """
        获取数据文件夹中所有的CSV文件
        
        Returns:
            List[str]: CSV文件路径列表
        """
        csv_files = glob.glob(os.path.join(self.data_folder, "*.csv"))
        if not csv_files:
            print(f"警告：在 {self.data_folder} 目录下没有找到CSV文件")
        else:
            print(f"找到 {len(csv_files)} 个CSV文件")
        return csv_files
    
    def process_file(self, file_path: Union[str, bytes]) -> List[str]:
        """
        处理单个CSV文件
        
        Args:
            file_path: 可以是文件路径字符串或文件对象
            
        Returns:
            List[str]: 提取的基因符号列表
        """
        try:
            df = pd.read_csv(file_path, usecols=[0], skiprows=[0])
            gene_symbols = df.iloc[:, 0].tolist()
            gene_symbols = [str(gene).strip() for gene in gene_symbols if pd.notna(gene)]
            gene_symbols = [gene for gene in gene_symbols if gene]
            return gene_symbols
            
        except Exception as e:
            print(f"处理文件时出错: {str(e)}")
            return []
    
    def process_all_files(self) -> List[str]:
        """
        处理文件夹中的所有CSV文件
        
        Returns:
            List[str]: 所有文件中提取的唯一基因符号列表
        """
        csv_files = self.get_csv_files()
        for file in csv_files:
            symbols = self.process_file(file)
            self.gene_symbols_list.extend(symbols)
        
        # 去重处理
        unique_genes = list(set(self.gene_symbols_list))
        # print(f"总共提取了 {len(unique_genes)} 个唯一的基因符号")
        return unique_genes

# 示例使用
if __name__ == "__main__":
    processor = GeneCardsProcessor()
    result = processor.process_all_files()
    print(result[:10])

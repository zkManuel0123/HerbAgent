import pandas as pd
from typing import List, Dict
from collections import defaultdict
from Entrez_ID import convert_symbols_to_entrez
import os

class HerbTargetSearch:
    def __init__(self):
        # 读取所有必需的数据文件
        self.herb_df = pd.read_excel(r"D:\HerbAgent\data\Herb_name.xlsx")
        self.ingredient_df = pd.read_excel(r"D:\HerbAgent\data\Herb_ingredient.xlsx")
        self.target_df = pd.read_excel(r"D:\HerbAgent\data\ingredient_target.xlsx")
        
        # 确保列名正确
        self.herb_df.columns = ['Herb_id', 'Herb_zh_name', 'Herb_en_name']
        self.ingredient_df.columns = ['Herb_id'] + [f'Col_{i}' for i in range(2, 4)] + ['Ingredient_id'] + [f'Col_{i}' for i in range(5, 7)]
        self.target_df.columns = ['Ingredient_id'] + [f'Col_{i}' for i in range(2, 4)] + ['Target_name'] + [f'Col_{i}' for i in range(5, 7)]

    def find_herb_ids(self, herb_names: List[str]) -> List[str]:
        """根据中药名称（中文或英文）查找对应的Herb_id"""
        herb_ids = []
        for name in herb_names:
            mask = (self.herb_df['Herb_zh_name'] == name) | (self.herb_df['Herb_en_name'] == name)
            found_ids = self.herb_df[mask]['Herb_id'].tolist()
            herb_ids.extend(found_ids)
        return herb_ids

    def find_ingredient_ids(self, herb_ids: List[str]) -> List[str]:
        """根据Herb_id查找对应的所有Ingredient_id"""
        ingredient_ids = []
        for herb_id in herb_ids:
            ingredients = self.ingredient_df[self.ingredient_df['Herb_id'] == herb_id]['Ingredient_id'].tolist()
            ingredient_ids.extend(ingredients)
        return ingredient_ids

    def find_targets(self, ingredient_ids: List[str]) -> List[str]:
        """根据Ingredient_id查找对应的Target_name"""
        targets = []
        for ingredient_id in ingredient_ids:
            found_targets = self.target_df[self.target_df['Ingredient_id'] == ingredient_id]['Target_name'].tolist()
            targets.extend(found_targets)
        return list(set(targets))  # 去重

    def save_results_to_file(self, formula_entrez: Dict[str, List[int]], output_path: str):
        """
        将结果保存为制表符分隔的txt文件
        
        Args:
            formula_entrez: 字典，键为方剂名，值为Entrez ID列表
            output_path: 输出文件路径
        """
        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # 写入文件
        with open(output_path, 'w', encoding='utf-8') as f:
            for formula_name, entrez_ids in formula_entrez.items():
                # 将方剂名和所有Entrez ID用制表符连接
                line = formula_name + '\t' + '\t'.join(map(str, entrez_ids))
                f.write(line + '\n')

    def search(self, formula_herbs: Dict[str, List[str]]) -> Dict[str, List[int]]:
        """
        主要搜索函数，处理多个方剂的中药列表，并将基因符号转换为Entrez ID
        
        Args:
            formula_herbs: 字典，键为方剂名，值为该方剂包含的中药名列表
            
        Returns:
            字典，键为方剂名，值为该方剂对应的所有Entrez ID列表
        """
        formula_targets = {}
        
        # 创建保存中间结果的目录
        herb_ingredient_dir = r'D:\HerbAgent\Random_Walk\randomwalk_data\formula_herb_ingredient'
        ingredient_target_dir = r'D:\HerbAgent\Random_Walk\randomwalk_data\formula_ingredient_target'
        os.makedirs(herb_ingredient_dir, exist_ok=True)
        os.makedirs(ingredient_target_dir, exist_ok=True)
        
        for formula_name, herb_names in formula_herbs.items():
            herb_ids = self.find_herb_ids(herb_names)
            ingredient_ids = self.find_ingredient_ids(herb_ids)
            
            # 保存herb-ingredient对应关系
            herb_ingredient_data = []
            for herb_id in herb_ids:
                ingredients = self.ingredient_df[self.ingredient_df['Herb_id'] == herb_id]['Ingredient_id'].tolist()
                for ingredient_id in ingredients:
                    herb_ingredient_data.append({'Herb_id': herb_id, 'Ingredient_id': ingredient_id})
            
            herb_ingredient_df = pd.DataFrame(herb_ingredient_data)
            # 使用formula_name作为文件名前缀
            herb_ingredient_path = os.path.join(herb_ingredient_dir, f'{formula_name}_herb_ingredient.csv')
            herb_ingredient_df.to_csv(herb_ingredient_path, index=False)
            
            # 保存ingredient-target对应关系
            ingredient_target_data = []
            targets = []
            for ingredient_id in ingredient_ids:
                found_targets = self.target_df[self.target_df['Ingredient_id'] == ingredient_id]['Target_name'].tolist()
                for target in found_targets:
                    ingredient_target_data.append({'Ingredient_id': ingredient_id, 'Target_name': target})
                    targets.append(target)
            
            ingredient_target_df = pd.DataFrame(ingredient_target_data)
            # 使用formula_name作为文件名前缀
            ingredient_target_path = os.path.join(ingredient_target_dir, f'{formula_name}_ingredient_target.csv')
            ingredient_target_df.to_csv(ingredient_target_path, index=False)
            
            # 预处理靶点名称，移除非标准基因符号
            cleaned_targets = []
            for target in targets:
                if not target.startswith(('SymMap:', 'TCMSP:')):
                    cleaned_targets.append(target)
            
            formula_targets[formula_name] = list(set(cleaned_targets))  # 去重
        
        # 将基因符号转换为Entrez ID
        formula_entrez = convert_symbols_to_entrez(formula_targets)
        
        # 保存结果到文件
        output_path = r'D:\HerbAgent\data\PPI_test_data\Herb_targets\herb.txt'
        self.save_results_to_file(formula_entrez, output_path)
        
        return formula_entrez

def main():
    # 使用示例
    searcher = HerbTargetSearch()
    
    # 测试用例
    formula_herbs = {
        "补血汤": ["当归", "白芍"],
        "示例方剂2": ["当归", "白芍", "熟地黄"]
    }
    
    results = searcher.search(formula_herbs)
    print("结果已保存到: D:\\HerbAgent\\data\\PPI_test_data\\Herb_targets\\herb.txt")
    
    # 打印结果统计
    for formula_name, entrez_ids in results.items():
        print(f"\n方剂名称: {formula_name}")
        print(f"靶点数量: {len(entrez_ids)}")
        print(f"前5个Entrez ID: {entrez_ids[:5] if entrez_ids else '未找到靶点'}")

if __name__ == "__main__":
    main()
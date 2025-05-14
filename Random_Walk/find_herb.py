#对随机游走的计算结果（结果只有化学成分的得分），计算前10的核心化学成分，通过herb_ingredient.csv查询ingredient找到对应的herbs，并计算每个herbs的得分
import pandas as pd
from pathlib import Path

class HerbFinder:
    def __init__(self):
        # 定义基础路径
        self.base_path = Path("D:/HerbAgent/Random_Walk")
        self.results_path = self.base_path / "results"
        self.herb_ingredient_path = self.base_path / "randomwalk_data/formula_herb_ingredient"
        self.output_path = self.base_path / "herb_top_results"
        
        # 确保输出目录存在
        self.output_path.mkdir(parents=True, exist_ok=True)

    def get_matching_files(self):
        """获取所有匹配的文件对"""
        matching_pairs = []
        
        # 遍历results目录下的所有文件
        for result_file in self.results_path.glob("*_ingredient_score.csv"):
            # 获取方剂名称（去掉后缀）
            formula_name = result_file.stem.split('_')[0]
            
            # 构建对应的herb_ingredient文件路径
            herb_ingredient_file = self.herb_ingredient_path / f"{formula_name}_herb_ingredient.csv"
            
            # 如果两个文件都存在，添加到匹配列表
            if herb_ingredient_file.exists():
                matching_pairs.append({
                    'formula_name': formula_name,
                    'result_file': result_file,
                    'herb_ingredient_file': herb_ingredient_file
                })
        
        return matching_pairs

    def process_single_formula(self, formula_name: str, result_file: Path, herb_ingredient_file: Path):
        """处理单个方剂的数据"""
        try:
            print(f"\n开始处理方剂: {formula_name}")
            
            # 读取数据
            results = pd.read_csv(result_file)
            herb_ingredient = pd.read_csv(herb_ingredient_file)

            # 获取top 10的ingredients
            top_ingredients = results.head(10)

            # 找到这些ingredients对应的herbs
            herbs_data = herb_ingredient[herb_ingredient['Ingredient_id'].isin(top_ingredients['Node'])]

            # 为每个ingredient添加其score
            herbs_data = herbs_data.merge(top_ingredients, 
                                        left_on='Ingredient_id', 
                                        right_on='Node', 
                                        how='left')

            # 计算每个herb的总分和统计信息
            herb_scores = herbs_data.groupby('Herb_id').agg({
                'Score': ['sum', 'count'],
                'Ingredient_id': lambda x: ', '.join(x)
            }).reset_index()

            # 整理列名
            herb_scores.columns = ['Herb_id', 'Total_Score', 'Top_Ingredient_Count', 'Top_Ingredients']

            # 按总分排序
            herb_scores = herb_scores.sort_values('Total_Score', ascending=False)

            # 保存结果
            output_file = self.output_path / f"{formula_name}_herb_score.csv"
            herb_scores.to_csv(output_file, index=False)
            
            # 打印详细信息
            print(f"\n{formula_name} 的排名前10的Ingredients对应的Herbs分析结果：")
            print(herb_scores.to_string(index=False))
            print(f"\n结果已保存到：{output_file}")

            # 打印详细的对应关系
            print(f"\n{formula_name} 每个Herb包含的Top Ingredients及其得分：")
            for _, herb_row in herb_scores.iterrows():
                print(f"\nHerb {herb_row['Herb_id']}:")
                herb_ingredients = herbs_data[herbs_data['Herb_id'] == herb_row['Herb_id']]
                for _, ing in herb_ingredients.iterrows():
                    print(f"  Ingredient {ing['Ingredient_id']}: {ing['Score']:.6f}")

            return True

        except Exception as e:
            print(f"处理方剂 {formula_name} 时出错: {str(e)}")
            return False

    def process_all_formulas(self):
        """处理所有匹配的方剂文件"""
        matching_pairs = self.get_matching_files()
        
        if not matching_pairs:
            print("未找到匹配的文件对")
            return
        
        for pair in matching_pairs:
            self.process_single_formula(
                pair['formula_name'],
                pair['result_file'],
                pair['herb_ingredient_file']
            )

def run_herb_analysis():
    """供外部模块调用的主函数"""
    finder = HerbFinder()
    finder.process_all_formulas()

if __name__ == "__main__":
    run_herb_analysis()


# from Random_Walk.find_herb import run_herb_analysis

# # 运行分析
# run_herb_analysis()


# 现在需要实现find_herb.py能被其他模块调用，并且要能使得读取，比如其中的results，
# 能够遍历读取D:\HerbAgent\Random_Walk\results下面所有的文件，这个目录下会储存所有需要计算完的方剂1、
# 方剂2的ingredient_score文件。而herb_ingredient这个输入，
# 要能依次遍历D:\HerbAgent\Random_Walk\randomwalk_data\formula_herb_ingredient下面的所有文件，
# 这个文件夹下面存储着方剂1、方剂2等等的herb_ingredient对应文件。
# 注意：results和herb_ingredient这两个变量每次遍历各自文件夹下面的文件要对应，
# 只需要按具体文件名查找读取即可，各自对应的文件都会以方剂名作为前缀，
# 比如results需要读取方剂1_ingredient_score.csv，
# 而herb_ingredient就要读取同样前缀的方剂1_herb_ingredient.csv，
# 这样两个匹配的文件才能作为一次有效输入到find_herb.py中。
# 最后每次的运行结果，保存在D:\HerbAgent\Random_Walk\herb_top_results这个目录下，
# 并以方剂名作为前缀命名，比如方剂1_herb_score.csv，方剂2的计算结果保存在同样的目录下，
# 以方剂2_herb_score.csv命名，方剂3等等以此类推。
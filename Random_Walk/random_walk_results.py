from random_walk import run_random_walk_analysis
from find_herb import run_herb_analysis
import pandas as pd
import os
import glob

def summarize_herb_scores():
    """汇总所有草药得分文件并保存，返回汇总的数据框"""
    # 设置文件路径
    result_dir = r"D:\HerbAgent\Random_Walk\herb_top_results"
    output_dir = r"D:\HerbAgent\Random_Walk\herb_top_results\summary_score"
    output_path = os.path.join(output_dir, "summary_score.csv")
    
    # 获取所有符合条件的文件
    file_pattern = os.path.join(result_dir, "*_herb_score.csv")
    score_files = glob.glob(file_pattern)
    
    if not score_files:
        print("未找到任何草药得分文件")
        return None
    
    # 读取并合并所有文件
    all_scores = []
    for file in sorted(score_files):  # 按文件名排序
        try:
            df = pd.read_csv(file, encoding='utf-8')
            file_name = os.path.basename(file)
            analysis_type = file_name.replace("_herb_score.csv", "").encode('utf-8').decode('utf-8')
            df['analysis_type'] = analysis_type
            all_scores.append(df)
            print(f"已读取文件: {file_name}")
        except Exception as e:
            print(f"读取文件 {file} 时出错: {str(e)}")
    
    if not all_scores:
        print("没有成功读取任何文件")
        return None
    
    # 合并所有数据框
    summary_df = pd.concat(all_scores, ignore_index=True)
    
    # 保存汇总结果
    try:
        summary_df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"汇总结果已保存到: {output_path}")
        return summary_df  # 确保返回数据框
    except Exception as e:
        print(f"保存汇总结果时出错: {str(e)}")
        return None

def transform_summary_results(summary_df):
    """转换汇总结果，将ID替换为具体名称"""
    try:
        # 读取映射表
        herb_mapping = pd.read_excel(r"D:\HerbAgent\data\Herb_name.xlsx")
        ingredient_mapping = pd.read_excel(r"D:\HerbAgent\data\Herb_ingredient.xlsx")
        
        # 创建新的数据框
        transformed_df = summary_df.copy()
        
        # 创建herb_id到中文名和英文名的映射字典
        herb_name_dict = herb_mapping.set_index('Herb_id').agg(
            lambda x: f"{x['Herb_zh_name']}（{x['Herb_en_name']}）", axis=1
        ).to_dict()
        
        # 创建ingredient_id到名称的映射字典
        ingredient_dict = ingredient_mapping.set_index('Ingredient_id')['Ingredient_name'].to_dict()
        
        # 转换herb_id为名称组合
        transformed_df['Herb_name'] = transformed_df['Herb_id'].map(herb_name_dict)
        
        # 转换化学成分ID为名称
        def convert_ingredients(ingredient_ids):
            if pd.isna(ingredient_ids):
                return ""
            ingredients = [ingredient_dict.get(i.strip(), i.strip()) 
                         for i in str(ingredient_ids).split(',')]
            return ', '.join(ingredients)
        
        transformed_df['Top_Ingredients'] = transformed_df['Top_Ingredients'].apply(convert_ingredients)
        
        # 重新排列列顺序，将Herb_name放在第一列
        cols = transformed_df.columns.tolist()
        cols.remove('Herb_name')
        transformed_df = transformed_df[['Herb_name'] + cols]
        
        # 保存转换后的结果
        output_path = r"D:\HerbAgent\Random_Walk\herb_top_results\summary_score\final_summary_score\final_summary_score.csv"
        transformed_df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"转换后的结果已保存到: {output_path}")
        
        return transformed_df
        
    except Exception as e:
        print(f"转换过程出错: {str(e)}")
        return None

def run_analysis():
    """执行完整的分析流程并返回结果"""
    results = {
        'success': False,
        'summary_df': None,
        'transformed_df': None,
        'message': ''
    }
    
    try:
        # 执行随机游走分析
        print("开始执行随机游走计算...")
        random_walk_success = run_random_walk_analysis()
        
        if not random_walk_success:
            results['message'] = "随机游走计算失败"
            return results
        
        # 执行草药寻找分析
        print("随机游走计算完成，开始寻找草药...")

        run_herb_analysis()
        
        # 汇总分析结果
        print("\n开始汇总草药得分...")
        summary_df = summarize_herb_scores()
        
        if summary_df is not None:
            print("\n开始转换结果...")
            transformed_df = transform_summary_results(summary_df)
            
            if transformed_df is not None:
                results['success'] = True
                results['summary_df'] = summary_df
                results['transformed_df'] = transformed_df
                results['message'] = "分析完成"
                
                print("\n转换后的结果预览:")
                print(transformed_df.head())
            else:
                results['message'] = "结果转换失败"
        else:
            results['message'] = "汇总分析失败"
            
        return results
        
    except Exception as e:
        results['message'] = f"执行过程出错: {str(e)}"
        return results

def main():
    """主函数用于直接运行时的执行"""
    results = run_analysis()
    if results['success']:
        print("\n分析成功完成！")
        print("\n原始结果预览：")
        print(results['summary_df'].head())
        print("\n转换后结果预览：")
        print(results['transformed_df'].head())
    else:
        print(f"\n分析失败: {results['message']}")

if __name__ == "__main__":
    main()




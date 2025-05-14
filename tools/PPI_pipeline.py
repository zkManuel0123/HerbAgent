import os
import sys

# 获取项目根目录
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 更新系统路径
sys.path.append(PROJECT_ROOT)

# 导入需要的模块
from PPI.Proximity import run_proximity_analysis
from tools.PPI_result_analyze import analyze_proximity_results

def run_ppi_pipeline():
    
    try:
        # 步骤1: 运行proximity分析
        print("开始运行Proximity分析...")
        try:
            run_proximity_analysis()
            print("Proximity分析已完成")
        except Exception as e:
            print(f"Proximity分析失败: {str(e)}")
            return False, None
            
        # 步骤2: 运行结果分析
        print("开始进行结果分析...")
        try:
            result_data, output_path = analyze_proximity_results()
            if result_data is None:
                print("结果分析未返回有效数据")
                return False, None
                
            return True, (result_data, output_path)
            
        except Exception as e:
            print(f"结果分析失败: {str(e)}")
            return False, None
                    
    except Exception as e:
        print(f"执行过程中出现错误: {str(e)}")
        return False, None

if __name__ == "__main__":
    success, results = run_ppi_pipeline()
    if success:
        result_data, output_path = results
        print("\n最终分析结果：")
        print(result_data)  
        

# success, results = run_ppi_pipeline()
# if success:
#     result_data, _ = results  # 使用 _ 忽略 output_path
    

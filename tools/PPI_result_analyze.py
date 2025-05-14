import os
import pandas as pd
import glob

def read_proximity_files(folder_path):
    """
    读取指定文件夹中的所有txt文件并合并数据
    
    Args:
        folder_path: 文件夹路径
        
    Returns:
        DataFrame: 合并后的数据表格
    """
    # 获取文件夹中所有的txt文件
    file_pattern = os.path.join(folder_path, "*.txt")
    file_list = glob.glob(file_pattern)
    
    if not file_list:
        print(f"在 {folder_path} 中没有找到txt文件")
        return None
    
    # 读取第一个文件来获取列名
    with open(file_list[0], 'r', encoding='utf-8') as f:
        header = f.readline().strip().split('\t')
    
    # 创建一个空的DataFrame来存储所有数据
    all_data = []
    
    # 读取每个文件的数据
    for file_path in file_list:
        try:
            # 从文件名中提取文件名（不包含扩展名）作为标识
            file_name = os.path.basename(file_path).replace('.txt', '')
            
            # 读取文件内容
            with open(file_path, 'r', encoding='utf-8') as f:
                data = f.readline().strip().split('\t')
            
            # 确保数据长度与列名长度相匹配
            if len(data) == len(header):
                # 创建一个字典，包含文件名和数据
                row_dict = dict(zip(header, data))
                row_dict['File_Name'] = file_name  # 添加文件名列
                all_data.append(row_dict)
            else:
                print(f"警告：文件 {file_name} 的数据列数与标题列数不匹配")
                
        except Exception as e:
            print(f"处理文件 {file_path} 时出错：{str(e)}")
    
    if not all_data:
        print("没有成功读取任何数据")
        return None
    
    # 将所有数据转换为DataFrame
    df = pd.DataFrame(all_data)
    
    # 重新排列列，使File_Name在最前面
    cols = ['File_Name'] + [col for col in df.columns if col != 'File_Name']
    df = df[cols]
    
    # 只保留需要的列（第1，2，3列和第7，8列）
    df = df.iloc[:, [0, 1, 2, 6, 7]]
    
    # 重命名列
    df.columns = ['file_name', 'drug', 'disease', 'z-value', 'p-value']
    
    # 需要添加数值列的类型转换
    df['z-value'] = pd.to_numeric(df['z-value'], errors='coerce')
    df['p-value'] = pd.to_numeric(df['p-value'], errors='coerce')
    
    return df

def analyze_proximity_results(folder_path=r"D:\HerbAgent\output\proximity"):
    
    result_df = read_proximity_files(folder_path)
    
    if result_df is not None:
        # 打印完整结果到控制台
        
        
           
        # 保存到Excel文件，包含标题行
        output_file = os.path.join(os.path.dirname(folder_path), "proximity_results.xlsx")
        result_df.to_excel(output_file, index=False)
        
        print(f"\n数据已保存到：{output_file}")
        
        
        return result_df, output_file
    
    return None, None

if __name__ == "__main__":
    result_data, output_path = analyze_proximity_results()

# 从csv文件中根据MeSH ID获取疾病的扩展关键词
import pandas as pd
import ast

def extract_keywords_by_mesh_id(file_path, mesh_id):
        # 读取CSV文件
        df = pd.read_csv(file_path)
        
        # 查找对应mesh_id的行
        row = df[df['UniqueID'] == mesh_id]
        
        if row.empty:
            print(f"未找到mesh_id: {mesh_id}")
            return []
        
        # 获取EntryTerms列的值
        entry_terms_str = row['EntryTerms'].iloc[0]
        
       # 将字符串形式的列表转换为实际的Python列表
        keywords = ast.literal_eval(entry_terms_str)
        return keywords
        
    

# 使用示例
file_path = r"D:\HerbAgent\data\mesh_data.csv"
mesh_id = "D013964"  # 替换为实际的mesh_id
keywords = extract_keywords_by_mesh_id(file_path, mesh_id)
print(keywords)

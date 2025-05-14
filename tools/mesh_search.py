# 从xml文件中提取疾病的标准名称和MeSH ID
import xml.etree.ElementTree as ET
import pandas as pd
from rapidfuzz import process, fuzz
from typing import Tuple, Optional

# 加载 XML 文件
def parse_mesh_xml():
    file_path = r"D:\HerbAgent\data\desc2024.xml"
    tree = ET.parse(file_path)
    root = tree.getroot()
    
    # 创建列表存储数据
    descriptor_names = []
    unique_ids = []
    entry_terms_list = []
    
    # 遍历每个 DescriptorRecord
    for record in root.findall("DescriptorRecord"):
        descriptor_name = record.find("DescriptorName/String").text
        unique_id = record.find("DescriptorUI").text
        entry_terms = [entry.text for entry in record.findall("ConceptList/Concept/TermList/Term/String")]
        
        # 添加到对应的列表
        descriptor_names.append(descriptor_name)
        unique_ids.append(unique_id)
        entry_terms_list.append(entry_terms)
    
    # 创建 DataFrame
    mesh_df = pd.DataFrame({
        'DescriptorName': descriptor_names,
        'UniqueID': unique_ids,
        'EntryTerms': entry_terms_list
    })
    
    return mesh_df

def find_disease_mesh(query: str, mesh_df: pd.DataFrame, score_cutoff: float = 80) -> Optional[Tuple[str, str, float]]:
    """
    查找疾病的标准名称和 MeSH ID
    
    Args:
        query: 用户输入的疾病名称
        mesh_df: MeSH 数据框
        score_cutoff: 相似度分数阈值
    
    Returns:
        Tuple(标准名称, MeSH ID, 相似度分数) 或 None
    """
    # 创建用于匹配的扩展 DataFrame
    expanded_terms = []
    
    for idx, row in mesh_df.iterrows():
        # 添加标准名称
        expanded_terms.append((row['DescriptorName'], row['UniqueID'], row['DescriptorName']))
        # 添加同义词
        for term in row['EntryTerms']:
            expanded_terms.append((row['DescriptorName'], row['UniqueID'], term))
        
# [示例数据]
#     ('Diabetes', 'D001234', 'Diabetes'),
#     ('Diabetes', 'D001234', 'High blood sugar'),
#     ('Diabetes', 'D001234', 'DM'),
#     ('Hypertension', 'D005678', 'Hypertension'),
#     ('Hypertension', 'D005678', 'High blood pressure')
# ]
    # 创建扩展后的 DataFrame
    expanded_df = pd.DataFrame(expanded_terms, columns=['StandardName', 'MeshID', 'SearchTerm'])
    
    # 使用 rapidfuzz 进行匹配，返回只包含一个最匹配结果
    matches = process.extractOne(
        query,
        expanded_df['SearchTerm'].tolist(),
        scorer=fuzz.WRatio,
        score_cutoff=score_cutoff
    )

    if matches is None:
        return None
    
    matched_term, score = matches[:2]  #元组解包 matched_term = "High blood sugar"  score = 85
    result_row = expanded_df[expanded_df['SearchTerm'] == matched_term].iloc[0]  #从 expanded_df 数据框中找到与 matched_term 相等的行，并取出第一行作为结果。
    
    return (result_row['StandardName'], result_row['MeshID'], score) #提取需要的字段，并返回一个元组




# 测试示例
if __name__ == "__main__":
    # 加载 MeSH 数据
    mesh_df = parse_mesh_xml()
    
    while True:
        query = input("请输入疾病名称（输入 'q' 退出）: ")
        if query.lower() == 'q':
            break
        # test_disease_search(query)

        # 在这里直接调用 find_disease_mesh 测试
        result = find_disease_mesh(query, mesh_df)
        if result:
            standard_name, mesh_id, score = result
            print(f"输入: {query}")
            print(f"标准名称: {standard_name}")
            print(f"MeSH ID: {mesh_id}")
            print(f"匹配分数: {score:.2f}")
        else:
            print(f"No matching disease found for '{query}' in MeSH database.")

        
    


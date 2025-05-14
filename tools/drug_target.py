import pandas as pd
import os
from Entrez_ID import convert_symbols_to_entrez

def load_data(drug_file, target_file):
    """
    加载药物和靶点数据文件
    """
    try:
        drug_df = pd.read_csv(drug_file, encoding='iso-8859-1')
        target_df = pd.read_csv(target_file, encoding='iso-8859-1')
        return drug_df, target_df
    except Exception as e:
        print(f"Error loading data files: {e}")
        return None, None

def search_drugs_by_diseases(drug_df, disease_keywords):
    """
    通过疾病关键词在indication列中搜索匹配的药物
    """
    matched_drugs = []
    disease_keywords = [keyword.lower() for keyword in disease_keywords]
    
    for idx, row in drug_df.iterrows():
        indication = str(row['indication']).lower()
        if any(keyword in indication for keyword in disease_keywords):
            matched_drugs.append({
                'drugbank_id': row['drugbank_id'],
                'name': row.get('name', 'N/A'),
                'indication': row['indication']
            })
    
    return matched_drugs

def get_drug_targets(target_df, drug_ids):
    """
    通过药物ID获取对应的基因靶点
    返回一个字典，key为药物ID，value为该药物的所有基因靶点列表
    """
    drug_targets = {}
    
    for drug_id in drug_ids:
        drug_data = target_df[target_df['drugbank_id'] == drug_id]
        targets = []
        for genes in drug_data['gene_name']:
            if pd.notna(genes):
                targets.extend(genes.split('|'))
        drug_targets[drug_id] = list(set(targets))
    
    return drug_targets

def save_drug_targets_to_file(drug_targets_dict: dict, output_path: str):
    """
    将药物靶点保存为制表符分隔的文本文件
    
    Args:
        drug_targets_dict: 字典，键为药物ID，值为Entrez ID列表
        output_path: 输出文件路径
    """
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 将结果写入文件
    with open(output_path, 'w') as f:
        for drug_id, targets in drug_targets_dict.items():
            # 将所有元素转换为字符串并用制表符连接
            line = drug_id + '\t' + '\t'.join(map(str, targets))
            f.write(line + '\n')

def search_disease_drug_targets(disease_keywords, drug_file_path='D:\\HerbAgent\\data\\drugbank.csv', 
                              target_file_path='D:\\HerbAgent\\data\\drugbank_target.csv'):
    """
    主函数：整合所有功能
    """
    # 加载数据
    drug_df, target_df = load_data(drug_file_path, target_file_path)
    if drug_df is None or target_df is None:
        return None
    
    # 搜索匹配的药物
    matched_drugs = search_drugs_by_diseases(drug_df, disease_keywords)
    
    if not matched_drugs:
        print("No matching drugs found for the given disease keywords.")
        return None
    
    # 获取药物ID列表
    drug_ids = [drug['drugbank_id'] for drug in matched_drugs]
    
    # 获取药物靶点
    drug_targets = get_drug_targets(target_df, drug_ids)
    
    # 转换为Entrez ID
    drug_targets_entrez = convert_symbols_to_entrez(drug_targets)
    
    # 保存结果到文件
    output_path = r'D:\HerbAgent\data\PPI_test_data\Drug_targets\drug.txt'
    save_drug_targets_to_file(drug_targets_entrez, output_path)
    
    return drug_targets_entrez

# 使用示例
if __name__ == "__main__":
    disease_keywords = ["Thyroid Neoplasms", "Thyroid Cancer", "Thyroid carcinoma"]
    # 目前不支持中文疾病名称
    results = search_disease_drug_targets(disease_keywords)
    print("Results have been saved to drug.txt")
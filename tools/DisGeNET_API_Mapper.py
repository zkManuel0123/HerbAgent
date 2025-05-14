import requests
import json
import urllib3

def get_disease_gene_targets(disease_cuis):
    """
    获取疾病对应的基因靶点
    
    Args:
        disease_cuis (list): 疾病CUI列表，例如 ["UMLS_C0028754", "UMLS_C0011849"]
    
    Returns:
        dict: 疾病CUI对应的基因名字字典，格式为 {"UMLS_XXX": ["gene1", "gene2", ...]}
    """
    
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    
    # API配置
    API_KEY = "a6332b0f-7e41-44f8-a1db-9813dc18e586"
    
    # 设置请求参数和头部
    params = {}
    HTTPheadersDict = {
        'Authorization': API_KEY,
        'accept': 'application/json'
    }
    
    # 初始化结果字典
    disease_gene_map = {}
    
    # 遍历每个疾病CUI
    for disease_cui in disease_cuis:
        # 更新查询参数
        params["disease"] = disease_cui
        
        # 初始化基因列表
        gene_targets = []
        
        # 获取总页数
        response = requests.get("https://api.disgenet.com/api/v1/gda/summary",\
                                params=params, headers=HTTPheadersDict, verify=False)
        response_parsed = json.loads(response.text)
        total_elements = response_parsed["paging"]["totalElements"]
        page_size = 100
        total_pages = (total_elements + page_size - 1) // page_size
        
        # 遍历所有页
        for page in range(total_pages):
            params["page_number"] = str(page)
            response = requests.get("https://api.disgenet.com/api/v1/gda/summary",\
                                    params=params, headers=HTTPheadersDict, verify=False)
            response_parsed = json.loads(response.text)
            
            # 提取基因名字
            if "payload" in response_parsed:
                for item in response_parsed["payload"]:
                    gene_targets.append(item["symbolOfGene"])
        
        # 存储结果
        disease_gene_map[disease_cui] = list(set(gene_targets))  # 使用set去重
    
    return disease_gene_map

# 示例使用
if __name__ == "__main__":
    # 测试用的疾病CUI列表
    test_cuis = ["UMLS_C0028754", "UMLS_C0011849"]
    
    # 获取结果
    result = get_disease_gene_targets(test_cuis)
    
    # 打印结果
    for disease_cui, genes in result.items():
        print(f"\n疾病 {disease_cui} 的基因靶点:")
        for gene in genes:
            print(f"Gene Symbol: {gene}")



# 使用字典格式返回结果，方便其他程序调用   

# 调用示例：
# from test2 import get_disease_gene_targets
# # 准备疾病CUI列表
# disease_cuis = ["UMLS_C0028754", "UMLS_C0011849"]

# # 获取疾病-基因对应关系
# disease_gene_map = get_disease_gene_targets(disease_cuis)

# # 使用结果
# for disease, genes in disease_gene_map.items():
#     # 进行后续处理
#     pass
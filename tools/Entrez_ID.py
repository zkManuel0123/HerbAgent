import mygene
import pandas as pd

def convert_symbols_to_entrez(input_dict):
    """
    将包含基因符号的字典转换为包含Entrez ID的字典
    input_dict格式: {'ID1': ['gene1', 'gene2'], 'ID2': ['gene3', 'gene4']}
    """
    # 初始化 MyGeneInfo 实例
    mg = mygene.MyGeneInfo()
    
    # 创建结果字典
    result_dict = {}
    
    # 遍历输入字典
    for key, gene_symbols in input_dict.items():
        # 批量查询当前列表中的基因符号
        result = mg.querymany(gene_symbols, scopes="symbol", fields="entrezgene", species="human")
        
        # 提取有效的Entrez ID
        entrez_ids = []
        for item in result:
            if 'entrezgene' in item and item['entrezgene'] is not None:
                entrez_ids.append(int(item['entrezgene']))
            
        result_dict[key] = entrez_ids
    
    return result_dict

# 使用示例
if __name__ == "__main__":
    # 示例输入字典
    # test_dict = {
    #     "Drug1": ["TP53", "BRCA1", "EGFR"],
    #     "Drug2": ["MYC", "KRAS", "PTEN"]
    # }

    test_dict = {
        "Drug1": ["GHR", "RBX1", "ADH4"]
        
    }
    
    # 转换并打印结果
    result = convert_symbols_to_entrez(test_dict)
    print(result)

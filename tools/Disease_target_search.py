import json
from typing import List, Dict, Set, Optional, Union
import sys
import os
import hashlib
import requests
import random
from collections import defaultdict
from dotenv import load_dotenv
import pandas as pd
from Entrez_ID import convert_symbols_to_entrez
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.UMLS_API_Mapper import UMLSMapper
from tools.DisGeNET_API_Mapper import get_disease_gene_targets
from tools.GeneCards_processor import GeneCardsProcessor  

load_dotenv()

class DiseaseTargetSearch:
    def __init__(self):
        self.umls_api_key = os.getenv('UMLS_API_KEY')
        self.baidu_appid = os.getenv('BAIDU_APP_ID')
        self.baidu_secretKey = os.getenv('BAIDU_SECRET_KEY')
        self.umls_mapper = UMLSMapper(self.umls_api_key)


    def baidu_translate(self, query: str) -> str:
        """
        使用百度翻译API进行翻译
        """
        url = "https://fanyi-api.baidu.com/api/trans/vip/translate"
        salt = random.randint(32768, 65536)
        sign = hashlib.md5(f"{self.baidu_appid}{query}{salt}{self.baidu_secretKey}".encode()).hexdigest()
        
        params = {
            'q': query,
            'from': 'zh',
            'to': 'en',
            'appid': self.baidu_appid,
            'salt': salt,
            'sign': sign
        }
        
        try:
            response = requests.get(url, params=params)
            result = response.json()
            if 'trans_result' in result:
                return result['trans_result'][0]['dst']
            else:
                print(f"翻译错误: {result}")
                return query
        except Exception as e:
            print(f"翻译请求错误: {str(e)}")
            return query
        
    def translate_chinese_disease(self, disease_name: str) -> str:
        """
        将中文疾病名称翻译为英文
        """
        try:
            # 检查是否包含中文字符
            if any('\u4e00' <= char <= '\u9fff' for char in disease_name):
                translated = self.baidu_translate(disease_name)
                return translated
            return disease_name
        except Exception as e:
            print(f"Translate error: {disease_name}: {str(e)}")
            return disease_name

    def preprocess_disease_list(self, disease_list: List[str]) -> List[str]:
        """
        预处理疾病列表：
        1. 翻译中文
        2. 标��化大小写（将每个单词首字母大写）
        3. 去重
        """
        # 先进行翻译
        translated_diseases = [self.translate_chinese_disease(disease) for disease in disease_list]
        
        # 标准化大小写：将每个单词首字母大写
        normalized_diseases = []
        for disease in translated_diseases:
            # 先将整个字符串转为小写，然后将每个单词首字母大写
            normalized = disease.lower().title()
            normalized_diseases.append(normalized)
            # print(f"标准化: {disease} -> {normalized}")  # 添加调试信息
        
        # 去重并返回
        return list(set(normalized_diseases))

    def get_umls_cuis(self, disease_list: List[str]) -> List[str]:
        """
        获取疾病的UMLS CUI编码
        """
        results = self.umls_mapper.batch_process(disease_list)
        cuis = []
        for disease, info in results.items():
            if 'cui' in info:
                cuis.append(f"UMLS_{info['cui']}")
        return cuis
    

# 测试代码
# if __name__ == "__main__":
#     # 创建 DiseaseTargetSearch 实例
#     searcher = DiseaseTargetSearch()

#     # 测试用疾病列表（包含中文和英文疾病名称）
#     disease_list = ["糖尿病", "高血压", "Diabetes Mellitus", "Hypertension"]

#     # Step 1: 测试 preprocess_disease_list
#     print("=== Step 1: 预处理疾病列表 ===")
#     preprocessed_diseases = searcher.preprocess_disease_list(disease_list)
#     print(f"输入疾病列表: {disease_list}")
#     print(f"预处理后的疾病列表: {preprocessed_diseases}")

#     # Step 2: 测试 get_umls_cuis
#     print("\n=== Step 2: 获取 UMLS CUI 编码 ===")
#     umls_cuis = searcher.get_umls_cuis(preprocessed_diseases)
#     print(f"疾病列表对应的 UMLS CUI 编码: {umls_cuis}")

    def get_gene_targets(self, cuis: List[str]) -> Set[str]:
        """
        获取疾病相关的基因靶点
        """
        result = get_disease_gene_targets(cuis)
        all_genes = set()
        for disease_cui, genes in result.items():
            all_genes.update(genes)
        return all_genes

    def get_genecards_targets(self) -> Set[str]:
        """
        处理GeneCards数据文件并获取基因靶点
        """
        genecards_genes = set()
        
        processor = GeneCardsProcessor()
        result = processor.process_all_files()
        genecards_genes.update(result)
        return genecards_genes

    def process(self, 
                disease_list: List[str], 
                include_genecards: bool = False) -> Dict:
        """
        处理疾病列表并返回基因靶点（转换为Entrez ID）
        """
        try:
            # 1. 预处理疾病列表
            processed_diseases = self.preprocess_disease_list(disease_list)
            
            # 2. 获取UMLS CUIs
            cuis = self.get_umls_cuis(processed_diseases)
            
            # 3. 获取DisGeNet基因靶点
            disgenet_targets = self.get_gene_targets(cuis)
            
            # 4. 如果需要，处理GeneCards数据
            genecards_targets = set()
            if include_genecards:
                genecards_targets = self.get_genecards_targets()
            
            # 5. 合并所有来源的靶点
            all_targets = disgenet_targets.union(genecards_targets)
            
            # 6. 转换为Entrez ID
            targets_dict = {
                "disease_targets": list(all_targets)
            }
            entrez_results = convert_symbols_to_entrez(targets_dict)
            
            # 获取结果
            result = {
                "status": "success",
                "original_diseases": disease_list,
                "processed_diseases": processed_diseases,
                "cuis": cuis,
                "gene_targets": {
                    "all": entrez_results["disease_targets"],  # Entrez IDs
                    "disgenet": list(disgenet_targets),
                    "genecards": list(genecards_targets) if include_genecards else []
                },
                "sources": {
                    "disgenet": True,
                    "genecards": include_genecards
                }
            }
            
            # 保存Entrez ID列表到CSV文件
            output_path = r'D:\HerbAgent\data\PPI_test_data\Disease_targets\disease.csv'
            df = pd.DataFrame(result['gene_targets']['all'], columns=['EntrezID'])
            df.to_csv(output_path, index=False)
            
            return result
            
        except Exception as e:
            return {
                "status": "error",
                "message": str(e),
                "original_diseases": disease_list
            }

# 使用示例
if __name__ == "__main__":
    searcher = DiseaseTargetSearch()
    test_diseases = ["甲状腺癌"]

    result = searcher.process(
        test_diseases,
        include_genecards=False,   
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"\n疾病基因已保存到: D:\\HerbAgent\\data\\PPI_test_data\\Disease_targets\\disease.csv")




# 提供了灵活的接口，可以选择是否包含GeneCards数据 (True/False)
# 返回结果中清晰地标明了数据来源
# 便于后期扩展，比如添加其他数据源
# 结果格式统一，便于前端处理
# 后期如果需要添加前端界面，只需要：
# 将文件上传功能添加到前端
# 在后端添加文件接收和验证的接口
# 调用现有的处理函数



#要在langchain中使用, 只要创建一个简单的包装函数
# def search_disease_targets(disease_list: List[str]) -> Dict:
#     searcher = DiseaseTargetSearch()
#     return searcher.process(disease_list)


#结果展示
# Searching for term: Lung Cancer
# Searching for term: Thyroid Cancer
# {
#   "status": "success",
#   "original_diseases": [
#     "甲状腺癌",
#     "Thyroid Cancer",
#     "肺癌"
#   ],
#   "processed_diseases": [
#     "Lung Cancer",
#     "Thyroid Cancer"
#   ],
#   "cuis": [
#     "UMLS_C0242379",
#     "UMLS_C0007115"
#   ],
#   "gene_targets": [
#     "BMP2",
#     "KRAS",
#     "EGFR-AS1",
#     "BRCA1",
#     "LOC126860202",
#     "ERBB2",
#     "MXRA5",
#     "ATM",
#     "PRKN",
#     "PALB2",
#     "CASP8",
#     "SMAD4",
#     "C11orf65",
#     "BAP1",
#     "PTPRT",
#     "CHEK2",
#     "ERCC6",
#     "BARD1",
#     "ERCC6",
#     "BARD1",
#     "BARD1",
#     "NFE2L2",
#     "NFE2L2",
#     "SMAD3",
#     "BRCA2",
#     "BRCA2",
#     "VHL",
#     "KMT2D",
#     "MLH1",
#     "KMT2D",
#     "MLH1",
#     "MLH1",
#     "PIK3CA",
#     "PIK3CA",
#     "EGFR",
#     "PPP2R1B",
#     "PPP2R1B",
#     "PGBD3",
#     "PGBD3",
#     "FASLG",
#     "SMAD9",
#     "BRAF",
#     "BRAF",
#     "SLC22A18",
#     "STK11",
#     "ALK"
#   ]
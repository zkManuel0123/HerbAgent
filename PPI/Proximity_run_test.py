from Proximity import ProximityAnalyzer
import pandas as pd

def read_disease_file(file_path: str) -> str:
    """
    读取疾病基因文件
    支持CSV格式，第一列为EntrezID
    """
    try:
        df = pd.read_csv(file_path)
        # 直接使用第一列的数据，不检查列名
        entrez_ids = df.iloc[:, 0].dropna().astype(int).astype(str)  # 确保是整数并转为字符串
        content = '\n'.join(entrez_ids)
        print(f"成功读取疾病基因文件，共 {len(entrez_ids)} 个基因")
        return content
    except Exception as e:
        raise Exception(f"读取疾病基因文件失败: {str(e)}")

def read_drug_file(file_path: str) -> str:
    """
    读取药物靶点文件
    支持制表符分隔的txt文件，第一列为药物名，后续列为EntrezID
    """
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        return content
    except Exception as e:
        raise Exception(f"读取药物靶点文件失败: {str(e)}")

def run_proximity_test(disease_file_path: str, drug_file_path: str, nsims: int = 1000) -> None:
    """
    运行近似性分析
    
    Args:
        disease_file_path: 疾病基因文件路径
        drug_file_path: 药物靶点文件路径
        nsims: 模拟次数
    """
    # 初始化分析器
    analyzer = ProximityAnalyzer()
    
    try:
        # 读取疾病基因文件
        disease_genes_content = read_disease_file(disease_file_path)
        print(f"读取到的疾病基因数量: {len(disease_genes_content.split())}")
        print(f"疾病基因内容示例: {disease_genes_content[:100]}...")  # 打印前100个字符
        
        # 读取药物靶点文件
        drug_targets_content = read_drug_file(drug_file_path)
        print(f"读取到的药物靶点文件行数: {len(drug_targets_content.splitlines())}")
        print(f"药物靶点内容示例: {drug_targets_content.splitlines()[0]}")  # 打印第一行
        
        # 读取网络文件（固定路径）
        network_path = r'D:\HerbAgent\data\PPI_test_data\PPI.csv'
        with open(network_path, 'r') as f:
            network_content = f.read()
            print(f"网络文件前100个字符: {network_content[:100]}...")
            
        # 运行分析
        try:
            results = analyzer.analyze_proximity(
                disease_genes_content=disease_genes_content,
                drug_targets_content=drug_targets_content,
                network_content=network_content,
                nsims=nsims
            )
        except Exception as e:
            print(f"分析过程中发生错误: {str(e)}")
            import traceback
            print("详细错误信息:")
            print(traceback.format_exc())
            return
        
        # 打印结果
        print("\n分析结果：")
        for result in results:
            print(f"\n药物: {result['drug']}")
            print(f"疾病: {result['disease']}")
            print(f"距离: {result['distance']:.4f}")
            print(f"Z-score: {result['z_score']:.4f}")
            print(f"P-value: {result['p_value']:.4f}")
            
    except Exception as e:
        print(f"发生错误: {str(e)}")
        import traceback
        print("详细错误信息:")
        print(traceback.format_exc())

if __name__ == "__main__":
    # 测试用例
    disease_file = r'D:\HerbAgent\data\PPI_test_data\disease.csv'
    drug_file = r'D:\HerbAgent\data\PPI_test_data\drug.txt'
    
    run_proximity_test(
        disease_file_path=disease_file,
        drug_file_path=drug_file
    )

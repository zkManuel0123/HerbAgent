import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

class ProximityAnalyzer:
    """
    药物-疾病近似性分析器
    """
    def __init__(self, file_path):
        """
        初始化分析器
        
        Parameters:
        -----------
        file_path : str
            Excel文件路径
        """
        self.file_path = Path(file_path)
        self.data = None
        self.stats = None
        self.drug_analysis = None
        
    def load_data(self):
        """加载数据并进行基础处理"""
        try:
            self.data = pd.read_excel(self.file_path)
            # 确保数值列的类型正确
            self.data['z-value'] = pd.to_numeric(self.data['z-value'])
            self.data['p-value'] = pd.to_numeric(self.data['p-value'])
            print(f"成功加载数据，共 {len(self.data)} 条记录")
            return True
        except Exception as e:
            print(f"加载数据时出错: {str(e)}")
            return False
    
    def calculate_basic_stats(self):
        """计算基本统计指标"""
        if self.data is None:
            print("请先加载数据")
            return
        
        self.stats = {
            'z值统计': {
                '平均值': self.data['z-value'].mean(),
                '中位数': self.data['z-value'].median(),
                '标准差': self.data['z-value'].std(),
                '最小值': self.data['z-value'].min(),
                '最大值': self.data['z-value'].max()
            },
            'p值统计': {
                '显著关联数(p<0.05)': len(self.data[self.data['p-value'] < 0.05]),
                '总关联数': len(self.data),
                '显著关联比例': len(self.data[self.data['p-value'] < 0.05]) / len(self.data)
            }
        }
        
    def analyze_drug_associations(self):
        """分析药物关联"""
        if self.data is None:
            print("请先加载数据")
            return
            
        self.drug_analysis = self.data.groupby('drug').agg({
            'z-value': ['mean', 'std', 'count'],
            'p-value': lambda x: (x < 0.05).sum()
        }).round(3)
        
        self.drug_analysis.columns = ['平均z值', 'z值标准差', '关联数量', '显著关联数']
        self.drug_analysis['显著关联比例'] = (self.drug_analysis['显著关联数'] / 
                                            self.drug_analysis['关联数量']).round(3)
        
    def plot_distributions(self, save_path=None):
        """
        绘制分布图
        
        Parameters:
        -----------
        save_path : str, optional
            图片保存路径
        """
        if self.data is None:
            print("请先加载数据")
            return
            
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Z-value分布图
        sns.histplot(data=self.data, x='z-value', ax=ax1)
        ax1.set_title('Z值分布')
        ax1.set_xlabel('Z值')
        ax1.set_ylabel('频数')
        
        # P-value分布图
        sns.histplot(data=self.data, x='p-value', ax=ax2)
        ax2.set_title('P值分布')
        ax2.set_xlabel('P值')
        ax2.set_ylabel('频数')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            print(f"图表已保存至: {save_path}")
        else:
            plt.show()
            
    def generate_report(self, output_dir=None):
        """
        生成分析报告
        
        Parameters:
        -----------
        output_dir : str, optional
            输出目录路径
        """
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
        
        # 准备报告内容
        report = []
        report.append("=== 药物-疾病近似性分析报告 ===\n")
        
        # 1. 基本统计信息
        report.append("1. 基本统计信息")
        report.append("-" * 50)
        for category, stats in self.stats.items():
            report.append(f"\n{category}:")
            for name, value in stats.items():
                report.append(f"  {name}: {value:.4f}")
        
        # 2. 药物分析结果
        report.append("\n\n2. 药物关联分析")
        report.append("-" * 50)
        report.append("\n按显著关联数排序的前5个药物:")
        top_drugs = self.drug_analysis.sort_values('显著关联数', ascending=False).head()
        report.append(str(top_drugs))
        
        # 保存报告
        report_text = "\n".join(report)
        if output_dir:
            report_file = output_dir / "分析报告.txt"
            with open(report_file, "w", encoding="utf-8") as f:
                f.write(report_text)
            print(f"报告已保存至: {report_file}")
            
            # 保存详细数据
            excel_file = output_dir / "药物分析详情.xlsx"
            self.drug_analysis.to_excel(excel_file)
            print(f"详细分析结果已保存至: {excel_file}")
            
            # 保存分布图
            self.plot_distributions(str(output_dir / "分布图.png"))
        else:
            print(report_text)

# 使用示例
def main():
    # 文件路径
    file_path = r"D:\HerbAgent\output\proximity_results.xlsx"
    output_dir = r"D:\HerbAgent\output\analysis_results"
    
    # 创建分析器实例
    analyzer = ProximityAnalyzer(file_path)
    
    # 执行分析流程
    if analyzer.load_data():
        analyzer.calculate_basic_stats()
        analyzer.analyze_drug_associations()
        analyzer.generate_report(output_dir)
    
if __name__ == "__main__":
    main()
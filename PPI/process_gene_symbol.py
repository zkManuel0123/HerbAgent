import pandas as pd

# 加载文件并调整格式
file_path = "C:\\Users\\manue\\Desktop\\HerbAgent\\PPI\\gene.txt"
output_path = "C:\\Users\\manue\\Desktop\\HerbAgent\\PPI\\gene_cleaned.txt"

# 读取文件并重命名列
df = pd.read_csv(file_path, sep="\t")
df.columns = ['EntrezID']  # 确保列名为 'EntrezID'

# 保存为新的制表符分隔文件
df.to_csv(output_path, sep="\t", index=False)
print(f"文件已调整并保存至: {output_path}")

# 绘制热图+网医/证-复方proximity

df1 = read.table("C:\\Users\\GaoKai\\Desktop\\network pharmacology\\data\\肝病药物_复方proximity计算\\syndrome\\formulae+syndrome_proximity.csv", 
            sep = ",", header=T)

# 长型数据转宽型矩阵
heat_proximity = reshape2::dcast(df1, disease~Formulae.ID, value.var = 'proximity.Z.score..z')
row.names(heat_proximity) = heat_proximity$disease
heat_proximity = as.matrix(heat_proximity[, -1])

write.table(heat_proximity,"C:\\Users\\GaoKai\\Desktop\\network pharmacology\\data\\肝病药物_复方proximity计算\\syndrome\\heat_proximity_matrix.csv",
            sep = ",", row.names = T)

# 排序整理后重新导入矩阵
df2 = read.table("C:\\Users\\GaoKai\\Desktop\\network pharmacology\\data\\肝病药物_复方proximity计算\\syndrome\\heat_proximity_matrix.csv", 
                 sep = ",", header=T)
row.names(df2) = df2[,2]
heat_proximity = as.matrix(df2[, -(1:2)])

# 绘制热图
library(ComplexHeatmap)

# 注意这是ComplexHeatmap::pheatmap
pheatmap(heat_proximity, cluster_row = FALSE, cluster_cols = FALSE,
         display_numbers = matrix(ifelse(heat_proximity < -1.5, "▲", ""),
                                  nrow(heat_proximity)), # 显著性与cutoff标注
         # legend_breaks = -15:10,
         # legend_labels = c("-15", "-5", "-1.5", "0", "1.5", "5", "15")
         # scale = "row", #行/列标准化
         color = colorRampPalette(c("firebrick3", "white", "navy"))(50), #填充颜色
         cellwidth = 50,
         cellheight = 30,
         main = "The network proximities of 5 TCM syndrome with 5 recommended formulas",
         angle_col = "45",
         row_names_side = "left", #行名位置
         column_names_side="top", #列名位置
         heatmap_legend_param = list(
           color_bar = "discrete", #以离散型方式标注连续性变量legend
           at = c(15, 10, 1.5, 0, -1.5, -10, -15),
           # labels = c("low", "zero", "high"),
           title = "z-score",
           legend_height = unit(5, "cm"),
           title_position = "topcenter"
         )
         )


# ++++++++++++++++++++++++++绘制热图+网医/疾病-药物proximity++++++++++++++++++++++++++

df1 = read.table("C:\\Users\\gaoka\\Desktop\\network pharmacology\\data\\肝病药物_复方proximity计算\\drug\\All_heatmap_proximity_negative.csv", 
                 sep = ",", header=T)


row.names(df1) = df1$Formulae.and.drug.ID
heat_proximity = as.matrix(df1[,-1])

# 绘制热图

# if (!require("BiocManager", quietly = TRUE))
#   install.packages("BiocManager")
# 
# BiocManager::install("ComplexHeatmap")

library(ComplexHeatmap)

# 注意这是ComplexHeatmap::pheatmap
pheatmap(heat_proximity, cluster_row = FALSE, cluster_cols = FALSE,
         display_numbers = matrix(ifelse(heat_proximity < -1.5, "▲", ""),
                                  nrow(heat_proximity)), # 显著性与cutoff标注
         # legend_breaks = -15:10,
         # legend_labels = c("-15", "-5", "-1.5", "0", "1.5", "5", "15")
         # scale = "row", #行/列标准化
         color = colorRampPalette(c("firebrick3", "white", "navy"))(50), #填充颜色
         cellwidth = 30,
         cellheight = 12,
         # main = "The network proximities of 2 HCC gene sets with drugs and formulas",
         angle_col = "45",
         row_names_side = "left", #行名位置
         column_names_side="top", #列名位置
         heatmap_legend_param = list(
           color_bar = "discrete", #以离散型方式标注连续性变量legend
           at = c(5, 1.5, 0, -1.5, -5, -10),
           # labels = c("low", "zero", "high"),
           title = "z-score",
           legend_height = unit(5, "cm"),
           title_position = "topcenter"
         )
)


# pdf(10, 4.5)


# ++++++++++++++++++++++++++绘制热图+网医/中医症状-中药proximity++++++++++++++++++++++++++

df1 = read.table("C:\\Users\\GaoKai\\Desktop\\network pharmacology\\data\\肝病药物_复方proximity计算\\TCM_symptom\\Syndrome4.csv", 
                 sep = ",", header=T)
head(df1)
df1 = df1[,-1]
row.names(df1) = df1[,1]
heat_proximity = as.matrix(df1[,-1])

# 绘制热图
library(ComplexHeatmap)

# 注意这是ComplexHeatmap::pheatmap
pheatmap(heat_proximity, cluster_row = TRUE, cluster_cols = FALSE,
         display_numbers = matrix(ifelse(heat_proximity < -1.5, "▲", ""),
                                  nrow(heat_proximity)), # 显著性与cutoff标注
         # legend_breaks = -15:10,
         # legend_labels = c("-15", "-5", "-1.5", "0", "1.5", "5", "15")
         scale = "column", #行/列标准化 column row
         color = colorRampPalette(c("firebrick3", "white", "navy"))(50), #填充颜色
         cellwidth = 30,
         cellheight = 20,
         # main = "The network proximities of TCM symptom with herbs",
         main = "Syndrome 4 & Formula 4",
         angle_col = "45",
         row_names_side = "right", #行名位置
         column_names_side="top", #列名位置
         heatmap_legend_param = list(direction = "horizontal",
                                       title = "scale(z-score)"
                                     )
         # heatmap_legend_param = list(
         #   direction = "horizontal",
         #   color_bar = "discrete", #以离散型方式标注连续性变量legend
         #   # at = c(15, 10, 1.5, 0, -1.5, -10, -15),
         #   # labels = c("low", "zero", "high"),
         #   title = "scale(z-score)"#,
         #   # legend_height = unit(5, "cm"),
         #   # title_position = "topleft"
         # )
)



heatmap_legend_param = list(direction = "horizontal")

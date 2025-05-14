
rm(list = ls()) 
options(stringsAsFactors = F)

 if (!require("BiocManager", quietly = TRUE))
   install.packages("BiocManager")
 
 BiocManager::install("RandomWalkRestartMH")
 
 if (!requireNamespace("BiocManager", quietly = TRUE))
   install.packages("BiocManager")
 
 BiocManager::install("supraHex")
 install.packages("rlang")
 install.packages("vctrs")
 install.packages("Rgraphviz")
 library(supraHex)
library(dnet)

#设置工作路径
wkdir <- "D:\\jiazhuangxian\\Network_JZX\\RWR\\9_fang" #这里需要改成自己的文件夹地址
setwd(wkdir)

# +++++++++++++++++++++++++++++++++++++++++++++
library(RandomWalkRestartMH)
library(igraph)

Formula_ingredient  <- read.csv("Formula_5_herb_ingredient.csv")

Formula_ingredient_MultiplexObject <- create.multiplex(list(Formula_ingredient = graph_from_data_frame(Formula_ingredient, directed = FALSE)))
Formula_ingredient_MultiplexObject


PPI  <- read.csv("PPI_LCC_edges.csv", header = FALSE)

PPI_MultiplexObject <- create.multiplex(list(PPI = graph_from_data_frame(PPI, directed = FALSE)))
PPI_MultiplexObject


Ingredient_target  <- read.csv("Formula_5_ingredient_target.csv")

GeneIngredientRelations_PPI <-
  Ingredient_target[which(Ingredient_target$Target_name %in%
                               PPI_MultiplexObject$Pool_of_Nodes),]
GeneIngredientRelations_PPI

## We create the MultiplexHet object.

PPI_Ingredient_Net <- create.multiplexHet(Formula_ingredient_MultiplexObject, PPI_MultiplexObject,
                                          GeneIngredientRelations_PPI)


PPI_Ingredient_Net


PPIHetTranMatrix <- compute.transition.matrix(PPI_Ingredient_Net)

# 读取Syndrome文件  
Syndrome <- read.csv("Syndrome_2.csv", header = TRUE)  
seed_nodes <- Syndrome$SYMBOL  

SeedDisease <- c()

## We launch the algorithm with the default parameters
RWRH_PPI_Disease_Results <-
  Random.Walk.Restart.MultiplexHet(PPIHetTranMatrix,
                                   PPI_Ingredient_Net, SeedDisease, seed_nodes)


Score <- RWRH_PPI_Disease_Results$RWRMH_Multiplex1
Score2 <- RWRH_PPI_Disease_Results$RWRMH_GlobalResults


write.csv(Score, file = "Formula_5_drug_sort_1seed---w.csv", row.names = FALSE)
write.csv(Score2, file = "Formula_5_GlobaResult_1seed---w.csv", row.names = FALSE)




# 基因ID转换

# org.Hs.eg.db包

#16个m6A甲基化相关的基因名字
# setwd("C:\\Users\\GaoKai\\Desktop\\") 

PPI=read.table("C:\\Users\\GaoKai\\Desktop\\PPI数据集\\PPI Set\\9. InnateDB\\innatedb_ppi.mitab",
               header=F, fill= T)  
head(PPI)

PPI1 = PPI[,1:2]
head(PPI1)

ID0 = PPI[,1]
ID1 = PPI[,2]

ID00 = unlist(strsplit(ID0, '-'))
ID11 = unlist(strsplit(ID1, '-'))

ID00 = strsplit(ID0, '-')
ID11 = strsplit(ID1, '-')
ID000 = sapply(ID00, '[', 2)
ID111 = sapply(ID11, '[', 2)

head(ID000)
class(ID000)

PPI1$geneid1 = ID000
PPI1$geneid2 = ID111
head(PPI1)


ID0 = PPI[,1]
ID1 = PPI[,2]

colnames(PPI1) = PPI1[1,]
head(PPI1)
PPI1 = PPI1[-1,]
head(PPI1)
dim(PPI1)

PPI2 = PPI1[((PPI1$KIN_ORGANISM == 'human')&(PPI1$SUB_ORGANISM == 'human')),]
head(PPI2)
dim(PPI2)

PPI3 = PPI2[,c(3,6,7)]
head(PPI3)
ID0 = PPI3[,1]
ID1 = PPI3[,3]
ensembls <- mapIds(org.Hs.eg.db, keys = c(ID0, ID1), keytype = "UNIPROT", column="ENTREZID")
head(ensembls) 

PPI3[,4:5] <- ensembls
head(PPI3)

abc = PPI3$SUB_GENE_ID == PPI3$V5
as.character(abc)
sum(abc = 'FALSE')
PPI3$ABC = PPI3$abc
sum(PPI3$ABC == "FALSE")
colnames(PPI3)[4] <- 'KIN_GENE_ID'

write.table(PPI1,"C:\\Users\\GaoKai\\Desktop\\PPI数据集\\PPI Set\\9. InnateDB\\innatedb_ppi.tsv",
            sep = "\t", quote = F, row.names = F, col.names = T)



#如果没有安装org.Hs.eg.db，需要先运行下面这条命令安装
#BiocManager::install("org.Hs.eg.db")

#加载org.Hs.eg.db
library(org.Hs.eg.db)

citation("org.Hs.eg.db")

#查看支持哪些ID
columns(org.Hs.eg.db)
# gene symbol转成Ensembl gene ID
ensembls <- mapIds(org.Hs.eg.db, keys = c(ID0, ID1), keytype = "UNIPROT", column="ENTREZID")
head(ensembls) 

PPI1[,3:4] <- ensembls
head(PPI1)

write.table(PPI1,"C:\\Users\\GaoKai\\Desktop\\PPI数据集\\PPI Set\\3. Interactome INSIDER\\H_sapiens_interfacesHQ.tsv",
            sep = "\t", quote = F, row.names = F, col.names = F)


# +++++++++++++++++++drugbank polypeptide_id to ENTREZID+++++++++++++++++++
PPI=read.table("C:\\Users\\GaoKai\\Desktop\\network pharmacology\\data\\Union+PPI_LCC.csv", 
               sep = ",", header=F, fill= T)
head(PPI)

ID0 = as.character(PPI[,1])
ID1 = as.character(PPI[,2])

ensembls <- mapIds(org.Hs.eg.db, keys = c(ID0, ID1), keytype = "ENTREZID", column="SYMBOL")
head(ensembls) 

PPI[,3:4] <- ensembls
head(PPI)

write.table(PPI,"C:\\Users\\GaoKai\\Desktop\\network pharmacology\\data\\Union+PPI_LCC_geneSymbol.tsv",
            sep = "\t", quote = F, row.names = F, col.names = F)

# ++++++++++++++++++++++++++++++++++++++ending++++++++++++++++++++++++++++++++++++++
if (!require("BiocManager", quietly = TRUE))
  install.packages("BiocManager")

BiocManager::install("org.Hs.eg.db")

library(org.Hs.eg.db)
#查看支持哪些ID
columns(org.Hs.eg.db)

PPI=read.table("C:\\Users\\gaoka\\Desktop\\TTD_uniprot_all.csv", 
               sep = ",", header=T)
head(PPI)



# PPI = PPI[,-3]
# head(PPI)

ID0 = as.character(PPI[,2])

SYMBOL <- mapIds(org.Hs.eg.db, keys = ID0, keytype = "UNIPROT", column="SYMBOL")
head(SYMBOL) 

PPI$SYMBOL <- SYMBOL
head(PPI)
# PPI2 = PPI[,c(1,6)]

write.table(PPI,"C:\\Users\\gaoka\\Desktop\\syndrome_of_liver-kidney_yin_deficiency_2.csv", 
            sep = ",", quote = F, row.names = F)

class(ID0)
ensembls <- mapIds(org.Hs.eg.db, keys = ID0, keytype = "UNIPROT", column="ENTREZID")
head(ensembls) 




# gene symbol转成Entrez gene ID
entriz <- mapIds(org.Hs.eg.db, keys = m6a_sym, keytype = "SYMBOL", column="ENTREZID")
entriz

#一次性转换到ENSEMBL ID,ENTREZ ID和UNIPROT ID
AnnotationDbi::select(org.Hs.eg.db, keys=m6a_sym,keytype="SYMBOL", columns = c("ENSEMBL","ENTREZID","UNIPROT"))


# clusterProfiler包

#如果没有安装clusterProfiler，需要先运行下面这条命令安装
#BiocManager::install("clusterProfiler")

#加载clusterProfiler
library(clusterProfiler)
s2ens = bitr(ID0, fromType="ENSEMBL", toType="ENTREZID", OrgDb="org.Hs.eg.db")
s2ens

PPI$v3 = s2ens

bitr(m6a_sym, fromType="SYMBOL", toType=c("ENSEMBL","ENTREZID"), OrgDb="org.Hs.eg.db")


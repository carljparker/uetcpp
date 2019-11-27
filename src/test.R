#############################
## UETCPP - Adaptation à Rcpp
#############################

#Package Rcpp
library(Rcpp)
library(RcppEigen)
#Package pour le clustering
library(VarSelLCM)
library(aricode)
sourceCpp("main2.cpp")

#Préparation des données test
data("heart")
dfHeart<-heart[,1:12]
coltypes=c(1,0,0,1,1,0,0,1,0,0,0,0)
dfHeart_list<-split(dfHeart, seq(nrow(dfHeart)))
dfHeart_list<-lapply(dfHeart_list,as.numeric)
names(dfHeart_list)<-NULL
attributes<-0:11
attributes_indices<-0:11
nodes_indices<-0:(nrow(dfHeart)-1)

#Distance uet
a<-build_randomized_tree_and_get_sim(dfHeart_list,10, coltypes, 1000,nodes_indices)
a<-as.matrix(a)

#Application au clustering hiérarchique
resHC<-cutree(hclust(dist(a),method = "ward.D2"),2)

#Comparaison aux classes d'origine
table(resHC,heart$Class)
ARI(resHC,heart$Class)

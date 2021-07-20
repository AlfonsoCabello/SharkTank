
# FINAL INDIVIDUAL PROJECT CODE #

###########################################################################
#Section 0: Data Preprocessing
# Section 0.1: GradientBoostingRegressor
# Section 0.2: Km Clustering 
#Section 1: Model buil-up and Predictions
# Section 1.1: GradientBoostingRegressor
# Section 1.2: Km Clustering
###########################################################################

###################-------------------- SECTION-0--------------------########################

#Libraries
#Check if user has the required packages installed, if not -> install
packages <- c("devtools", "ggplot2", "ggpubr",'randomForest','fastDummies','tidyr','caret','gbm','e1071','factoextra','lmtest','plm','car','treemapify')
install.packages(setdiff(packages, rownames(installed.packages())))

#Run libraries
library(ggplot2)
library(ggpubr)
library(randomForest)
library(fastDummies)
library(dplyr)
library(tidyr)
library(caret)
library(e1071)
library(gbm)
library(factoextra)
library(lmtest)
library(plm)
library(car)
library(treemapify)

#Extract dataset
dataset_o=read.csv('shark_tank.csv')
attach(dataset_o)

# (0.1) DATA PREPROCESSING - GradientBoostingRegressor ###################

############Features with no value

#Extract features that do not add value to the model. Variables like episode, names, season and Sharks that closed a deal are omitted.
dataset_f = subset(dataset_o, select = -c(description,episode,entrepreneurs1,entrepreneurs2,entrepreneurs3,location,website,season,shark1,shark2,shark3,shark4,shark5,title,episode.season,Back_Sharks,	Deal_Shark1,	Deal_Shark2,	Deal_Shark3,	Deal_Shark4,	Deal_Shark5))
dataset_f = drop_na(dataset_f)

############Dummify

#Convert to factors variables that already are coded 0/1
dataset_f$deal=ifelse(deal=='TRUE',1
dataset_f$deal=as.factor(dataset_f$deal)

dataset_f$has_website=as.factor(dataset_f$has_website)

dataset_f$Multiple.Entreprenuers=ifelse(Multiple.Entreprenuers=='TRUE',1,0)
dataset_f$Multiple.Entreprenuers=as.factor(dataset_f$Multiple.Entreprenuers)

#Create new table that has all the variables + dummies of the variables 'categories' and 'state'.
dataset_f_cat <- dummy_cols(dataset_f, select_columns = c('category','state'), remove_selected_columns = TRUE)

#Convert to dataframe
dataset_f_cat = data.frame(dataset_f_cat)
######

############OutlierTest

#Logistic regression to be used in the bonferroni test
mlogit = glm(deal~.,data=dataset_f_cat, family = "binomial") 
summary(mlogit)
outlierTest(mlogit)

#rstudent unadjusted p-value Bonferroni p
#312 -3.712376         0.00020532     0.083156

dataset_f_cat = dataset_f_cat[-c(312),]

##############Feature Selection 

### Random Forest
set.seed (1) 
forest_fs = randomForest(deal~.,ntree = 3000,data=dataset_f_cat,importance=TRUE,do.trace=200,na.action = na.omit)

importance(forest_fs)
varImpPlot(forest_fs)

f_to_useRF=data.frame(importance(forest_fs))
write.csv(f_to_useRF,"fs_rf_RF.csv", row.names = TRUE)

### GBM
set.seed (1) 
boosted_fs=gbm(as.integer(deal) - 1 ~.,data = dataset_f_cat, distribution= "bernoulli", n.trees=1000, interaction.depth=3) 
summary(boosted_fs)
f_to_useB=data.frame(summary(boosted_fs))
#write.csv(f_to_useB,"fs_rfb.csv", row.names = TRUE)

#Features selected for analysis
dataset_fs_gbm = subset(dataset_f_cat, select = c(deal, len_title,	askedFor,	exchangeForStake,	state_CA,	count_entrepreneurs,	category_Novelties,	state_FL,	category_Specialty.Food,	state_NY,	Multiple.Entreprenuers,	has_website,	state_TX,	category_Baby.and.Child.Care,	category_Online.Services,	state_IL,	category_Toys.and.Games,	category_Storage.and.Cleaning.Products,	category_Consumer.Services,	category_Personal.Care.and.Cosmetics,	state_GA,	category_Electronics,	state_UT,	state_NJ,	category_Professional.Services,	state_NC,	state_OR,	state_PA))



# (0.2) DATA PREPROCESSING - Km Clustering ###################


############Features with no value

#Extract features that do not add value to the model. Variables like episode, names, season and Sharks that closed a deal are omitted.
dataset_cluster = subset(dataset_o, select = -c(deal,description,episode,category,entrepreneurs1,entrepreneurs2,entrepreneurs3,location, state,website, has_website,season,shark1,shark2,shark3,shark4,shark5,title,episode.season, Multiple.Entreprenuers,Back_Sharks,	Deal_Shark1,	Deal_Shark2,	Deal_Shark3,	Deal_Shark4,	Deal_Shark5))
dataset_cluster = drop_na(dataset_cluster)

############Feature Scaling
dataset_cluster_std = scale(dataset_cluster)

set.seed(123)
### Elbow method
# Elbow method for kmeans to find the right number of clusters
fviz_nbclust(dataset_cluster_std, kmeans, method = "wss") +geom_vline(xintercept = 3, linetype = 2)

# Average silhouette for kmeans to find the right number of clusters
fviz_nbclust(dataset_cluster_std, kmeans, method = "silhouette")


###################-------------------- SECTION-1--------------------########################

# (1.1) MODEL BUIL-UP AND PREDICTIONS - GradientBoostingRegressor ###################

set.seed (1) 

#Model Build up
boosted1=gbm(as.integer(deal) - 1 ~.,data = dataset_fs_gbm, distribution= "bernoulli", n.trees=3000, interaction.depth=3) 
boosted1
summary(boosted1)

#Get predictions
predicted=predict(boosted1, newdata=dataset_fs_gbm, type ='response', n.trees=1000)
dataset_fs_gbm$predicted2=ifelse(predicted>.5,1,0)

#Confusion matrix to know accuracy
dataset_fs_gbm$predicted2=as.factor(dataset_fs_gbm$predicted2)
confusionMatrix(dataset_fs_gbm$predicted2, dataset_fs_gbm$deal)

#Accuracy = 89%


# (1.2) MODEL BUIL-UP AND PREDICTIONS - Km Clustering ###################

set.seed(123)
c = kmeans(dataset_cluster_std, 3, iter.max = 10, nstart = 25)

#means of clusters using data
aggregate(dataset_cluster, by=list(cluster=c$cluster), mean)

#add clusters to dataset
dataset_cluster$cluster_km = c$cluster

###################-------------------- END --------------------########################


################## Data Analysis

###Proportions of categories
library(dplyr)
dataset_treemap1 = dplyr::filter(dataset_o, deal == 'TRUE')

dataset_treemap2 = subset(dataset_treemap1, select = c(category,valuation, askedFor))

dataset_treemap_g = dataset_treemap2 %>%
  dplyr::group_by(category)%>%
  dplyr::summarize(total_val = sum(valuation),total_askedfor = sum(askedFor))
  
dataset_treemap_g %>% 
ggplot(aes(area = total_askedfor, fill = total_val, label = category)) +
  geom_treemap() +
  geom_treemap_text(fontface = "italic", colour = "white", place = "centre",
                    grow = TRUE)
%>% 
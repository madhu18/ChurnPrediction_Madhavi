####Focusing on targeted customers.###
#-----customer is exited or not---------#
#--Cleaning work space
rm(list=ls(all=T))
#---setting working directory
setwd("E:/Retail/Artificial_Neural_Networks")
#----reading and storing data in a dataframe
#set.seed(1234)
Data <- read.csv("Churn_Modelling.csv",header = T,sep=",")
table(is.na(Data)) 
colSums(is.na(Data))
apply(is.na(Data),2,sum) 

#---structure -----
str(Data)
summary(Data,maxsum = 15)
       
#----checking for missing values-------
 sum(is.na(Data))
#----------preprocessing steps----------------
##-----------------Removing duplicated  rows--------
#Data1 <- Data[duplicated(Data),]
Data<- Data[!duplicated(Data),]
summary(Data)
str(Data)
#---converting into factor
Data$Exited <-  as.factor(Data$Exited)
#---------------------Data Exploration------------------
#Plotting Dependent variable
x=Data$Balance
y=Data$Exited
par(mfrow=c(1,1))
#checking for outliers
plot(y, x,xlab="Target",ylab="Frequency",main="Distribution of Dependent variable")
library(ggplot2)
#checking Frequenct distribution for class/Target variable
plot(y, xlab="Target",type="b")
ggplot(aes(x=Exited, y=Balance),data=Data)+geom_bar(mapping=NULL,data=NULL,stat="Summary") 
ggplot(aes(x=Exited, y=Balance,colour=CreditScore),data=Data)+geom_jitter(alpha=0.3) +stat_summary(fun.y="mean", geom="line") 
ggplot(aes(x=Exited, y=Age,colour=CreditScore),data=Data)+geom_jitter(alpha=0.3) +stat_summary(fun.y="mean", geom="line") 
#---------------------------

#------Spliting the data 70:30-------------------
set.seed(1234)
rowIDs = seq(1, nrow(Data), 1)
trainRowIDs = sample(rowIDs, round(.7 * length(rowIDs)))
testIDs = sample(setdiff(rowIDs , trainRowIDs), round(.3 * length(rowIDs)))
#-----removing attribute Transdate 
main_train = Data[trainRowIDs,] 
main_test= Data[testIDs,]
train_data<-subset(main_train, select=-c(CustomerId,RowNumber))
test_data<-subset(main_test, select=-c(CustomerId,RowNumber))
#seperating numeric  variables
train_data_numer<-subset(train_data,select=c( CreditScore ,Tenure,Balance,EstimatedSalary))
test_data_numer<-subset(test_data,select=c( CreditScore ,Tenure,Balance,EstimatedSalary))
#___________________________________________
#Normalize the numerical attributes having large difference 
library(vegan)
#find max of each coln
traing_set_max <- apply(train_data_numer,2,max)
train_set_std <- sweep(train_data_numer, 2, traing_set_max, `/`)
test_set_std <- sweep(test_data_numer, 2, traing_set_max, `/`) 

#---------Seperating categorical variables
train_data_categ<-subset(train_data,select=-c( CreditScore ,Tenure,Balance,EstimatedSalary))
test_data_categ<-subset(test_data,select=-c( CreditScore ,Tenure,Balance,EstimatedSalary))
#coverting required variables into factor
#train_data_categ$Age<-as.factor(train_data_categ$Age)
train_data_categ$HasCrCard<-as.factor(train_data_categ$HasCrCard)
train_data_categ$IsActiveMember<-as.factor(train_data_categ$IsActiveMember)
str(train_data_categ)
#test_data_categ$Age<-as.factor(test_data_categ$Age)
test_data_categ$HasCrCard<-as.factor(test_data_categ$HasCrCard)
test_data_categ$IsActiveMember<-as.factor(test_data_categ$IsActiveMember)
str(test_data_categ)
#creating new dataframe bynormalize numeric attributes and categorical varibles
train_set_new<- data.frame(train_set_std,train_data_categ)
str(train_set_new)
test_set_new<- data.frame(test_set_std,test_data_categ)
str(test_set_new)

#-----------Decision trees------------------
dt_train_set <- train_set_new
dt_test_set <- test_set_new
str(dt_train_set)
library(C50)
dtC50  <- C5.0(Exited ~ ., data = dt_train_set, rules=TRUE)
dtC50 <- C5.0(Exited ~ ., data = dt_train_set)
summary(dtC50)
#calculates variable importance
C5imp(dtC50, pct=TRUE)
#plotting model
plot(dtC50)
#----predicting on train data and metrics
pred_t_dt= predict.C5.0(dtC50,newdata=dt_train_set, type="class")
a <-table(dt_train_set$Exited,pred_t_dt)
train_accuracy_dt <- sum(diag(a))/sum(a) * 100
train_accuracy_dt
#recall is low because the data set has more zeros than ones.
recall_train_dt <-( (a[2,2]) / ((a[2,1])+ (a[2,2]))) *100
recall_train_dt
precision_train_dt<- ( (a[2,2]) / ((a[1,2])+ (a[2,2]))) *100
#---test prediction
pred_test_dt= predict.C5.0(dtC50,newdata=dt_test_set, type="class")
a1 <-table(dt_test_set$Exited,pred_test_dt)
test_accuracy_dt <- sum(diag(a1))/sum(a1) * 100
test_accuracy_dt
recall_test_dt <-( (a1[2,2]) / ((a1[2,1])+ (a1[2,2]))) *100
recall_test_dt
precision_test_dt<- ( (a1[2,2]) / ((a1[1,2])+ (a1[2,2]))) *100
precision_test_dt
##---------------- Naive Bayes----
nb_train_set <- train_set_new
nb_test_set <- test_set_new
library(e1071)
nb <- naiveBayes(Exited~., data = nb_train_set)
nb
pred_t_nb = predict(nb,newdata=nb_train_set, type="class")
a2 <-table(nb_train_set$Exited,pred_t_nb)
train_accuracy_nb<- sum(diag(a2))/sum(a2) * 100
train_accuracy_nb
recall_train_nb <-( (a2[2,2]) / ((a2[2,1])+ (a2[2,2]))) *100
recall_train_nb
precision_train_nb<- ( (a2[2,2]) / ((a2[1,2])+ (a2[2,2]))) *100
#---test prediction
pred_test_nb=predict(nb,newdata= nb_test_set, type="class")
a3 <-table(nb_test_set$Exited, pred_test_nb)
test_accuracy_nb <- sum(diag(a3))/sum(a3) * 100
test_accuracy_nb
recall_test_nb <-( (a3[2,2]) / ((a3[2,1])+ (a3[2,2]))) *100
recall_test_nb
precision_test_nb<- ( (a3[2,2]) / ((a3[1,2])+ (a3[2,2]))) *100
##writing all the model outputs to dataframe
model_eval_output <- data.frame(DT=c(train_accuracy_dt,test_accuracy_dt,recall_train_dt,recall_test_dt,precision_train_dt,precision_test_dt),
                                NB=c(train_accuracy_nb,test_accuracy_nb,recall_train_nb,recall_test_nb,precision_train_nb,precision_test_nb),
                                row.names = c("acc_train","accu_test","recall_train","recall_test","precision_train","precision_test"))
model_eval_output
###-------------evaluation output to excel 
write.csv(model_eval_output,"evalmetrics.csv")
###predicted values to excel of all models
data_final1 <- data.frame("dt"=pred_t_dt,"nb" = pred_t_nb,"target" = train_set_new$Exited )
write.csv(data_final1,"predtrain.csv")
test_data_final1 <- data.frame("dt1"=pred_test_dt,"nb1" = pred_test_nb,"target1" =test_set_new$Exited)
write.csv(test_data_final1,"predtest.csv")
#####################################
#----------------------END----------------------------------------
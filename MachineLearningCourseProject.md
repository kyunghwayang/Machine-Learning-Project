---
title: "Machine Learning Course Project"
author: "K. Yang"
date: "10/15/2020"
output: 
  html_document:
    keep_md: true
---



Data source: [http://web.archive.org/web/20161224072740/http:/groupware.les.inf.puc-rio.br/har](http://web.archive.org/web/20161224072740/http:/groupware.les.inf.puc-rio.br/har)

Ugulino, W.; Cardador, D.; Vega, K.; Velloso, E.; Milidiu, R.; Fuks, H. Wearable Computing: Accelerometers' Data Classification of Body Postures and Movements

The data documents the ways in which 6 male participants performed barbell lifts by reading inputs from accelerometers on the belt, forearm, arm, and dumbell of each participant. The data is summarized in the last variable, classe, which has five values, A, B, C, D, and E. Each participant was asked to perform one set of 10 repetitions of a specified exercised in five different manners: exactly according to the specification (Class A), throwing the elbows to the front (Class B), lifting the dumbbell only halfway (Class C), lowering the dumbbell only halfway (Class D) and throwing the hips to the front (Class E).

In this report, I predict the type of class based on the given data. Specifically, I intend to use the total accelerometer data read from belts, forearms, arms, and dumbell accelerometers to predict the ways in which an individual performs the exercise. 

**1. INPUT**


```r
url.training <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
url.testing <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"

if(!file.exists("data")) dir.create("data")

download.file (url.training, destfile = "./data/training")
download.file (url.testing, destfile = "./data/testing")

training <- read.csv("./data/training", na.strings=c("NA","","#DIV/0!"))
testing <- read.csv("./data/testing", na.strings=c("NA","","#DIV/0!"))

# data cleaning
training <- training[colSums(is.na(training))==0]
testing <- testing[colSums(is.na(testing))==0]

# remove nonpredictors from the sets
training <- training[, -c(1:7)]
testing <- testing[, -c(1:7)]
dim(training)
```

```
## [1] 19622    53
```

```r
dim(testing)
```

```
## [1] 20 53
```

**2. EDA**


```r
library(caret)
```

```
## Loading required package: lattice
```

```
## Loading required package: ggplot2
```

```r
library(ggplot2)
set.seed(1234)

# [11] "total_accel_belt"
# [49] "total_accel_arm"
# [102] "total_accel_dumbbell" 
# [140] "total_accel_forearm" *



par(mfrow=c(1, 4))
boxplot(total_accel_forearm~classe, data=training)
boxplot(total_accel_dumbbell~classe, data=training)
boxplot(total_accel_arm~classe, data=training)
boxplot(total_accel_belt~classe, data=training)
```

![](MachineLearningCourseProject_files/figure-html/unnamed-chunk-2-1.png)<!-- -->

**3. Building a prediction model**  

I partitioned the training data set into inTraining and inTesting and built a model using randomForest() on the inTraining. The importance of variables was calculated from the model 


```r
library(randomForest)
```

```
## randomForest 4.6-14
```

```
## Type rfNews() to see new features/changes/bug fixes.
```

```
## 
## Attaching package: 'randomForest'
```

```
## The following object is masked from 'package:ggplot2':
## 
##     margin
```

```r
library(caret)

#partition
inTrain <- createDataPartition(y=training$classe, p=0.75, list=FALSE)

inTraining <- training[inTrain, ]
inTesting <- training[-inTrain, ]
dim(inTraining)
```

```
## [1] 14718    53
```

```r
dim(inTesting)
```

```
## [1] 4904   53
```

```r
inTraining$classe <- factor(inTraining$classe)
inTesting$classe <- factor(inTesting$classe)

# ctrl <- trainControl (method="cv", number=5, allowParallel = TRUE)
# 
# mod1 <- train(classe~., data=training, list=FALSE, preProcess=c("center", "scale"), trControl=ctrl) # default method="rf"


rf_classifiers<- randomForest(classe~., data=inTraining, ntree=100, importance=TRUE)
#rf_classifiers$importance

# predict_training <- predict(rf_classifiers, newdata=inTraining)
# confusionMatrix(predict_training, inTraining$classe)
```



**4. Evaluation**  

The random forest model was evaluated on the inTesting dataset. The overall accuracy was .99. 


```r
predict <- predict(rf_classifiers, newdata=inTesting)
confusionMatrix(predict, inTesting$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1394    5    0    0    0
##          B    1  943    9    0    0
##          C    0    1  846    5    0
##          D    0    0    0  799    3
##          E    0    0    0    0  898
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9951          
##                  95% CI : (0.9927, 0.9969)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9938          
##                                           
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9993   0.9937   0.9895   0.9938   0.9967
## Specificity            0.9986   0.9975   0.9985   0.9993   1.0000
## Pos Pred Value         0.9964   0.9895   0.9930   0.9963   1.0000
## Neg Pred Value         0.9997   0.9985   0.9978   0.9988   0.9993
## Prevalence             0.2845   0.1935   0.1743   0.1639   0.1837
## Detection Rate         0.2843   0.1923   0.1725   0.1629   0.1831
## Detection Prevalence   0.2853   0.1943   0.1737   0.1635   0.1831
## Balanced Accuracy      0.9989   0.9956   0.9940   0.9965   0.9983
```





**QUIZ**
[The previous part was modified as I did not predict the quiz well enough and this part is added after the revision]


```r
predict.quiz <- predict(rf_classifiers, newdata=testing)
predict.quiz
```

```
##  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 
##  B  A  B  A  A  E  D  B  A  A  B  C  B  A  E  E  A  B  B  B 
## Levels: A B C D E
```


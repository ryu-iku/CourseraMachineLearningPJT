---
title: "Weight Lifting Exercise Dataset Prediction"
author: "Liu Yu"
output: html_document
---



## Introduction
In this analysis, a weight lifting exercise dataset was analyzed to quantify the exercise. The data are from accelerometers on the belt, forearm, arm, and dumbbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways. More information is available from the website here: http://groupware.les.inf.puc-rio.br/har (see the section on the Weight Lifting Exercise Dataset).

## Data preprocessing
Download files from links provided by the coursera lesson, and read in the file data.

```r
download.file("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv",
              destfile = "training.csv")
download.file("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv", 
              destfile = "pml-testing.csv")
training <- read.csv("pml-training.csv")
testing <- read.csv("pml-testing.csv")
```
Remove unrelated columns from the training data.

```r
library(dplyr)
training <- select(training, 
                   -which(names(training) %in% 
                              c("X",
                                "raw_timestamp_part_1",
                                "raw_timestamp_part_2",
                                "cvtd_timestamp")))
```
Change all the factor values to number for data processing.

```r
training$user_name <- as.numeric(training$user_name)
training$new_window <- as.numeric(training$new_window)
sapply(names(training), function(x){
    if (class(training[,x]) == "factor" & x != "classe"){
        training[,x] <<- as.numeric(as.character(training[,x]))
    }
})
```
Because some columns have too many NA data, they should be removed to make the data processing more efficient.

```r
colWithNa <- apply(training, 2, function(x)sum(is.na(x)))
training <- select(training, 
                   -which(names(training) %in% names(colWithNa[colWithNa > 0])))
```

## Model selection
### Model selection strategy

1) Try several usual model for multiple-classification, with a simple training setting.
2) Check whether any model has an over-95% accuracy. If yes, then choose the the model with highest accuracy as the final model. If no, then continue next step.
3) Choose models with comparatively better accuracy and less time for the next step.
4) Tune parameters of selected models.
5) Choose the tuned model with the highest accuracy as the final model.

### Check models' accuracy and time cost with simple settings.

There are 8 models to be checked:

	Naive Bayes model
	Support Vector Machines with Linear Kernel model (Linear SVM)
	k-Nearest Neighbors model
	Support Vector Machines with Radial Basis Function Kernel model (Radial SVM)
	Bagged Model
	Stochastic Gradient Boosting model
	Random Forest model
	Neural Network model

Naive Bayes model, Linear SVM and k-Nearest Neighbors model, are expected to have less time costs, due to their relatively less complexity. The rest 5 models are expected to have higher accuracies but more time costs.

The R library of ```caret``` will be used in this project. Each model will be trained with the ```train``` method of ```caret```. For simplicity, the default cross validation will be used at first.

```r
# default setting of trainControl for the cross validation:
default.trainCtrl <- trainControl()
default.trainCtrl[1:5]
```

#### Naive Bayes model

```r
library(caret)
modelNb <- train(classe ~ ., data = training, method = "nb")
```


```r
modelNb$times$everything[3] # Time used
```

```
## elapsed 
## 2625.25
```

```r
max(modelNb$results$Accuracy) # Accuracy
```

```
## [1] 0.7639468
```

#### Linear SVM

```r
svmLin <- train(classe ~ ., data = training, method = "svmLinear")
```


```r
svmLin$times$everything[3] # Time used
```

```
## elapsed 
## 1284.13
```

```r
max(svmLin$results$Accuracy) # Accuracy
```

```
## [1] 0.7931265
```

#### k-Nearest Neighbors model

```r
modelKnn <- train(classe ~ ., data = training, method = "knn")
```


```r
modelKnn$times$everything[3] # Time used
```

```
## elapsed 
## 1350.26
```

```r
max(modelKnn$results$Accuracy) # Accuracy
```

```
## [1] 0.9145543
```

#### Radial SVM

```r
modelSvmRadial <- train(classe ~ ., data = training,
                  method = "svmRadial",
                  tuneLength = 9,
                  preProc = c("center","scale"))
```


```r
modelSvmRadial$times$everything[3] # Time used
```

```
##  elapsed 
## 15265.47
```

```r
max(modelSvmRadial$results$Accuracy) # Accuracy
```

```
## [1] 0.9921361
```

#### Bagged Model

```r
modelBag <- train(classe ~ ., data = training,
                  method = "bag",
                  B = 10,
                  bagControl = bagControl(fit = ctreeBag$fit,
                                          predict = ctreeBag$pred,
                                          aggregate = ctreeBag$aggregate))
```


```r
modelBag$times$everything[3] # Time used
```

```
## elapsed 
## 2255.08
```

```r
max(modelBag$results$Accuracy) # Accuracy
```

```
## [1] 0.9693223
```

#### Stochastic Gradient Boosting model

```r
modelGBM <- train(classe ~ ., data = training, method = "gbm")
```


```r
modelGBM$times$everything[3] # Time used
```

```
## elapsed 
## 2453.84
```

```r
max(modelGBM$results$Accuracy) # Accuracy
```

```
## [1] 0.9863236
```
#### Random Forest model

```r
modelRF01 <- train(classe ~ ., data = training, method = "rf")
```


```r
modelRF01$times$everything[3] # Time used
```

```
## elapsed 
## 7181.99
```

```r
max(modelRF01$results$Accuracy) # Accuracy
```

```
## [1] 0.997291
```
#### Neural Network model

```r
modelNnt <- train(classe ~ ., data = training, method = "nnet", maxit = 4000)
```


```r
modelNnt$times$everything[3] # Time used
```

```
##  elapsed 
## 55541.56
```

```r
max(modelNnt$results$Accuracy) # Accuracy
```

```
## [1] 0.680046
```

### Conclusion of model selection
There are 4 models with an accuracy over 95%:

    Radial SVM (Accuracy: 0.9921361)
    Bagged model (Accuracy: 0.9693223)
    Stochastic Gradient Boosting model (Accuracy: 0.9863236)
    Random Forest model (Accuracy: 0.997291)

According to the model selection strategy, Random Forest model is the most accurate and the final model of this analysis.
The expected out-of-sample error of the final model is: 1 - Accuracy = 0.002709.

## Predict results of the testing data with the final model

Preprocess the test data before prediction.

```r
testing$user_name <- as.numeric(testing$user_name)
testing$new_window <- as.numeric(testing$new_window)
sapply(names(testing), function(x){
    if (class(testing[,x]) == "factor" & x != "classe"){
        testing[,x] <<- as.numeric(as.character(testing[,x]))
    }
})
```

Predict results of the testing data with the final model ```modelRF01```.

```r
predict(modelRF01, newdata = testing)
```

```
##  [1] B A B A A E D B A A B C B A E E A B B B
## Levels: A B C D E
```
According to the coursera quiz answers, the prediction accuracy is 100%.

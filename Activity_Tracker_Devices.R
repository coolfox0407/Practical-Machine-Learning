## This assigment is part of coursera's "Data Science: Practical Machine Learning" module
## Week 4 assignment: Weight Lifting Exercises activity from Personal Activity Tracking Devices
## Description: This R script loads "Weight Lifting Exercises" dataset from 
## "pml-training.csv" and "pml-testing.csv" files, pre-processes, cleans data,
## perform exploratory data analysis, design prediction model using ML algorithms

## R File Name: "Activity_Tracker_Devices.R"
## Input Files: "pml-training.csv" and "pml-testing.csv"
## Ouput Files: "Activity_Tracker_Devices.html"

## Date of submission of assigment: 16-September-2017
## GitHub User Name: coolfox0407
## Author: Hariharan D


# Load the necessary R libraries.

# Note: It is assumed that the below R libraries are already installed.

library(caret)
library(rpart)
library(rpart.plot)
library(rattle)
library(randomForest)
library(corrplot)
library(Rmisc)
library(ggplot2)
library(RColorBrewer)

# Load Dataset and Clean the Data

set.seed(12345)

trainURL <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"

testURL <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"

training <- read.csv(url(trainURL), na.strings=c("NA","#DIV/0!",""))

testing <- read.csv(url(testURL), na.strings=c("NA","#DIV/0!",""))

# Creating partition with Training dataset

trainData <- createDataPartition(training$classe, p=0.7, list=FALSE)
trainingSet <- training[trainData, ]
testingSet <- training[-trainData, ]

dim(trainingSet)
dim(testingSet)


# Cleansing the data by removing "Near Zero Variance (NZV)", "NA" and ID variables.

# Removing "NZV"

NZV <- nearZeroVar(trainingSet)
trainingSet <- trainingSet[, -NZV]
testingSet  <- testingSet[, -NZV]

dim(trainingSet)
dim(testingSet)

# Removing "NA"

dataNA <- sapply(trainingSet, function(x) mean( is.na(x) ) ) > 0.95
trainingSet <- trainingSet[, dataNA == FALSE]
testingSet  <- testingSet[, dataNA == FALSE]

dim(trainingSet)
dim(testingSet)

# Removing ID variables

trainingSet <- trainingSet[, -(1:5)]
testingSet  <- testingSet[, -(1:5)]

dim(trainingSet)
dim(testingSet)

# Exploratory Analysis - Correlation Analysis

corMatrix <- cor(trainingSet[, -54])

png("Correlation Plot.png", width = 500, height = 500)

corrplot(corMatrix, order = "FPC", method = "color", type = "lower", tl.cex = 0.6, tl.col = rgb(0, 0, 0))

dev.off()

# The highly correlated variables are shown in dark blue square fills in the correlation plot. Let us now verify the dataspread characteristics visually using GGPLOT. 

# Multi-plot Analysis

p1 <- ggplot(trainingSet, aes(classe, pitch_forearm)) + geom_boxplot(aes(fill=classe))

p2 <- ggplot(trainingSet, aes(classe, magnet_arm_x)) + geom_boxplot(aes(fill=classe))

png("GGPLOT.png", width = 500, height = 500)

multiplot(p1,p2,cols=2)

dev.off()

# From the above plot, we can infer that there is no clear separation of classes.

# 1. Prediction Model - Decision Tree

# Model Fit

set.seed(12345)

fitDecisionTree <- rpart(classe ~ ., data = trainingSet, method="class")

png("Decision Tree_1.png", width = 500, height = 500)

fancyRpartPlot(fitDecisionTree)

dev.off()

# Prediction on Testing Dataset

predictDecisionTree <- predict(fitDecisionTree, newdata = testingSet, type="class")
confMatDecisionTree <- confusionMatrix(predictDecisionTree, testingSet$classe)
confMatDecisionTree

# Plotting Confusion Matrix

png("Decision Tree_2.png", width = 500, height = 500)

plot(confMatDecisionTree$table, col = confMatDecisionTree$byClass, main = paste("Decision Tree Confusion Matrix Accuracy : ", round(confMatDecisionTree$overall['Accuracy'], 4)*100, "%"))

dev.off()

# 2.Prediction Model - Random Forests

# Model fit

set.seed(12345)

controlRF <- trainControl(method = "cv", number = 3, verboseIter = FALSE)
fitRandomForests <- train(classe ~ ., data = trainingSet, method = "rf", trControl = controlRF)
fitRandomForests$finalModel

# Prediction on Testing Dataset

predictRandomForests <- predict(fitRandomForests, newdata = testingSet)
confMatRandomForests <- confusionMatrix(predictRandomForests, testingSet$classe)
confMatRandomForests

# Plotting Confusion Matrix

png("Random Forests.png", width = 500, height = 500)

plot(confMatRandomForests$table, col = confMatRandomForests$byClass, main = paste("Random Forests Confusion Matrix Accuracy : ", round(confMatRandomForests$overall['Accuracy'], 4)*100, "%"))

dev.off()

# 3.Prediction Model - Generalized Boosted Regression

# Model fit

set.seed(12345)

controlGBR <- trainControl(method = "repeatedcv", number = 5, repeats = 1)
fitGBR  <- train(classe ~ ., data = trainingSet, method = "gbm", trControl = controlGBR, verbose = FALSE)
fitGBR$finalModel

# Prediction on Testing Dataset

predictGBR <- predict(fitGBR, newdata = testingSet)
confMatGBR <- confusionMatrix(predictGBR, testingSet$classe)
confMatGBR

# Plotting Confusion Matrix

png("Generalized Boosted Regression.png", width = 500, height = 500)

plot(confMatGBR$table, col = confMatGBR$byClass, main = paste("GBR Confusion Matrix Accuracy : ", round(confMatGBR$overall['Accuracy'], 4)*100, "%"))

dev.off()

# Predicting Results on Test Dataset with Selected Model

# The accuracy of above 3 Prediction Models are as follows;

print(paste("1. Decision Tree:", round(confMatDecisionTree$overall['Accuracy'], 4)*100,"%"),quote = FALSE)

print(paste("2. Random Forests:", round(confMatRandomForests$overall['Accuracy'], 4)*100,"%"),quote = FALSE)

print(paste("3. Generalized Boosted Regression:", round(confMatGBR$overall['Accuracy'], 4)*100,"%"),quote = FALSE)

print(paste("The expected out-of-sample error of Random Forests is", round(100*(1-round(confMatRandomForests$overall['Accuracy'], 4)),2),"%, which is least among all the above selected models. Hence Random Forests model will be used to predict the test dataset."),quote = FALSE)


# Predict Test Data

set.seed(12345)

predictTEST <- predict(fitRandomForests, newdata = testing)
predictTEST


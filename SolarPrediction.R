rm(list = ls())
graphics.off()

#setwd("./Users/Mohan/Documents/SolarPrediction/")
#Load the required libraries
library(MASS)
library(leaps)
library(glmnet)
library(tree)
library(gbm)
library(randomForest)

set.seed(123)

solar_data = read.csv('SolarPrediction.csv')
head(solar_data)

#==========================================
# 1) Exploratory Data Analysis
#==========================================

#==========================================
# 2) Feature Engineering
#==========================================
#  Let`s get more features from the time series variables.

solar_data$Date <- as.Date(solar_data$Date, format = "%m/%d/%Y")
solar_data$month <- as.numeric(format(solar_data$Date, "%m"))

Time1 <- strptime(solar_data$Time,"%H:%M:%S")
solar_data$hour <- as.numeric(format(Time1, "%H"))

Time2 <- strptime(solar_data$TimeSunRise, "%H:%M:%S")
Time3 <- strptime(solar_data$TimeSunSet, "%H:%M:%S")
SunRisesec <- as.numeric(format(Time2, "%H"))*3600 + as.numeric(format(Time2, "%M"))*60 + as.numeric(format(Time2, "%S"))
SunSetsec <- as.numeric(format(Time3, "%H"))*3600 + as.numeric(format(Time3, "%M"))*60 + as.numeric(format(Time3, "%S"))
solar_data$DaylightTime <- (SunSetsec - SunRisesec)/3600

# Drop unnecessary features and seperate response variable.
X = subset(solar_data, select = -c(UNIXTime, Date, Time, TimeSunRise, TimeSunSet))
Y = solar_data$Radiation

head(X)
write.csv(X, file = "radiation.csv")

#===============================================
# 3) Applying Statistical Data Mining Methods
#===============================================

par(mfrow = c(2,2))
#pairs(X)
cor(X)

train = sample(1:nrow(X), nrow(X)*0.65)
train_data = X[train, ]
test_data = X[-train, ]
test_data.Y = test_data$Radiation

#======================
#(i) Linear Regression
#======================

lm.fit <- lm(Radiation ~ ., data = train_data)
summary(lm.fit)
#From the summary, we can see that there is a relationship between the predictors and response 
#as the F-Statistic is far from 1 (with a small p-value) indicating evidence against the null hypothesis.

#Looking at the p-values associated with each predictor's t-statistic, we see that all the predictors have
#statistically significant relationship with the response variable "Radiation".

#The regression coefficient for temperature suggests that for every 1 degree increase in temperature(in fahrenheit)
#the solar radiation will on average increase by 44 in value. 

lm.pred <- predict(lm.fit, newdata = test_data)
lm.mse <- mean((lm.pred - test_data.Y)^2)
lm.mse
#MSE is 38230.24

par(mfrow = c(2,2))
plot(lm.fit)

#Looking at the residual vs fitted plot, we can see that the spread of the residuals is larger in the middle of
#fitted values(predicted values). So the spread is not constant => non constant variance of error terms.
#Hence the data suffers from Heteroscedasticity. Also, the curve shape may indicate non-linearity in the data 

#=======================================
#(ii) Best subset selection (exhaustive)
#=======================================

regfit.full <- regsubsets(Radiation ~ ., data = train_data, nvmax = ncol(train_data)-1)
reg.summary <- summary(regfit.full)
windows()
plot(reg.summary$rss, xlab="Subset Size", ylab="RSS", col = "red")
which.min(reg.summary$rss)
windows()
plot(reg.summary$cp, xlab="Subset Size", ylab="Mallows Cp", col = "red")
which.min(reg.summary$cp)
windows()
plot(reg.summary$bic, xlab="Subset Size", ylab="BIC", col = "red")
which.min(reg.summary$bic)

#Lets choose the best subset based on Mallows' Cp. 
#We have minimum Cp for subset size 8, hence all variables are included in the best subset.
coef(regfit.full, which.min(reg.summary$cp))

#===============================================================
#(iii) Ridge Regression with lambda chosen from cross validation
#===============================================================

cv_train_X <- model.matrix(Radiation ~ ., data = train_data)[, -1]
cv_train_Y <- train_data$Radiation

cv_test_X <- model.matrix(Radiation ~ ., data = test_data)[, -1]
cv_test_Y <- test_data$Radiation

# Get lambda from cv using train data
cv.ridge <- cv.glmnet(cv_train_X, cv_train_Y, alpha = 0)
windows()
plot(cv.ridge, main = "Ridge model - MSE vs lambda")
ridge_lambda <- cv.ridge$lambda.min
ridge_lambda

#Apply Ridge regression with that lambda on test data
ridge.fit <- glmnet(cv_train_X, cv_train_Y, alpha = 0)
coef(ridge.fit)
windows()
plot(ridge.fit, xvar="lambda",label=TRUE, main = "Ridge")
ridge.pred <- predict(ridge.fit, s = ridge_lambda, newx = cv_test_X, type = "response")
ridge.mse <- mean((ridge.pred - cv_test_Y)^2)
ridge.mse
#Ridge MSE is 39159.89

#==============================================================
#(iv) Lasso Regression with lambda chosen from cross validation
#==============================================================

cv.lasso <- cv.glmnet(cv_train_X, cv_train_Y, alpha = 1)
windows()
plot(cv.lasso, main = "Lasso model - MSE vs lambda")
lasso_lambda <- cv.lasso$lambda.min
lasso_lambda

#Apply Lasso regression with that lambda on test data
lasso.fit <- glmnet(cv_train_X, cv_train_Y, alpha = 1)
coef(lasso.fit)
windows()
plot(lasso.fit, xvar="lambda",label=TRUE, main = "Lasso")
lasso.pred <- predict(lasso.fit, s = lasso_lambda, newx = cv_test_X, type = "response")
lasso.mse <- mean((lasso.pred - cv_test_Y)^2)
lasso.mse
#Lasso MSE is 38236.44

lasso.pred.coeff <- predict(lasso.fit, s = lasso_lambda, newx = cv_test_X, type = "coefficients")
lasso.pred.coeff
rownames(lasso.pred.coeff)[which(lasso.pred.coeff != 0)]

#There is no improvement when lasso, ridge is used over OLS. This can be because of the below reasoning.
#When there is a problem of overfitting in least squares fit, regularized models perform well on test set
#as they reduce variance (by shrinking the coefficients).
#However, overfitting is not a problem for this dataset as n >> p(=> low variance). Hence OLS will perform
#well on test observations implying that there is no need for regularization.

#==================
#(v) Decision Trees
#==================

tree.fit <- tree(Radiation ~ ., data = train_data)
summary(tree.fit)
windows()
plot(tree.fit)
text(tree.fit)

cv.tree.fit <- cv.tree(tree.fit)
windows()
plot(cv.tree.fit$size, cv.tree.fit$dev, type = 'b', col = "red")

prune.tree.fit <- prune.tree(tree.fit, best = 5)
windows()
plot(prune.tree.fit)
text(prune.tree.fit)

tree.pred <- predict(tree.fit, newdata = test_data)
tree.mse <- mean((tree.pred - test_data.Y)^2)
tree.mse
windows()
plot(tree.pred, test_data.Y)

prune.tree.pred <- predict(prune.tree.fit, newdata = test_data)
prune.tree.mse <- mean((prune.tree.pred - test_data.Y)^2)
prune.tree.mse

#===================
#(vi) Boosting
#===================

boost.fit <- gbm(Radiation ~ ., data = train_data, distribution = "gaussian", n.trees = 3000, interaction.depth = 8)
summary(boost.fit)
boost.pred <- predict(boost.fit, newdata = test_data, n.trees = 3000)
boost.mse <- mean((boost.pred - test_data.Y)^2)
boost.mse

#===================
#(vii) Bagging
#===================

bag.fit <- randomForest(Radiation ~ ., data = train_data, mtry = ncol(train_data)-1, ntree = 500)
bag.pred <- predict(bag.fit, test_data)
bag.mse <- mean((bag.pred - test_data.Y)^2)
bag.mse

#===================
#(viii) RandomForest
#===================

rf.fit <- randomForest(Radiation ~ ., data = train_data, ntree = 500, importance = TRUE)
rf.pred <- predict(rf.fit, test_data)
rf.mse <- mean((rf.pred - test_data.Y)^2)
rf.mse

windows()
partialPlot(rf.fit, pred.data = train_data, x.var = "hour")
windows()
partialPlot(rf.fit, pred.data = train_data, x.var = "Temperature")

#=================================================================================
#(ix) Comparision of all the statistical data mining models we have applied so far
#=================================================================================

all_mse <- c(lm.mse, ridge.mse, lasso.mse, tree.mse, boost.mse, bag.mse, rf.mse)
model_names <- c("OLS", "Ridge", "Lasso", "Regression tree", "Boosting", "Bagging", "RandomForests")
barplot(all_mse, names.arg = model_names, ylab = "MSE",col = "green")

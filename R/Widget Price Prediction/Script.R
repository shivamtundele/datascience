#Console Output Open
sink("Logs/ConsoleOutput.txt")

#########################################################################################
#LOADING LIBRARIES
#########################################################################################

library(plyr)
library(caret)
library(Hmisc)
library(mboost)
library(mlbench)
library(lattice)
library(ggplot2)
library(data.table)
library(corrplot)

#########################################################################################
#DATA OVERVIEW 
#########################################################################################

#Loading Data
data.initial <- fread("widgets.csv")

#Structure of Data
str(data.initial)

#Summary of Data
summary(data.initial)

#Descriptive Statistical Summary of Data
describe(data.initial)

#List of Rows of Data With Missing Values
data.missing <- data.initial[!complete.cases(data.initial),]
data.missing

#Removing Rows with Negative Values From Price
data.initial <- data.initial[data.initial$price != "-99", ]

#########################################################################################
#DATA ANALYSIS VIA MANIPULATION
#########################################################################################

#Mean Price of Widget (With NAs)
mean(data.initial$price) 

#Mean Price of Widget (Without NAs)
mean(data.initial$price, na.rm=TRUE)


#Most Sold Construction Material
describe(data.initial$construction)

#Greatest Mean Price Construction Material
ddply(data.initial, .(construction), plyr :: summarize,  Mean=mean(price))

#Greatest Max Price Of Combination Of Construction Material And Style
compute.combination <- ddply(data.initial, .(construction,style), plyr :: summarize,  Mean=mean(price))
max(compute.combination$Mean)
compute.combination

#Correlation Coefficient Between Widget Size And Height
cor.test(data.initial$size, data.initial$height, method = "pearson", use="complete.obs")


#########################################################################################
#DATA VISUALIZATION
#########################################################################################

#Scatter Plot of Widget Size And Price
fig.scatterplot <- ggplot(data.initial, aes(size, price))
fig.scatterplot + geom_point()

png("R Visualization/ScatterplotA.png",width=2000, height=2000, res=300)
fig.scatterplot + geom_point(aes(shape = factor(quality))) + scale_shape(solid = FALSE)
dev.off()

png("R Visualization/ScatterplotB.png",width=2000, height=2000, res=300)
fig.scatterplot + geom_point(aes(colour = quality))
dev.off()


#Histogram Of Widget Height
png("R Visualization/HistogramA.png",width=2000, height=2000, res=300)
ggplot(data=data.initial, aes(data.initial$weight, label=(data.initial$construction))) + 
  geom_histogram(breaks=seq(20, 50, by = 2), col="red", fill="pink", 
                 alpha = .2) + labs(title="Widget Weight") + labs(y="Frequency", x="Weight (gm)") 
dev.off()

#Histogram Of Widget Height By Construction Type
#histogram( ~ weight | construction, data=data.initial, ylab="Frequency", xlab="Weight (gm)",
  #         main="Widget Weight", layout=c(7,1))

png("R Visualization/HistogramB.png",width=2000, height=2000, res=300)
histogram( ~ weight | construction, data=data.initial, ylab="Frequency", xlab="Weight (gm)",
           main="Widget Weight")
dev.off()

#layout(matrix(1), widths = lcm(12), heights = lcm(12))

#Box Plot Of Prices By Widget Style
function.median <- function(x)
  {
    return(data.frame(y=median(x),label=median(x,na.rm=T)))
  }

png("R Visualization/BoxPlotA.png",width=2000, height=2000, res=300)
ggplot(data.initial, aes(x = style, y = price, fill = style)) + geom_boxplot() + 
  labs(title=" box plot of prices by widget style") +
  labs(y="Price ($USD)", x="Widget Style") +
  stat_summary(fun.y = median, geom="point",colour="black", size=3) +
  stat_summary(fun.data = function.median, geom="text", vjust=-0.7)
dev.off()

#########################################################################################
#DATA TRANSFORMATION
#########################################################################################

#Creating Intermediate Data Frame For Transformation
data.transform <- data.frame(data.initial)

#Removing Useless Variables
data.transform$widget_id <- NULL
data.transform$construction <- as.factor(data.transform$construction)
data.transform$quality <- as.factor(data.transform$quality)
data.transform$style <- as.factor(data.transform$style)

#Removing NAs and Creating Final Data Frame
data.final <- na.omit(data.transform)

#Reproducible Result
set.seed(28262)

#Sample Indexes
indexes = sample(1:nrow(data.final), size=0.2*nrow(data.final))

#Splitting Data Into Train (80%) and Test (20%) Data Frames
data.train = data.final[-indexes,]
data.test = data.final[indexes,]

#Correlation Between Variables
data.correlation <- cor(data.final[,c("height","price","size","weight", "zip")], method = "pearson")
data.correlation

png("R Visualization/Correlation.png",width=2000, height=2000, res=300)
corrplot(data.correlation, method="number")
dev.off()

#########################################################################################
#DATA MODELLING
#########################################################################################

#======================================================================
#LINEAR MODEL (DEFAULT R LIBRARY)
#======================================================================
# layout(matrix(1), widths = lcm(5), heights = lcm(5))

#Reproducible Result
set.seed(28262)

#Model 1
model.lm.1 <- lm(formula = price~., data = data.train) 
summary(model.lm.1)

par(mfrow = c(1,1))
png('R Visualization/LinearModel01%03d.png', width=2000, height=2000, res=300)
plot(model.lm.1, 1:6, ask = FALSE)
dev.off()

model.lm.1.varImp <- varImp(model.lm.1)
model.lm.1.varImp

#Predict via Model 1
predict.lm.1<-predict(model.lm.1,data.test) 
predict.lm.1.modelvalues<-data.frame(obs = data.test$price, pred=predict.lm.1)
defaultSummary(predict.lm.1.modelvalues)

#Model 2
model.lm.2 <- lm(formula = price ~ style + quality + height + size + construction, data = data.train)
summary(model.lm.2)

par(mfrow = c(1,1))
png('R Visualization/LinearModel02%03d.png', width=2000, height=2000, res=300)
plot(model.lm.1, 1:6, ask = FALSE)
dev.off()

model.lm.2.varImp <- varImp(model.lm.2)
model.lm.2.varImp

#Predict via Model 2
predict.lm.2 <- predict(model.lm.2, data.test)  
predict.lm.2.modelvalues<-data.frame(obs = data.test$price, pred=predict.lm.2)
defaultSummary(predict.lm.2.modelvalues)

#Comparison of Original Price And Predicted Price Values
predict(model.lm.2, data.test, se.fit = TRUE)
predict.w.plim <- predict(model.lm.2, data.test, interval = "prediction")

png('R Visualization/LinearModel.PredVsObs.png', width=2000, height=2000, res=300)
matplot(data.test$price,predict.w.plim,lty = c(1,2,2,3,3), type = "l", ylab = "Predicted Price", xlab = "Original Price")
dev.off()

#======================================================================
#COMPARISON OF LINEAR AND GBM MODEL (CARET LIBRARY)
#======================================================================

#Prepare Training Schema
control <- trainControl(method="repeatedcv", number=10, repeats=3)

#Caret Linear Model Train
set.seed(28262)
model.caret.Lm <- train(price ~ style + quality + height + size + construction, data=data.train, method="lm", trControl=control)
model.caret.Lm

#Caret Linear Model Predict
predict.caret.Lm <- predict(model.caret.Lm, newdata = data.test)
predict.caret.Lm.modelvalues <- data.frame(obs = data.test$price, pred=predict.caret.Lm)
defaultSummary(predict.caret.Lm.modelvalues)

#Caret GBM Model
set.seed(28262)
model.caret.Gbm <- train(price ~ style + quality + height + size + construction, data=data.train, method="gbm", 
                         trControl=control, verbose=FALSE)
model.caret.Gbm

#Caret GBM Model Predict
predict.caret.Gbm <- predict(model.caret.Gbm, newdata = data.test)
predict.caret.Gbm.modelvalues <- data.frame(obs = data.test$price, pred=predict.caret.Gbm)
defaultSummary(predict.caret.Gbm.modelvalues)

#Collecting Resamples
set.seed(28262)
model.caret.results <- resamples(list(LM=model.caret.Lm, GBM=model.caret.Gbm))

#Summary
summary(model.caret.results)

#BoxPlot Of Results
png('R Visualization/CaretModelsA.PredVsObs.png', width=2000, height=2000, res=300)
bwplot(model.caret.results)
dev.off()

#DotPlot Of Results
png('R Visualization/CaretModelsB.PredVsObs.png', width=2000, height=2000, res=300)
dotplot(model.caret.results)
dev.off()


#Print Results
print(model.caret.results$values)

#======================================================================
#RANDOMFOREST MODEL (RANDOMFOREST LIBRARY)
#======================================================================
library(rattle)
write.csv(data.train, "train.csv")
write.csv(data.test, "test.csv")
rattle()

#Console Output Close
sink()

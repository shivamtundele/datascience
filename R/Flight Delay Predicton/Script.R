#Console Output Open
sink("ConsoleOutput.txt")
#----------------------------------------------------------------------------------------------------------------------------
# LOAD LIBRARIES
#----------------------------------------------------------------------------------------------------------------------------
library(data.table)
library(ggplot2)
library(Hmisc)
library(corrplot)
library(Hmisc)
library(Rcmdr)
library(caret)
library(randomForest)

#----------------------------------------------------------------------------------------------------------------------------
# LOAD DATA
#----------------------------------------------------------------------------------------------------------------------------
flights <- fread("Dataset/Flights.csv")

#----------------------------------------------------------------------------------------------------------------------------
# DATA OVERVIEW
#----------------------------------------------------------------------------------------------------------------------------
#Structure of Data
str(flights) 
#Summary of Data
summary(flights) 

#----------------------------------------------------------------------------------------------------------------------------
# DATA TRANSFORMATION
#----------------------------------------------------------------------------------------------------------------------------
#Removing Empty Column
flights$V25 <- NULL 

#Stripping Date Column into Year, Month, Day and Day of Week.
flights$DATE <- as.Date(as.character(flights$FL_DATE), format='%Y-%m-%d')
flights$YEAR <- format(flights$DATE, '%Y')
flights$MONTH <- format(flights$DATE, '%m')
flights$DAY <- format(flights$DATE, '%d')
flights$DAY_OF_WEEK <- weekdays(as.Date(flights$DATE))

#Structure of new dataframe
str(flights)

#Transformation of data
flights.transformed <- data.frame(flights)
flights.transformed$DayofMonth <- as.integer(flights$DAY)
flights.transformed$DayofWeek <- as.factor(flights$DAY_OF_WEEK)
flights.transformed$DepTime <- as.integer(flights$DEP_TIME)
flights.transformed$CRSDepTime <- as.integer(flights$CRS_DEP_TIME)
flights.transformed$ArrTime <- as.integer(flights$ARR_TIME)
flights.transformed$CRSArrTime <- as.integer(flights$CRS_ARR_TIME)
flights.transformed$UniqueCarrier <- as.factor(flights$CARRIER)
flights.transformed$FlightNum <- as.integer(flights$FL_NUM)
flights.transformed$Origin <- as.factor(flights$ORIGIN)
flights.transformed$Dest <- as.factor(flights$DEST)
flights.transformed$Dist <- as.integer(flights$DISTANCE)
flights.transformed$TaxiOut <- as.integer(flights$TAXI_OUT)
flights.transformed$TaxiIn <- as.integer(flights$TAXI_IN)
flights.transformed$WheelsOff <- as.integer(flights$WHEELS_OFF)
flights.transformed$WheelsOn <- as.integer(flights$WHEELS_ON)
flights.transformed$Diverted  <- as.integer(flights$DIVERTED)
flights.transformed$Distance <- as.integer(flights$DISTANCE)
flights.transformed$Cancelled <- as.integer(flights$CANCELLED)
flights.transformed$CancellationCode <- as.factor(flights$CANCELLATION_CODE)
flights.transformed$CarrierDelay <- as.integer(flights$CARRIER_DELAY)
flights.transformed$WeatherDelay <- as.integer(flights$WEATHER_DELAY)
flights.transformed$NASDelay <- as.integer(flights$NAS_DELAY)
flights.transformed$SecurityDelay <- as.integer(flights$SECURITY_DELAY)

#Removing variables not of use
flights.final <- flights.transformed[-c(1:29)]
flights.final <- flights.final[-c(9,10)]

#Removing NA values
flights.final <- na.omit(flights.final)

#----------------------------------------------------------------------------------------------------------------------------
# DATA ANALYSIS
#----------------------------------------------------------------------------------------------------------------------------

#Plot of flights by Carriers
png("R Visualization/FlightsByCarrier.png",width=2000, height=2000, res=300)
carrier=data.frame(flights.final$UniqueCarrier)
qplot(x=flights.final$UniqueCarrier, data=carrier, fill=flights.final$UniqueCarrier)
dev.off()

#Analyzing data precisely
describe(flights.final)
summary(flights.final$DepTime)
summary(flights.final$CarrierDelay)
summary(flights.final$Origin)
summary(flights.final$Dest)
summary(flights.final$Dist)

#Correlation
flights.correlation <- cor(flights.final[,c("DayofMonth","DepTime","CRSDepTime","ArrTime",
                                                  "CRSArrTime","FlightNum","Dist","TaxiOut",
                                                  "TaxiIn","WheelsOff","WheelsOn","Distance",
                                                  "CarrierDelay","WeatherDelay","NASDelay"
                                                  ,"SecurityDelay")])


print(flights.correlation)

#Plot of Correlation
png("R Visualization/CorrelationPlot.png",width=2000, height=2000, res=300)
corrplot(flights.correlation, method = "circle")
dev.off()

#Adjusted P-values
flights.pvalue <- rcorr.adjust(flights.final[,c("DayofMonth","DepTime","CRSDepTime","ArrTime",
                                                "CRSArrTime","FlightNum","Dist","TaxiOut",
                                                "TaxiIn","WheelsOff","WheelsOn","Distance",
                                                "CarrierDelay","WeatherDelay","NASDelay"
                                                ,"SecurityDelay")],type="pearson")
print(flights.pvalue)

#----------------------------------------------------------------------------------------------------------------------------
# DATA MODELLING
#----------------------------------------------------------------------------------------------------------------------------
#Division of dataframe into train and test data frames
# Train = 70% of total data
# Test = Remaining 30% of total data
nrows <- nrow(flights.final)

#Train dataframe
flights.train <- flights.final[1:((nrows)*.7),]
flights.train.x <- flights.train[,c(-20)]
flights.train.y <- flights.train[,c(20)]

#Test dataframe
flights.test <- flights.final[((nrows)*.7):nrows,]
flights.test.x <- flights.test[,c(-20)]
flights.test.y <- flights.test[,c(20)]


#Function to compute Precision, Recall and F1-Measure
get_metrics <- function(predicted, actual) {
  tp = length(which(predicted == TRUE & actual == TRUE))
  tn = length(which(predicted == FALSE & actual == FALSE))
  fp = length(which(predicted == TRUE & actual == FALSE))
  fn = length(which(predicted == FALSE & actual == TRUE))
  
  precision = tp / (tp+fp)
  recall = tp / (tp+fn)
  F1 = 2*precision*recall / (precision+recall)
  accuracy = (tp+tn) / (tp+tn+fp+fn)
  
  v = c(precision, recall, F1, accuracy)
  v
}

#Reproducible results
set.seed(28262)

#RandomForest Modelling
flight.rf.model <- randomForest(flights.train.x, flights.train.y,
                             mtry=2, keep.forest=TRUE, importance=TRUE,do.trace=TRUE, ntree=100)

#Summary of RandomForest Model
summary(flight.rf.model)

#Error plot of RandomForest Model
png("R Visualization/RandomForest.png",width=2000, height=2000, res=300)
plot(flight.rf.model, log="y")
dev.off()

#Variable Importance Plot of RandomForest model
png("R Visualization/VariableImportance.png",width=2000, height=2000, res=300)
varImpPlot(flight.rf.model)
dev.off()

#RandomForest Prediction
flight.rf.pred <- predict(flight.rf.model, newdata=flights.test.x, type="response")

#Evaluating precision of generated model
m.rf = get_metrics(as.logical(flight.rf.pred), as.logical(flights.test.y))
print(sprintf("Random Forest: precision=%0.2f, recall=%0.2f, F1=%0.2f, accuracy=%0.2f", m.rf[1], m.rf[2], m.rf[3], m.rf[4]))

#Console Output Close
sink()

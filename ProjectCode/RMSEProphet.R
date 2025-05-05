
# Load the dataset for IBM stock
IBM_Goldmine <- read.csv("~/Desktop/Project Machine Learning/GoldmineIBM.csv")

# Load the dplyr library for data manipulation
library(dplyr)

# Rename columns to fit Prophet's expected format
IBM_Goldmine <- IBM_Goldmine %>% rename(ds = DATE)
IBM_Goldmine <- IBM_Goldmine %>% rename(y = avg_buy_price_LR)

# Set the logistic growth cap and fit the Prophet model
IBM_Goldmine$cap <- 240
m5 <- prophet(IBM_Goldmine, yearly.seasonality = TRUE, weekly.seasonality = TRUE)

```


# Create a dataframe with 500 future days and forecast
future5 <- make_future_dataframe(m5, periods = 500)
future5$cap <- 240
fcst <- predict(m5, future5)

# Plot forecasted results
plot(m5, fcst)

#Closing price/price forecast 

# View the last few predicted values from the forecast
tail(fcst)



# Perform cross-validation on the Prophet model
future5 <- cross_validation(
  m5, 
  horizon = 30,       # Forecast 30 days ahead
  units = 'days', 
  initial = 365,      # Use first 365 days for initial training
  period = 90         # Perform updates every 90 days
)

# Reassign cap and predict again after cross-validation (possible redundancy)
future8$cap <- 240
fcst <- predict(m5, future5)

# Plot forecast after cross-validation
plot(m5, fcst)



# Evaluate forecast performance metrics such as RMSE, MAE, etc.
PerformanceIBM <- performance_metrics(future5)

# Display the first few rows of the performance metrics
head(PerformanceIBM)

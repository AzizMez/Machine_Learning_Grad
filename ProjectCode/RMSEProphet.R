
IBM_Goldmine <- read.csv("~/Desktop/Project Machine Learning/GoldmineIBM.csv")



library(dplyr)
IBM_Goldmine <- IBM_Goldmine %>% rename(ds = DATE)
IBM_Goldmine <-IBM_Goldmine %>% rename(y = avg_buy_price_LR)

IBM_Goldmine$cap <- 240
m5 <- prophet(IBM_Goldmine, yearly.seasonality = TRUE, weekly.seasonality = TRUE)
```


future5 <- make_future_dataframe(m5, periods = 500)
future5$cap <- 240
fcst <- predict(m5, future5)
plot(m5, fcst)


Closing price/price forecast 

tail(fcst)



future5 <- cross_validation(
  m5, 
  horizon = 30,
  units = 'days',
  initial = 365,
  period = 90)
future8$cap <- 240
fcst <- predict(m5, future5)
plot(m5, fcst)



PerformanceIBM<- performance_metrics(future5)
head(PerformanceIBM)

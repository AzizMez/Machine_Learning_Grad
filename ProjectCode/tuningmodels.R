data <- read.csv('./GoldmineIBM.csv')

install.packages('prophet')
library(dplyr)
library(lubridate)
library(caret)
library(xgboost)
library(forecast)
library(glmnet)
library(randomForest) 
library(prophet)


data$DATE <- as.Date(data$DATE)
data <- data[, -which(sapply(data, class) == "character")]

new_resp <- rep(NA, nrow(data))
window <- 30

for(i in 1:nrow(data)){
  
  id <- which.min(abs(data$DATE -  data$DATE[i] - window))
  new_resp[i] <- data$avg_buy_price_LR[id]
}

data$price_30_days_out <- new_resp


train <- data %>% filter(DATE < "2020-01-01")
test <- data %>% filter(DATE >= "2020-01-01")

train_x <- train %>% select(-price_30_days_out, -DATE)
train_y <- train$price_30_days_out

test_x <- test %>% select(-price_30_days_out, -DATE)
test_y <- test$price_30_days_out

#linear
lm1 <- lm(price_30_days_out ~ ., # Set formula
          data = train)
summary(lm1)

predictions1 <- predict(lm1, test)
accuracy(test$price_30_days_out, predictions1)

#backward selection
lm_bwd <- step(lm1, direction='backward', k=log(nrow(train)))
summary(lm_bwd)

predictions2 <- predict(lm_bwd, test)
accuracy(test$price_30_days_out, predictions2)


#random forest

set.seed(111111) # Set random number generator seed for reproducability
# Use random forest to do bagging
bag_mod <- randomForest(price_30_days_out ~., # Set tree formula
                        data = na.omit(train[,2:182]), # Set dataset
                        mtry = 181, # Set mtry to number of variables 
                        ntree = 200) # Set number of trees to use
bag_mod

# Careful this can take a long time to run
trees <- c(10, 25, 50, 100, 200, 500, 1000) # Create vector of possible tree sizes
nodesize <- c(1, 10, 25, 50, 100, 200, 500, 1000) # Create vector of possible node sizes

params <- expand.grid(trees, nodesize) # Expand grid to get data frame of parameter combinations
names(params) <- c("trees", "nodesize") # Name parameter data frame
res_vec <- rep(NA, nrow(params)) # Create vector to store accuracy results

for(i in 1:nrow(params)){ # For each set of parameters
  set.seed(111111) # Set seed for reproducability
  print(i)
  mod <- randomForest(price_30_days_out ~., # Set tree formula
                      data = na.omit(train[,2:182]), # Set dataset
                      mtry = 181, # Set number of variables
                      importance = FALSE,  # 
                      ntree = params$trees[i], # Set number of trees
                      nodesize = params$nodesize[i],
                      do.trace = TRUE) # Set node size
  res_vec[i] <- mod$mse[length(mod$mse)] # Calculate out of bag accuracy
}

res_db <- cbind.data.frame(params, res_vec) # Join parameters and accuracy results
names(res_db)[3] <- "oob_error" # Name accuracy results column
res_db # Print accuracy results column

res_db$trees <- as.factor(res_db$trees) # Convert tree number to factor for plotting
res_db$nodesize <- as.factor(res_db$nodesize) # Convert node size to factor for plotting
g_rf <- ggplot(res_db, aes(y = trees, x = nodesize, fill = oob_error)) + # set aesthetics
  geom_tile() + # Use geom_tile for heatmap
  theme_bw() + # Set theme
  scale_fill_gradient2(low = "blue", # Choose low color
                       mid = "white", # Choose mid color
                       high = "red", # Choose high color
                       midpoint =mean(res_db$oob_accuracy), # Choose mid point
                       space = "Lab", 
                       na.value ="grey", # Choose NA value
                       guide = "colourbar", # Set color bar
                       aesthetics = "fill") + # Select aesthetics to apply
  labs(x = "Node Size", y = "Number of Trees", fill = "OOB Error") # Set labels
g_rf # Generate plot

res_db[which.min(res_db$oob_error),]

set.seed(111111)
best_rf <- randomForest(price_30_days_out ~., # Set tree formula
                          data = na.omit(train[,2:182]), # Set dataset
                          mtry = 181, # Set number of variables 
                          ntree = 200, # Set number of trees
                          nodesize = 1) # Set node size

rf_preds <- predict(best_rf, test) # Create predictions for test data
library(Metrics)
rmse(test$price_30_days_out[!is.na(rf_preds)], rf_preds[!is.na(rf_preds)])

#xgboost
dtrain <- as.matrix(train_x)
dtest <- as.matrix(test_x)
set.seed(111111)
bst_1 <- xgboost(
  data = dtrain,
  label = train_y,
  nrounds = 100,
  verbose = 1,
  print_every_n = 20
)

preds <- predict(bst_1, dtest)
rmse(actual, preds)

###### 1 - Tune max depth and min child weight ######


# Be Careful - This can take a very long time to run
max_depth_vals <- c(3, 5, 7, 10, 15) # Create vector of max depth values
min_child_weight <- c(1,3,5,7, 10, 15) # Create vector of min child values

# Expand grid of parameter values
cv_params <- expand.grid(max_depth_vals, min_child_weight)
names(cv_params) <- c("max_depth", "min_child_weight")
# Create results vector
rmse_vec  <- rep(NA, nrow(cv_params)) 
# Loop through results
for(i in 1:nrow(cv_params)){
  set.seed(111111)
  bst_tune <- xgb.cv(data = dtrain,
                     label = train_y,
                     
                     nfold = 5, # Use 5 fold cross-validation
                     
                     eta = 0.1, # Set learning rate
                     max.depth = cv_params$max_depth[i], # Set max depth
                     min_child_weight = cv_params$min_child_weight[i], # Set minimum number of samples in node to split
                     
                     
                     nrounds = 100, # Set number of rounds
                     early_stopping_rounds = 20, # Set number of rounds to stop at if there is no improvement
                     
                     verbose = 1, # 1 - Prints out fit
                     nthread = 1, # Set number of parallel threads
                     print_every_n = 20 # Prints out result every 20th iteration
                     
  ) # Set evaluation metric to use
  
  rmse_vec[i] <- bst_tune$evaluation_log$test_rmse_mean[bst_tune$best_ntreelimit]
  
  
}


 # Join results in dataset
res_db <- cbind.data.frame(cv_params, rmse_vec)
names(res_db)[3] <- c("rmse") 
res_db$max_depth <- as.factor(res_db$max_depth) # Convert tree number to factor for plotting
res_db$min_child_weight <- as.factor(res_db$min_child_weight) # Convert node size to factor for plotting
# Print AUC heatmap
g_2 <- ggplot(res_db, aes(y = max_depth, x = min_child_weight, fill = rmse)) + # set aesthetics
  geom_tile() + # Use geom_tile for heatmap
  theme_bw() + # Set theme
  scale_fill_gradient2(low = "blue", # Choose low color
                       mid = "white", # Choose mid color
                       high = "red", # Choose high color
                       midpoint =mean(res_db$rmse), # Choose mid point
                       space = "Lab", 
                       na.value ="grey", # Choose NA value
                       guide = "colourbar", # Set color bar
                       aesthetics = "fill") + # Select aesthetics to apply
  labs(x = "Minimum Child Weight", y = "Max Depth", fill = "RMSE") # Set labels
g_2 # Generate plot




###### 2 - Gamma Tuning ######


gamma_vals <- c(0, 0.05, 0.1, 0.15, 0.2) # Create vector of gamma values

# Be Careful - This can take a very long time to run
set.seed(111111)
rmse_vec  <- rep(NA, length(gamma_vals))
for(i in 1:length(gamma_vals)){
  bst_tune <- xgb.cv(data = dtrain, # Set training data
                     label = train_y,
                     nfold = 5, # Use 5 fold cross-validation
                     
                     eta = 0.1, # Set learning rate
                     max.depth = 10, # Set max depth
                     min_child_weight = 5, # Set minimum number of samples in node to split
                     gamma = gamma_vals[i], # Set minimum loss reduction for split
                     
                     
                     
                     nrounds = 100, # Set number of rounds
                     early_stopping_rounds = 20, # Set number of rounds to stop at if there is no improvement
                     
                     verbose = 1, # 1 - Prints out fit
                     nthread = 1, # Set number of parallel threads
                     print_every_n = 20 # Prints out result every 20th iteration
  ) # Set evaluation metric to use
  
  
  rmse_vec[i] <- bst_tune$evaluation_log$test_rmse_mean[bst_tune$best_ntreelimit]
  
  
}

# Lets view our results to identify the value of gamma to use:

# Gamma results
# Join gamma to values
cbind.data.frame(gamma_vals, rmse_vec)


###### 3 - Subsample and Column sample Tuning ######

# Be Careful - This can take a very long time to run
subsample <- c(0.6, 0.7, 0.8, 0.9, 1) # Create vector of subsample values
colsample_by_tree <- c(0.6, 0.7, 0.8, 0.9, 1) # Create vector of col sample values

# Expand grid of tuning parameters
cv_params <- expand.grid(subsample, colsample_by_tree)
names(cv_params) <- c("subsample", "colsample_by_tree")
# Create vectors to store results
rmse_vec <- rep(NA, nrow(cv_params)) 
# Loop through parameter values
for(i in 1:nrow(cv_params)){
  set.seed(111111)
  bst_tune <- xgb.cv(data = dtrain, # Set training data
                     label = train_y,
                     nfold = 5, # Use 5 fold cross-validation
                     
                     eta = 0.1, # Set learning rate
                     max.depth = 10, # Set max depth
                     min_child_weight = 5, # Set minimum number of samples in node to split
                     gamma = 0, # Set minimum loss reduction for split
                     subsample = cv_params$subsample[i], # Set proportion of training data to use in tree
                     colsample_bytree = cv_params$colsample_by_tree[i], # Set number of variables to use in each tree
                     
                     nrounds = 150, # Set number of rounds
                     early_stopping_rounds = 20, # Set number of rounds to stop at if there is no improvement
                     
                     verbose = 1, # 1 - Prints out fit
                     nthread = 1, # Set number of parallel threads
                     print_every_n = 20 # Prints out result every 20th iteration
  ) # Set evaluation metric to use
  
  
  rmse_vec[i] <- bst_tune$evaluation_log$test_rmse_mean[bst_tune$best_ntreelimit]
  
  
}







# visualise tuning sample params

res_db <- cbind.data.frame(cv_params, rmse_vec)
names(res_db)[3] <- c("rmse") 
res_db$subsample <- as.factor(res_db$subsample) # Convert tree number to factor for plotting
res_db$colsample_by_tree <- as.factor(res_db$colsample_by_tree) # Convert node size to factor for plotting
g_4 <- ggplot(res_db, aes(y = colsample_by_tree, x = subsample, fill = rmse)) + # set aesthetics
  geom_tile() + # Use geom_tile for heatmap
  theme_bw() + # Set theme
  scale_fill_gradient2(low = "blue", # Choose low color
                       mid = "white", # Choose mid color
                       high = "red", # Choose high color
                       midpoint =mean(res_db$rmse), # Choose mid point
                       space = "Lab", 
                       na.value ="grey", # Choose NA value
                       guide = "colourbar", # Set color bar
                       aesthetics = "fill") + # Select aesthetics to apply
  labs(x = "Subsample", y = "Column Sample by Tree", fill = "RMSE") # Set labels
g_4 # Generate plot




###### 4 - eta tuning ######

# Use xgb.cv to run cross-validation inside xgboost
set.seed(111111)
bst_mod_1 <- xgb.cv(data = dtrain, # Set training data
                    label = train_y,
                    nfold = 5, # Use 5 fold cross-validation
                    
                    eta = 0.3, # Set learning rate
                    max.depth = 10, # Set max depth
                    min_child_weight = 5, # Set minimum number of samples in node to split
                    gamma = 0, # Set minimum loss reduction for split
                    subsample = 1, # Set proportion of training data to use in tree
                    colsample_bytree =  1, # Set number of variables to use in each tree
                    
                    nrounds = 1000, # Set number of rounds
                    early_stopping_rounds = 20, # Set number of rounds to stop at if there is no improvement
                    
                    verbose = 1, # 1 - Prints out fit
                    nthread = 1, # Set number of parallel threads
                    print_every_n = 20 # Prints out result every 20th iteration
) # Set evaluation metric to use


set.seed(111111)
bst_mod_2 <- xgb.cv(data = dtrain, # Set training data
                    label = train_y,
                    nfold = 5, # Use 5 fold cross-validation
                    
                    eta = 0.1, # Set learning rate
                    max.depth =  10, # Set max depth
                    min_child_weight = 5, # Set minimum number of samples in node to split
                    gamma = 0, # Set minimum loss reduction for split
                    subsample = 1 , # Set proportion of training data to use in tree
                    colsample_bytree = 1, # Set number of variables to use in each tree
                    
                    nrounds = 1000, # Set number of rounds
                    early_stopping_rounds = 20, # Set number of rounds to stop at if there is no improvement
                    
                    verbose = 1, # 1 - Prints out fit
                    nthread = 1, # Set number of parallel threads
                    print_every_n = 20 # Prints out result every 20th iteration
) # Set evaluation metric to use

set.seed(111111)
bst_mod_3 <- xgb.cv(data = dtrain, # Set training data
                    label = train_y,
                    nfold = 5, # Use 5 fold cross-validation
                    
                    eta = 0.05, # Set learning rate
                    max.depth = 10, # Set max depth
                    min_child_weight = 5 , # Set minimum number of samples in node to split
                    gamma = 0, # Set minimum loss reduction for split
                    subsample = 1 , # Set proportion of training data to use in tree
                    colsample_bytree =  1, # Set number of variables to use in each tree
                    
                    nrounds = 1000, # Set number of rounds
                    early_stopping_rounds = 20, # Set number of rounds to stop at if there is no improvement
                    
                    verbose = 1, # 1 - Prints out fit
                    nthread = 1, # Set number of parallel threads
                    print_every_n = 20 # Prints out result every 20th iteration
) # Set evaluation metric to use


set.seed(111111)
bst_mod_4 <- xgb.cv(data = dtrain, # Set training data
                    label = train_y,
                    nfold = 5, # Use 5 fold cross-validation
                    
                    eta = 0.01, # Set learning rate
                    max.depth = 10, # Set max depth
                    min_child_weight = 5, # Set minimum number of samples in node to split
                    gamma = 0, # Set minimum loss reduction for split
                    subsample = 1 , # Set proportion of training data to use in tree
                    colsample_bytree = 1, # Set number of variables to use in each tree
                    
                    nrounds = 1000, # Set number of rounds
                    early_stopping_rounds = 20, # Set number of rounds to stop at if there is no improvement
                    
                    verbose = 1, # 1 - Prints out fit
                    nthread = 1, # Set number of parallel threads
                    print_every_n = 20 # Prints out result every 20th iteration
) # Set evaluation metric to use



set.seed(111111)
bst_mod_5 <- xgb.cv(data = dtrain, # Set training data
                    label = train_y,
                    nfold = 5, # Use 5 fold cross-validation
                    
                    eta = 0.005, # Set learning rate
                    max.depth = 10, # Set max depth
                    min_child_weight = 5, # Set minimum number of samples in node to split
                    gamma = 0, # Set minimum loss reduction for split
                    subsample = 1 , # Set proportion of training data to use in tree
                    colsample_bytree = 1, # Set number of variables to use in each tree
                    
                    nrounds = 1000, # Set number of rounds
                    early_stopping_rounds = 20, # Set number of rounds to stop at if there is no improvement
                    
                    verbose = 1, # 1 - Prints out fit
                    nthread = 1, # Set number of parallel threads
                    print_every_n = 20 # Prints out result every 20th iteration
                    
) # Set evaluation metric to use



# eta plots

# Extract results for model with eta = 0.3
pd1 <- cbind.data.frame(bst_mod_1$evaluation_log[,c("iter", "test_rmse_mean")], rep(0.3, nrow(bst_mod_1$evaluation_log)))
names(pd1)[3] <- "eta"
# Extract results for model with eta = 0.1
pd2 <- cbind.data.frame(bst_mod_2$evaluation_log[,c("iter", "test_rmse_mean")], rep(0.1, nrow(bst_mod_2$evaluation_log)))
names(pd2)[3] <- "eta"
# Extract results for model with eta = 0.05
pd3 <- cbind.data.frame(bst_mod_3$evaluation_log[,c("iter", "test_rmse_mean")], rep(0.05, nrow(bst_mod_3$evaluation_log)))
names(pd3)[3] <- "eta"
# Extract results for model with eta = 0.01
pd4 <- cbind.data.frame(bst_mod_4$evaluation_log[,c("iter", "test_rmse_mean")], rep(0.01, nrow(bst_mod_4$evaluation_log)))
names(pd4)[3] <- "eta"
# Extract results for model with eta = 0.005
pd5 <- cbind.data.frame(bst_mod_5$evaluation_log[,c("iter", "test_rmse_mean")], rep(0.005, nrow(bst_mod_5$evaluation_log)))
names(pd5)[3] <- "eta"
# Join datasets
plot_data <- rbind.data.frame(pd1, pd2, pd3, pd4, pd5)
# Converty ETA to factor
plot_data$eta <- as.factor(plot_data$eta)
# Plot points
g_6 <- ggplot(plot_data, aes(x = iter, y = test_rmse_mean, color = eta))+
  geom_point(alpha = 0.5) +
  theme_bw() + # Set theme
  theme(panel.grid.major = element_blank(), # Remove grid
        panel.grid.minor = element_blank(), # Remove grid
        panel.border = element_blank(), # Remove grid
        panel.background = element_blank()) + # Remove grid 
  labs(x = "Number of Trees", title = "RMSE v Number of Trees",
       y = "RMSE", color = "Learning \n Rate")  # Set labels
g_6

# Plot lines
g_7 <- ggplot(plot_data, aes(x = iter, y = test_rmse_mean, color = eta))+
  geom_smooth(alpha = 0.5) +
  theme_bw() + # Set theme
  theme(panel.grid.major = element_blank(), # Remove grid
        panel.grid.minor = element_blank(), # Remove grid
        panel.border = element_blank(), # Remove grid
        panel.background = element_blank()) + # Remove grid 
  labs(x = "Number of Trees", title = "RMSE v Number of Trees",
       y = "RMSE", color = "Learning \n Rate")  # Set labels
g_7




# fit final xgb model
set.seed(111111)
bst_final <- xgboost(data = dtrain, # Set training data
                     label = train_y,
                     
                     
                     eta = 0.05, # Set learning rate
                     max.depth =  10, # Set max depth
                     min_child_weight = 5, # Set minimum number of samples in node to split
                     gamma = 0, # Set minimum loss reduction for split
                     subsample =  1, # Set proportion of training data to use in tree
                     colsample_bytree = 1, # Set number of variables to use in each tree
                     
                     nrounds = 100, # Set number of rounds
                     early_stopping_rounds = 20, # Set number of rounds to stop at if there is no improvement
                     
                     verbose = 1, # 1 - Prints out fit
                     nthread = 1, # Set number of parallel threads
                     print_every_n = 20 # Prints out result every 20th iteration
                     
) # Set evaluation metric to use

library(Metrics)
actual <- test$price_30_days_out
boost_preds <- predict(bst_final, dtest)
rmse(actual, boost_preds)
results <- test %>%
  mutate(
    predicted = boost_preds,
    error = actual - predicted
  )

ggplot(results, aes(x = DATE)) +
  geom_line(aes(y = actual, color = "Actual")) +
  geom_line(aes(y = boost_preds, color = "Predicted")) +
  labs(title = "Actual vs Predicted", y = "Response Variable") +
  theme_minimal()

#prophet
IBM_Goldmine <-  data [,-184]
IBM_Goldmine <- IBM_Goldmine %>% rename(ds = DATE)
IBM_Goldmine <-IBM_Goldmine %>% rename(y = avg_buy_price_LR)

set.seed(111111)
IBM_Goldmine$cap <- 240
m5 <- prophet(IBM_Goldmine, yearly.seasonality = TRUE, weekly.seasonality = TRUE)



future5 <- make_future_dataframe(m5, periods = 500)
future5$cap <- 240
fcst <- predict(m5, future5)
plot(m5, fcst)


Closing price/price forecast 

tail(fcst)


set.seed(111111)
future5 <- cross_validation(
  m5, 
  horizon = 30,
  units = 'days',
  initial = 365,
  period = 90)
future5$cap <- 240
fcst <- predict(m5, future5)
plot(m5, fcst)



PerformanceIBM<- performance_metrics(future5)
tail(PerformanceIBM)
PerformanceIBM[28, "rmse"]


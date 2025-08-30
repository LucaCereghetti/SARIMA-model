rm(list=ls( all = TRUE )) # clears the entire R environment
#_____________________________________________________________________________________________________________________________________________
#_____________________________________________________________________________________________________________________________________________
#                                         SARIMA MODELING WORKFLOW
#_____________________________________________________________________________________________________________________________________________
#_____________________________________________________________________________________________________________________________________________

# Step 1: Import and visualize the time series data
# Step 2: Data splitting into training and test sets
# Step 3: Apply transformations and differencing (seasonal and/or non-seasonal)
# Step 4: Conduct Augmented Dickey-Fuller (ADF) test to assess stationarity
# Step 5: Perform Ljung-Box test (Q-statistic) to check for autocorrelation
# Step 6: Plot ACF and PACF to identify potential AR, MA, SAR, and SMA components
# Step 7: Estimate coefficients (not using Yule-Walker method)
# Step 8: Fit multiple candidate models while applying the parsimony principle
# Step 9: Evaluate models using AIC, SSE, and p-values to select the best fit
# Step 10: Forecast fitting SARIMA models based on identified parameters
# Step 11: Conduct residual diagnostics (Q-Q plot and Ljung-Box test) to confirm white noise behavior
# Step 12: Estimation of the forecast precision

#_____________________________________________________________________________________________________________________________________________
#                                         STEP 1 - DATA PREPARATION
#_____________________________________________________________________________________________________________________________________________
data <- read.csv("C:/Users/Melectronics/OneDrive/Documenti/Master Finance/Thesis Idea/Codes/2_Multi Regression/AAPL_prices.csv", sep = ";", header = TRUE, stringsAsFactors = FALSE)

data$Date <- as.Date(data$Date, format="%d.%m.%Y")
SP <- data[, -ncol(data)] # removing the vol colum
SP <- SP[nrow(SP):1, ] # invert the series. Starting from 2010
SP.ts = ts(SP[,2], start = c(2010, 1), frequency =252) # Creation of the time series
plot(SP.ts, main = "S&P Time Series", ylab = "S&P Index", xlab = "Time")

anyNA(SP.ts) # check if there is missing value. FALSE: no missing value
#results: FALSE

# Decompose the series in trend, seasonal and the irregular part
decomp <- decompose(SP.ts) # Check if there is seasonality. decompose() -> additive model for default
decomp$seasonal
plot(decomp)
# **Seasonality**: refers to patterns in data that repeat at regular, predictable intervals over time. These patterns typically occur alongside broader trends and are characterized by their consistent timing and cyclical nature. For example, in the stock market, certain sectors like retail often show seasonal increases in stock prices around the end-of-year holiday season due to higher consumer spending.
# **Residue (or residual)** represents the short-term irregular fluctuations in a time series that cannot be explained by trend or seasonality. These variations are random, unsystematic, and unpredictable—often caused by unexpected external events. In time series decomposition, the residual is what remains after removing the trend and seasonal components from the original data. For example, in financial markets, a sudden drop in stock prices due to an unexpected geopolitical event or a surprise regulatory announcement would be captured as a residual component.

#----------------------------------------------------------------------
# EXTRACTING A SHORTER DATASET
#----------------------------------------------------------------------
# Here we extract shorter data windows, as running SARIMA on the full dataset 
# yielded only p-values below 0.05.

# Now we will use the best training window identified in the previous analysis
SP_3y <- ts(tail(SP.ts, 756), start=c(2022, 1), frequency=252)    # try windows of 756, 1008, ecc.
n <- length(SP_3y)                 # Total number of observations
split_index <- floor(0.85 * n)      # 80% index
train.ts_3y <- window(SP_3y, end = time(SP_3y)[split_index])
length(train.ts_3y)
# result: 642

# Creating Training and Test Sets
n <- length(SP.ts)                 # Total number of observations
split_index <- floor(0.85 * n)      # 80% index
train.ts <- window(SP.ts, end = time(SP.ts)[split_index])
test.ts <- window(SP.ts, start = time(SP.ts)[split_index + 1])
tidx <- time(SP.ts)
# 642 values ending exactly at the split point
last_train  <- window(SP.ts,
                      start = tidx[split_index - 642 + 1],
                      end   = tidx[split_index])
# Sanity checks
length(last_train)   # expect 642
length(test.ts)   # expect 567
# Plot
plot(SP.ts, type = "l", col = "gray", lwd = 1.5,
     main = "Train/Test Window Around the 85% Split",
     xlab = "Time", ylab = "Price")
lines(last_train, col = "blue", lwd = 2)
lines(test.ts, col = "red",  lwd = 2)
# Mark the split boundary
abline(v = tidx[split_index], lty = 2)
points(tidx[split_index], SP.ts[split_index], pch = 19)
legend("topleft",
       legend = c("Full series", "Train (last 642)", "Test (567)"),
       col = c("gray", "blue", "red"), lwd = 2, lty = 1)

# Sanity checks
length(last_train)   # expect 642
length(test.ts)   # expect 567

train.ts <- last_train

#----------------------------------------------------------------------
# TRANSFORMATIONS AND DIFFERENCING
#----------------------------------------------------------------------
log_diff_train.ts_d <- diff(log(train.ts)) # Regular (trend) differencing - log stabilizes the variance, and transforms multiplicative relationships into additive. During the time in a multiplicative the series can growth or shrink.
plot(log_diff_train.ts_d, main = "Log transformation + first differenciation", col = "blue", type='l',  ylab = "First Log Difference (log(y_t) - log(y_{t-1}))")

# Comment: The differenced log series shows higher variance in the early period and lower variance later, 
# suggesting heteroskedasticity. Since ARIMA assumes constant variance, this may affect model adequacy.

# Seasonal differencing to remove the seasonal component:
log_diff_train.ts_D <- diff(log_diff_train.ts_d, 252)  # Seasonal differencing at lag = 252
log.diff.mean.zero=log_diff_train.ts_D-mean(log_diff_train.ts_D) 
plot(log.diff.mean.zero, main = "Log transformation + seasonal difference (lag 252)", col = "blue", type='l', ylab = "Seasonal Log Difference (log(y_t) - log(y_{t-252}))")

# Comment: After applying seasonal differencing (lag = 252), the variance appears more stable 
# and the series oscillates around zero, reducing heteroskedasticity. 
# This makes the data more suitable for ARIMA/SARIMA modeling.


# plot the histogram and QQ
par( mfrow=c(1,2) ) 
hist(log.diff.mean.zero, breaks = 15, main="Histogram of Transformed Series", xlab="%", col = "lightgray", probability = TRUE) 
curve(dnorm(x, mean = mean(log.diff.mean.zero), sd = sd(log.diff.mean.zero)), col = "blue", lwd = 2, add = TRUE)
qqnorm(log.diff.mean.zero,main="Q-Q Plot of Transformed Series") 
qqline(log.diff.mean.zero) 

#----------------------------------------------------------------------
# STATIONARITY AND AUTOCORRELATION TESTS
#----------------------------------------------------------------------
# Augmented Dickey–Fuller (ADF) test to formally assess stationarity
library(tseries)
adf_result_diff <- adf.test(log_diff_train.ts_D)
print(adf_result_diff)
# results: p-value = 0.01 < 0.05 accept the null hypothesis, the series is stationary

# BOX-Pierce test (autocorrelation between previous lags)
Box.test(log.diff.mean.zero, lag=log(length(log.diff.mean.zero))) 
# results: X-squared = 5.4764, df = 5.9636, p-value = 0.4799 -> P-value is more than 0.05 so we accept the null Hypothesis NON-autocorrelation in residuals.

#----------------------------------------------------------------------
# ACF AND PACF ANALYSIS
#----------------------------------------------------------------------
par(mfrow=c(2,1)) 
# Converting to numeric ensures that lags are represented as whole numbers, corresponding directly to trading days
acf(as.numeric(log.diff.mean.zero), main = "ACF of MA(q)", lag.max = 100)  # as.numeric() is used because ts objects display time on a fractional scale (e.g., in years).
pacf(as.numeric(log.diff.mean.zero), main = "PACF of AR(p)", lag.max = 100)
# observation: 
# ACF -> q=0,1,2   Q=0,1,2,3    SMA(Q): 6, ~58, ~61, ~80.   
# PACF -> p=0,1,2  P=0,1,2      SAR(P): 6, ~80

# conclusion:
# SARIMA(p,1,q,P,1,Q)_252    First differentiation (d) is in the NON seasonal, and the Second differentiation (D) is 1 in the SEASONAL
# 0 ≤𝑝,𝑞,𝑃,𝑄 ≤ 3

#_____________________________________________________________________________________________________________________________________________
#                                         STEP 2 - FIT SARIMA MODELS
#_____________________________________________________________________________________________________________________________________________
# In this part we will fit differents SARIMA models.
# To avoid overfitting—a common issue in ARMA models—and redundancy, p, q, P, and Q are limited to a maximum of 2.

d=1  # if we use x=log.diff.mean.zero we do not need to differentiate again or we can put d=1 if we use log(train.ts)
DD=1 # if we use x=log.diff.mean.zero we do not need to differentiate again or we can put DD=1 if we use log(train.ts)
per=5 # period of seasonality: weekly 5, monthly 21, quarterly 63, yearly 252 
for(p in 1:3){
  for(q in 1:3){
    for(P in 1:3){
      for(Q in 1:3){
        if(p+d+q+P+DD+Q<=10){  # the parsimony principle
          model<-arima(x=log(train.ts), order = c((p-1),d,(q-1)), seasonal = list(order=c((P-1),DD,(Q-1)), period=per))
          pval<-Box.test(model$residuals, lag=log(length(model$residuals)))
          sse<-sum(model$residuals^2)
          cat(p-1,d,q-1,P-1,DD,Q-1,per, 'AIC=', model$aic, ' SSE=',sse,' p-VALUE=', pval$p.value,'\n')
        }
      }
    }
  }
}

# If the model fails to converge, it generally indicates optimizer or numerical issues — for example, 
# parameters lying close to the unit circle or the likelihood becoming non-finite at certain trial values.

# output:
# 0 1 0 0 1 0 5 AIC= -2698.253  SSE= 0.5335806  p-VALUE= 0 
# 0 1 0 0 1 1 5 AIC= -3047.657  SSE= 0.2991376  p-VALUE= 0.02308673 
# 0 1 0 0 1 2 5 AIC= -3048.218  SSE= 0.29761  p-VALUE= 0.2435941 
# 0 1 0 1 1 0 5 AIC= -2835.127  SSE= 0.4281737  p-VALUE= 0.001981563 
# 0 1 0 1 1 1 5 AIC= -3047.98  SSE= 0.2977376  p-VALUE= 0.2097806 
# 0 1 0 1 1 2 5 AIC= -3046.569  SSE= 0.2975126  p-VALUE= 0.2613386 
# 0 1 0 2 1 0 5 AIC= -2889.239  SSE= 0.3914269  p-VALUE= 0.04712006 
# 0 1 0 2 1 1 5 AIC= -3047.653  SSE= 0.2971763  p-VALUE= 0.3168392 
# 0 1 0 2 1 2 5 AIC= -3049.243  SSE= 0.2959452  p-VALUE= 0.3183403 
# 0 1 1 0 1 0 5 AIC= -2697.082  SSE= 0.5328846  p-VALUE= 0 
# 0 1 1 0 1 1 5 AIC= -3050.237  SSE= 0.2969618  p-VALUE= 0.08683103 
# 0 1 1 0 1 2 5 AIC= -3050.173  SSE= 0.2957567  p-VALUE= 0.4322576 
# 0 1 1 1 1 0 5 AIC= -2837.784  SSE= 0.4250122  p-VALUE= 0.01545702 
# 0 1 1 1 1 1 5 AIC= -3050.025  SSE= 0.2958358  p-VALUE= 0.4018031 
# 0 1 1 1 1 2 5 AIC= -3048.384  SSE= 0.2957084  p-VALUE= 0.4409718 
# 0 1 1 2 1 0 5 AIC= -2891.017  SSE= 0.3890868  p-VALUE= 0.1682545 
# 0 1 1 2 1 1 5 AIC= -3049.141  SSE= 0.295515  p-VALUE= 0.4676989 
# 0 1 2 0 1 0 5 AIC= -2695.887  SSE= 0.5322086  p-VALUE= 0 
# 0 1 2 0 1 1 5 AIC= -3048.343  SSE= 0.2968764  p-VALUE= 0.08547884 
# 0 1 2 0 1 2 5 AIC= -3048.328  SSE= 0.29563  p-VALUE= 0.4463649 
# 0 1 2 1 1 0 5 AIC= -2836.35  SSE= 0.4246352  p-VALUE= 0.02126968 
# 0 1 2 1 1 1 5 AIC= -3048.178  SSE= 0.295708  p-VALUE= 0.4146416 
# 0 1 2 2 1 0 5 AIC= -2889.634  SSE= 0.3887089  p-VALUE= 0.2297538 
# 1 1 0 0 1 0 5 AIC= -2697.142  SSE= 0.5328344  p-VALUE= 0 
# 1 1 0 0 1 1 5 AIC= -3050.372  SSE= 0.2968819  p-VALUE= 0.08678757 
# 1 1 0 0 1 2 5 AIC= -3050.313  SSE= 0.2956692  p-VALUE= 0.4378667 
# 1 1 0 1 1 0 5 AIC= -2838.07  SSE= 0.4248189  p-VALUE= 0.01778326 
# 1 1 0 1 1 1 5 AIC= -3050.167  SSE= 0.2957461  p-VALUE= 0.4075193 
# 1 1 0 1 1 2 5 AIC= -3048.517  SSE= 0.2956404  p-VALUE= 0.4422947 
# 1 1 0 2 1 0 5 AIC= -2891.275  SSE= 0.3889267  p-VALUE= 0.1862284 
# 1 1 0 2 1 1 5 AIC= -3049.255  SSE= 0.2954425  p-VALUE= 0.4704759 
# 1 1 1 0 1 0 5 AIC= -2754.659  SSE= 0.4819674  p-VALUE= 0 
# 1 1 1 0 1 1 5 AIC= -3049.084  SSE= 0.2965101  p-VALUE= 0.08647593 
# 1 1 1 0 1 2 5 AIC= -3049.24  SSE= 0.2951897  p-VALUE= 0.4892646 
# 1 1 1 1 1 0 5 AIC= -2837.832  SSE= 0.4236297  p-VALUE= 0.03337825 
# 1 1 1 1 1 1 5 AIC= -3049.076  SSE= 0.2952705  p-VALUE= 0.4558285 
# 1 1 1 2 1 0 5 AIC= -2890.916  SSE= 0.3879087  p-VALUE= 0.323681 
# 1 1 2 0 1 0 5 AIC= -2755.148  SSE= 0.480268  p-VALUE= 0 
# 1 1 2 0 1 1 5 AIC= -3047.095  SSE= 0.2965123  p-VALUE= 0.08728873 
# 1 1 2 1 1 0 5 AIC= -2835.832  SSE= 0.4236301  p-VALUE= 0.03330406 
# 2 1 0 0 1 0 5 AIC= -2695.97  SSE= 0.5321386  p-VALUE= 0 
# 2 1 0 0 1 1 5 AIC= -3048.524  SSE= 0.2967723  p-VALUE= 0.08399403 
# 2 1 0 0 1 2 5 AIC= -3048.519  SSE= 0.2955166  p-VALUE= 0.4490449 
# 2 1 0 1 1 0 5 AIC= -2836.606  SSE= 0.4244626  p-VALUE= 0.02332499 
# 2 1 0 1 1 1 5 AIC= -3048.37  SSE= 0.295597  p-VALUE= 0.4174581 
# 2 1 0 2 1 0 5 AIC= -2889.97  SSE= 0.3884996  p-VALUE= 0.2567621 
# 2 1 1 0 1 0 5 AIC= -2756.138  SSE= 0.4796747  p-VALUE= 0 
# 2 1 1 0 1 1 5 AIC= -3046.39  SSE= 0.2968878  p-VALUE= 0.08508841 
# 2 1 1 1 1 0 5 AIC= -2835.832  SSE= 0.4236297  p-VALUE= 0.03336791 
# 2 1 2 0 1 0 5 AIC= -2755.243  SSE= 0.4785378  p-VALUE= 0 

# BEST RESULT: 1 1 0 0 1 2 5 AIC= -3050.313  SSE= 0.2956692  p-VALUE= 0.4378667 

#----------------------------------------------------------------------
# RESIDUALS ANALYSIS
#----------------------------------------------------------------------
library(astsa)
fit <-sarima(log(train.ts), 1,1,0,0,1,2,5)
fit

# Coefficients: 
#            Estimate     SE        t.value       p.value
#     ar1   -0.0832     0.0410     -2.0268        0.0431
#     sma1  -0.9259     0.0434    -21.3204        0.0000
#     sma2  -0.0602     0.0430     -1.4004        0.1619

# sigma^2 estimated as 0.0004647089 on 633 degrees of freedom 

# AIC = -4.79609  AICc = -4.79603  BIC = -4.76807

res <- residuals(fit$fit)

Box.test(res,
         type = "Ljung-Box",
         lag = floor(log(length(res))))   # log needs integer, so wrap with floor()
# Using log(n) lags is standard for long time series to avoid overfitting with noise.
# result: X-squared = 6.4239, df = 6, p-value = 0.3774
# Ho: No autocorrelation
# H1: Autocorrelation
# result: here is no significant autocorrelation in the residuals 
# This means there is no significant autocorrelation in the log-differenced data,
# so the series can be considered approximately white noise (random).

# Since we are working with a long time series, we use log(length(res)) to determine a reasonable number of lags for the Ljung-Box test. 
# Time series data often exhibit strong autocorrelation at early lags, which tends to diminish as the lag increases. 
# Including a larger number of lags can dilute this signal, as the additional lags often contain minimal or no autocorrelation.

#_____________________________________________________________________________________________________________________________________________
#                                                 FORECAST
#_____________________________________________________________________________________________________________________________________________
#                        STEP 3 – OUT-OF-SAMPLE EVALUATION: SARIMA MODELS VS. ACTUAL TEST DATA 
#_____________________________________________________________________________________________________________________________________________
install.packages("forecast")
library(forecast)

# BEST RESULT: 1 1 0 0 1 2 5 AIC= -3050.313  SSE= 0.2956692  p-VALUE= 0.4378667 

model <- arima(x = log(train.ts), order = c(1,1,0), seasonal = list(order = c(0,1,2), period = 5)) 
fc <- forecast(model, h = length(test.ts)) # Forecast 150 steps ahead: h = length(test.ts)

plot(fc) # ‘forecast’ package
forecast(model)


# Create time axis for forecast and test
last_train_time <- time(train.ts)[length(train.ts)]
forecast_time <- seq(from = last_train_time + 1/frequency(train.ts), by = 1/frequency(train.ts), length.out = length(test.ts))  

log_test_ts <- log(test.ts) # log-transformed the test value to match forecast

# Plot: Training + Forecast + Test on same chart
plot(log(train.ts), 
     xlim = range(time(train.ts)[550], forecast_time[length(forecast_time)]), 
     ylim = range(c(log(train.ts)-0.3, fc$mean, log_test_ts)+0.3), 
     main = "Train + Forecast vs. Test", 
     xlab = "Time", 
     ylab = "Log Value", 
     lwd = 1)

# Add forecast line
lines(forecast_time, fc$mean, col = "red", lwd = 1)

# Add 80% and 95% confidence intervals
lines(forecast_time, fc$lower[,2], col = "gray", lty = 2)
lines(forecast_time, fc$upper[,2], col = "gray", lty = 2)

# Add actual test data (log-transformed)
lines(forecast_time, log_test_ts, col = "green", lwd = 1)

# Add legend
legend("topleft", legend = c("Train", "Forecast", "Test", "95% CI"),
       col = c("black", "red", "green", "gray"), 
       lty = c(1,1,1,2), 
       lwd = 1)

#----------------------------------------------------------------------
# OUT-OF-SAMPLE ERROR ASSESSMENT: SARIMA MODEL VS. ACTUAL TEST DATA
#----------------------------------------------------------------------
actual_value <- log(test.ts)
# actual_value <- test.ts
predicted_forecast   <- as.numeric(fc$mean) # extrapolate the forecasts from fc
# predicted_forecast <- exp(fc$mean)


# TRANSFORMATION OF PRICE FORECAST ERRORS INTO RETURN ERRORS
last_price_train <- tail(train.ts, 1)
# out-of-sample prices series
pred_prices <- exp(as.numeric(predicted_forecast))
test_prices <- as.numeric(test.ts)
pred_prices_full <- c(last_price_train, pred_prices)
test_prices_full  <- c(last_price_train, test_prices)
# simple returns: r_t = (P_t - P_{t-1}) / P_{t-1}
pred_ret <- diff(pred_prices_full) / head(pred_prices_full, -1)
test_ret  <- diff(test_prices_full)  / head(test_prices_full,  -1)

# returns errors
errors_ret <- test_ret - pred_ret

MSE <- mean(errors_ret^2)     # Mean Squared Error
RMSE <- sqrt(MSE)         # Root Mean Squared Error
MAE <- mean(abs(errors_ret))  # Mean Absolute Error
MAPE <- mean(abs(errors_ret) / abs(actual_value)) * 100  # Mean Absolute Percentage Error
MAD <- median(abs(errors_ret))  # Median Absolute Deviation

# R-squared
SSE <- sum((errors_ret)^2)
SST <- sum((test.ts - mean(test.ts))^2)
R2  <- 1 - (SSE / SST)

# Directional accuracy
ok <- complete.cases(pred_ret, test_ret)
table(ok)     # quanti TRUE/FALSE
DA_forecast <- mean(sign(test_ret[ok]) == sign(pred_ret[ok]))

cat("MAE:", MAE,
    "| MSE:", MSE,
    "| RMSE:", RMSE,
    "| MAPE:", MAPE, "%",
    "| MAD:", MAD,
    "| DA:", round(DA_forecast  * 100, 2), "%",
    "| R²:", R2, "\n")
# MAE: 0.01129182 | MSE: 0.0002420715 | RMSE: 0.01555865 | MAPE: 0.2186084 % | MAD: 0.008247624 | DA: 48.68 % | R²: 0.9999997 

#_____________________________________________________________________________________________________________________________________________
#                 STEP 4 – OUT-OF-SAMPLE EVALUATION: ROLLING-WINDOW SARIMA MODEL VS. ACTUAL TEST DATA
#_____________________________________________________________________________________________________________________________________________
# 
# ----------------------------------------------------------
# PRELIMINARY ANALYSIS OF ROLLING WINDOWS
# ----------------------------------------------------------
# Before implementing the walk-forward analysis, it is important to verify
# that the model can accurately forecast the value at t+1.
# Since running this evaluation over 113 steps can be computationally intensive,
# the goal is to explore ways to reduce the computational load 
# without significantly compromising forecast accuracy.

# forecast n+1:
length(train.ts)
length(test.ts)

model_1 <- arima(train.ts, order = c(1,1,0), seasonal = list(order = c(0,1,2), period = 5)) 
model_2 <- arima(tail(train.ts, 300), order = c(1,1,0), seasonal = list(order = c(0,1,2), period = 21)) # i used tail(train.ts, 300) to trying to reduce the number of data that model use for computation
model_3 <- arima(train.ts, order = c(1,1,0), seasonal = list(order = c(0,1,2), period = 21))
model_4 <- arima(train.ts, order = c(1,1,0), seasonal = list(order = c(0,1,2), period = 63)) # reduce the period at 20 and remove the D to avoid a double differentiation

# forecast
fc_1 <- forecast(model_1, h = 1)
fc_2 <- forecast(model_2, h = 1)
fc_3 <- forecast(model_3, h = 1)
fc_4 <- forecast(model_3, h = 1)

cat("Last value of train.ts:", tail(train.ts, 1), "\n")
cat("First value of test.ts :", head(test.ts, 1), "\n")
cat("Forecast fc_1:", fc_1$mean, "\n")
cat("Forecast fc_2:", fc_2$mean, "\n")
cat("Forecast fc_3:", fc_3$mean, "\n")
cat("Forecast fc_3:", fc_4$mean, "\n")

# Ground truth reference values:
# - Last observed value in train.ts  : 149.84 
# - First true value in test.ts      : 142.48  

# Forecast results from three different models:
# - fc_1 → Forecast = 150.2337  
# - fc_2 → Forecast = 149.7557  
# - fc_3 → Forecast = 149.8416   
# - fc_4 → Forecast = 149.8416   

# Commentary:
# All forecasts are reasonably close to the actual test value (142.48),
# with fc_3 and fc_4 being the most accurate in this case.
# Based on the one-step-ahead forecast results, the ARIMA model used in fc_3 
# demonstrates the highest accuracy relative to the actual value, making it the most 
# suitable choice for the subsequent walk-forward analysis.

# ----------------------------------------------------------
# ROLLING-WINDOW FORECAST
# ----------------------------------------------------------
# Parameters
n_forecasts <- length(test.ts)  # Number of forecast steps you want to simulate
window_size <- length(train.ts)  # Fixed rolling window size
# window_size <- length(SP_shorter)

# Initialize storage
approx_forecasts <- numeric(n_forecasts)

# Step 1: Initialize training window
window <- tail(train.ts, window_size)
window
log(window)

# True walk-forward: use real next point, not the forecasted one
window <- tail(train.ts, window_size)

# try with log.diff.mean.zero instead log(window) since log.diff.mean.zero has the mean on 0
# BEST RESULT: 1 1 0 0 1 2 5 AIC= -3050.313  SSE= 0.2956692  p-VALUE= 0.4378667

for (i in 1:n_forecasts) {
  model <- arima(x = log(window), order = c(1,1,0), seasonal = list(order = c(0,1,2), period = 21))
  fc <- forecast(model, h = 1)
  approx_forecasts[i] <- exp(fc$mean[1])
  
  # Use the real next observation, not the forecast
  window <- ts(c(tail(window, window_size - 1), test.ts[i]),
               start = time(window)[2],
               frequency = frequency(window))
}

approx_forecasts
test.ts

# Convert forecast vector to time series
forecast_start_time <- time(train.ts)[length(train.ts)] + 1/frequency(train.ts)
walk_forward_forecasts_ts <- ts(approx_forecasts, start = forecast_start_time, frequency = frequency(train.ts))

SP_shorter_zoom <- window(SP.ts, start = 2022.6)

# Plot
plot(SP_shorter_zoom, col = "gray", lwd = 1.5,
     main = "Walk-Forward Forecast using True Values",
     xlab = "Time", ylab = "AAPL Index")

lines(window(test.ts, start = time(walk_forward_forecasts_ts)[1]), col = "green", lwd = 2)
lines(walk_forward_forecasts_ts, col = "red", lwd = 2)

legend("topright", legend = c("True Series", "True Test", "Walk-Forward Forecast"),
       col = c("gray", "green", "red"), lty = 1, lwd = 2)


# Build a comparison table (align lengths just in case)
y_true   <- as.numeric(test.ts)
y_pred   <- as.numeric(walk_forward_forecasts_ts)

n <- min(length(y_true), length(y_pred))

comparison <- data.frame(
  Time           = time(test.ts)[seq_len(n)],
  True_Value     = y_true[seq_len(n)],
  ARIMA_Forecast = y_pred[seq_len(n)]
)

print(head(comparison, 10))

#----------------------------------------------------------------------
# OUT-OF-SAMPLE ERROR ASSESSMENT: ROLLING-WINDOW SARIMA MODEL VS. ACTUAL TEST DATA
#----------------------------------------------------------------------
actual_value <- log(test.ts)
predicted_forecast   <- log(approx_forecasts)

# TRANSFORMATION OF PRICE FORECAST ERRORS INTO RETURN ERRORS
last_price_train <- tail(train.ts, 1)
# out-of-sample prices series
pred_prices <- exp(as.numeric(predicted_forecast))
test_prices <- as.numeric(test.ts)

pred_prices_full <- c(last_price_train, pred_prices)
test_prices_full  <- c(last_price_train, test_prices)

# simple returns: r_t = (P_t - P_{t-1}) / P_{t-1}
pred_ret <- diff(pred_prices_full) / head(pred_prices_full, -1)
test_ret  <- diff(test_prices_full)  / head(test_prices_full,  -1)

# returns errors
errors_ret <- test_ret - pred_ret

MSE <- mean(errors_ret^2)     # Mean Squared Error
RMSE <- sqrt(MSE)         # Root Mean Squared Error
MAE <- mean(abs(errors_ret))  # Mean Absolute Error
MAPE <- mean(abs(errors_ret) / abs(actual_value)) * 100  # Mean Absolute Percentage Error
MAD <- median(abs(errors_ret))  # Median Absolute Deviation

# R-squared
SSE <- sum((errors_ret)^2)
SSE
SST <- sum((actual_value - mean(actual_value))^2)
SST
R2  <- 1 - (SSE / SST)

ok <- complete.cases(pred_ret, test_ret)
table(ok)     # quanti TRUE/FALSE
DA_forecast <- mean(sign(test_ret[ok]) == sign(pred_ret[ok]))

cat("MAE:", MAE,
    "| MSE:", MSE,
    "| RMSE:", RMSE,
    "| MAPE:", MAPE, "%",
    "| MAD:", MAD,
    "| DA:", round(DA_forecast  * 100, 2), "%",
    "| R²:", R2, "\n")
# results: MAE: 0.01141331 | MSE: 0.000244684 | RMSE: 0.01564238 | MAPE: 0.2209306 % | MAD: 0.008699607 | DA: 51.06 % | R²: 0.990913 

























rm(list=ls( all = TRUE )) # clears the entire R environment

#---------------------------------------------------------------------
# Autoregressive Processes of order 2 -EXAMPLE
#---------------------------------------------------------------------

set.seed(2017)
phi1=0.6
phi2=-0.2
X.t<- arima.sim(list(ar=c(phi1,phi2)), n=1000) # converts the numeric vector X in time series
# phi_2 < 1 + phi 1
# phi_2 < 1 - phi 1
par(mfrow=c(3,1)) # dimensions of the image
plot(X.t, main = "AR(2) Time Series on WHite Noise, phi1= 0.6, phi2=-0.2")
acf(X.t, main="Autocorrelation Function") 
acf(X.t, type="partial", main="Partial Autocorrelation Function, phi1= 0.6, phi2= -0.2")

#---------------------------------------------------------------------
# ACF values
#---------------------------------------------------------------------
X.acf=acf(X.t, main = "Autocorrelation of AR(2) Time Series")
r.coef = X.acf$acf # Run the code several times without setting the seed 
print(r.coef) # value of the ACF


#---------------------------------------------------------------------
# Coefficient phi_1 and phi_2 Estimation
#---------------------------------------------------------------------
# estimation of the coefficients phi_1 and phi_2
ar(na.omit(X.t), order.max = 5)
# results:
#     Coefficients:
#         1        2  
#       0.6053  -0.1965  
# very close to the right parameters: phi1 = 0.6 and phi2 = -0.2


#---------------------------------------------------------------------
# PACF values at lag h using the residuals from forward and backward regressions.
#---------------------------------------------------------------------
# we assume that we want estimate h=3      
# Create lagged variables
x <- as.numeric(X.t)  # convert to numeric vector for indexing
n <- length(x)
h <- 3  # forecast horizon

# BACKWARD regression: estimate x_t from x_{t+1}, x_{t+2}
Y_back <- x[1:(n - h + 1)]
X1_back <- x[2:(n - h + 2)]
X2_back <- x[3:(n - h + 3)]
df_back <- data.frame(Y = Y_back, X1 = X1_back, X2 = X2_back)
model_back <- lm(Y ~ X1 + X2, data = df_back)
resid_back <- resid(model_back)  # e_t = x_t - hat{x}_t

# FORWARD regression: estimate x_{t+h} from x_{t+1}, x_{t+2}
Y_fwd <- x[h:n]
X1_fwd <- x[(h-1):(n - 1)]
X2_fwd <- x[(h-2):(n - 2)]
df_fwd <- data.frame(Y = Y_fwd, X1 = X1_fwd, X2 = X2_fwd)
model_fwd <- lm(Y ~ X1 + X2, data = df_fwd)
resid_fwd <- resid(model_fwd)  # e_{t+h} = x_{t+h} - hat{x}_{t+h}

# Align residuals (same length)
min_len <- min(length(resid_back), length(resid_fwd))
resid_back <- resid_back[1:min_len]
resid_fwd <- resid_fwd[1:min_len]

# Compute partial autocorrelation
phi_hh <- cor(resid_back, resid_fwd)

# Output result
cat("Partial autocorrelation at lag h =", h, "is:", round(phi_hh, 4), "\n")

# AUTOMATIC ----------------------------

mat <- data.frame(
  x_t   = x[1:(n - h)],
  x_1   = x[2:(n - h + 1)],
  x_2   = x[3:(n - h + 2)],
  x_th  = x[(1 + h):(n)]
)

# Estimate partial correlation between x_t and x_{t+h}, controlling for x_{t+1}, ..., x_{t+h-1}
result <- pcor(mat)

# Show the partial correlation between x_t and x_{t+h}
phi_hh <- result$estimate["x_t", "x_th"]
cat("Estimated partial autocorrelation at lag", h, "is:", round(phi_hh, 4), "\n")






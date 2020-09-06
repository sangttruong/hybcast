# Import Data
raw=read.csv("C:\\Users\\ri158\\OneDrive - Cummins\\Africa Correlation\\Raw.csv")
dim(raw)
names(raw)
head(raw)

# Visualizing data
raw2 <- raw[,c(-1,-2,-4,-6,-8,-10,-12,-14,-16,-18,-20,-22,-24,-26)]
raw3 <- raw[,c(-1, -3, -5, -7, -9, -11, -13, -15, -17, -19, -21, -23, -25, -27)]
pairs(raw2)
pairs(raw3)

# Lag function
lagpad <- function(x, k) {
  if (k>0) {
    return (c(rep(NA, k), x)[1 : length(x)] );
  }
  else {
    return (c(x[(-k+1) : length(x)], rep(NA, -k)));
  }
}

# Generate Lag variable
raw$L1N.Oil=lagpad(raw$N.Oil,1)
raw$L2N.Oil=lagpad(raw$N.Oil,2)
raw$L3N.Oil=lagpad(raw$N.Oil,3)
raw$L4N.Oil=lagpad(raw$N.Oil,4)
raw$L1DN.Oil=lagpad(raw$DN.Oil,1)
raw$L2DN.Oil=lagpad(raw$DN.Oil,2)
raw$L3DN.Oil=lagpad(raw$DN.Oil,3)
raw$L4DN.Oil=lagpad(raw$DN.Oil,4)
raw$L1N.Au=lagpad(raw$N.Au,1)
raw$L2N.Au=lagpad(raw$N.Au,2)
raw$L3N.Au=lagpad(raw$N.Au,3)
raw$L4N.Au=lagpad(raw$N.Au,4)
raw$L1DN.Au=lagpad(raw$DN.Au,1)
raw$L2DN.Au=lagpad(raw$DN.Au,2)
raw$L3DN.Au=lagpad(raw$DN.Au,3)
raw$L4DN.Au=lagpad(raw$DN.Au,4)
raw$L1N.Cu=lagpad(raw$N.Cu,1)
raw$L2N.Cu=lagpad(raw$N.Cu,2)
raw$L3N.Cu=lagpad(raw$N.Cu,3)
raw$L4N.Cu=lagpad(raw$N.Cu,4)
raw$L1DN.Cu=lagpad(raw$DN.Cu,1)
raw$L2DN.Cu=lagpad(raw$DN.Cu,2)
raw$L3DN.Cu=lagpad(raw$DN.Cu,3)
raw$L4DN.Cu=lagpad(raw$DN.Cu,4)
raw$L1N.Coc=lagpad(raw$N.Coc,1)
raw$L2N.Coc=lagpad(raw$N.Coc,2)
raw$L3N.Coc=lagpad(raw$N.Coc,3)
raw$L4N.Coc=lagpad(raw$N.Coc,4)
raw$L1DN.Coc=lagpad(raw$DN.Coc,1)
raw$L2DN.Coc=lagpad(raw$DN.Coc,2)
raw$L3DN.Coc=lagpad(raw$DN.Coc,3)
raw$L4DN.Coc=lagpad(raw$DN.Coc,4)
raw$L1N.ZARUSD=lagpad(raw$N.ZARUSD,1)
raw$L2N.ZARUSD=lagpad(raw$N.ZARUSD,2)
raw$L3N.ZARUSD=lagpad(raw$N.ZARUSD,3)
raw$L4N.ZARUSD=lagpad(raw$N.ZARUSD,4)
raw$L1DN.ZARUSD=lagpad(raw$DN.ZARUSD,1)
raw$L2DN.ZARUSD=lagpad(raw$DN.ZARUSD,2)
raw$L3DN.ZARUSD=lagpad(raw$DN.ZARUSD,3)
raw$L4DN.ZARUSD=lagpad(raw$DN.ZARUSD,4)
raw$L1N.AOAUSD=lagpad(raw$N.AOAUSD,1)
raw$L2N.AOAUSD=lagpad(raw$N.AOAUSD,2)
raw$L3N.AOAUSD=lagpad(raw$N.AOAUSD,3)
raw$L4N.AOAUSD=lagpad(raw$N.AOAUSD,4)
raw$L1DN.AOAUSD=lagpad(raw$DN.AOAUSD,1)
raw$L2DN.AOAUSD=lagpad(raw$DN.AOAUSD,2)
raw$L3DN.AOAUSD=lagpad(raw$DN.AOAUSD,3)
raw$L4DN.AOAUSD=lagpad(raw$DN.AOAUSD,4)
raw$L1N.NGNUSD=lagpad(raw$N.NGNUSD,1)
raw$L2N.NGNUSD=lagpad(raw$N.NGNUSD,2)
raw$L3N.NGNUSD=lagpad(raw$N.NGNUSD,3)
raw$L4N.NGNUSD=lagpad(raw$N.NGNUSD,4)
raw$L1DN.NGNUSD=lagpad(raw$DN.NGNUSD,1)
raw$L2DN.NGNUSD=lagpad(raw$DN.NGNUSD,2)
raw$L3DN.NGNUSD=lagpad(raw$DN.NGNUSD,3)
raw$L4DN.NGNUSD=lagpad(raw$DN.NGNUSD,4)
raw$L1N.GHSUSD=lagpad(raw$N.GHSUSD,1)
raw$L2N.GHSUSD=lagpad(raw$N.GHSUSD,2)
raw$L3N.GHSUSD=lagpad(raw$N.GHSUSD,3)
raw$L4N.GHSUSD=lagpad(raw$N.GHSUSD,4)
raw$L1DN.GHSUSD=lagpad(raw$DN.GHSUSD,1)
raw$L2DN.GHSUSD=lagpad(raw$DN.GHSUSD,2)
raw$L3DN.GHSUSD=lagpad(raw$DN.GHSUSD,3)
raw$L4DN.GHSUSD=lagpad(raw$DN.GHSUSD,4)
raw$L1N.BWPUSD=lagpad(raw$N.BWPUSD,1)
raw$L2N.BWPUSD=lagpad(raw$N.BWPUSD,2)
raw$L3N.BWPUSD=lagpad(raw$N.BWPUSD,3)
raw$L4N.BWPUSD=lagpad(raw$N.BWPUSD,4)
raw$L1DN.BWPUSD=lagpad(raw$DN.BWPUSD,1)
raw$L2DN.BWPUSD=lagpad(raw$DN.BWPUSD,2)
raw$L3DN.BWPUSD=lagpad(raw$DN.BWPUSD,3)
raw$L4DN.BWPUSD=lagpad(raw$DN.BWPUSD,4)
raw$L1N.MADUSD=lagpad(raw$N.MADUSD,1)
raw$L2N.MADUSD=lagpad(raw$N.MADUSD,2)
raw$L3N.MADUSD=lagpad(raw$N.MADUSD,3)
raw$L4N.MADUSD=lagpad(raw$N.MADUSD,4)
raw$L1DN.MADUSD=lagpad(raw$DN.MADUSD,1)
raw$L2DN.MADUSD=lagpad(raw$DN.MADUSD,2)
raw$L3DN.MADUSD=lagpad(raw$DN.MADUSD,3)
raw$L4DN.MADUSD=lagpad(raw$DN.MADUSD,4)
raw$L1N.XOFUSD=lagpad(raw$N.XOFUSD,1)
raw$L2N.XOFUSD=lagpad(raw$N.XOFUSD,2)
raw$L3N.XOFUSD=lagpad(raw$N.XOFUSD,3)
raw$L4N.XOFUSD=lagpad(raw$N.XOFUSD,4)
raw$L1DN.XOFUSD=lagpad(raw$DN.XOFUSD,1)
raw$L2DN.XOFUSD=lagpad(raw$DN.XOFUSD,2)
raw$L3DN.XOFUSD=lagpad(raw$DN.XOFUSD,3)
raw$L4DN.XOFUSD=lagpad(raw$DN.XOFUSD,4)
raw$L1N.ZMWUSD=lagpad(raw$N.ZMWUSD,1)
raw$L2N.ZMWUSD=lagpad(raw$N.ZMWUSD,2)
raw$L3N.ZMWUSD=lagpad(raw$N.ZMWUSD,3)
raw$L4N.ZMWUSD=lagpad(raw$N.ZMWUSD,4)
raw$L1DN.ZMWUSD=lagpad(raw$DN.ZMWUSD,1)
raw$L2DN.ZMWUSD=lagpad(raw$DN.ZMWUSD,2)
raw$L3DN.ZMWUSD=lagpad(raw$DN.ZMWUSD,3)
raw$L4DN.ZMWUSD=lagpad(raw$DN.ZMWUSD,4)
raw$L1N.MZNUSD=lagpad(raw$N.MZNUSD,1)
raw$L2N.MZNUSD=lagpad(raw$N.MZNUSD,2)
raw$L3N.MZNUSD=lagpad(raw$N.MZNUSD,3)
raw$L4N.MZNUSD=lagpad(raw$N.MZNUSD,4)
raw$L1DN.MZNUSD=lagpad(raw$DN.MZNUSD,1)
raw$L2DN.MZNUSD=lagpad(raw$DN.MZNUSD,2)
raw$L3DN.MZNUSD=lagpad(raw$DN.MZNUSD,3)
raw$L4DN.MZNUSD=lagpad(raw$DN.MZNUSD,4)

head(raw)

# Mutil-linear regression: ZARUSD
fitN.ZARUSD1=lm(N.ZARUSD~N.Oil+N.Au+N.Cu+N.Coc+N.AOAUSD+N.NGNUSD+N.GHSUSD+N.BWPUSD+N.MADUSD+N.XOFUSD+N.ZMWUSD+N.MZNUSD, data = raw)
summary(fitN.ZARUSD1) #Adjusted R-squared:  0.9516 

fitN.ZARUSD2=lm(N.ZARUSD~N.Oil+N.Au+N.Cu+N.AOAUSD+N.NGNUSD+N.GHSUSD+N.BWPUSD+N.XOFUSD+N.ZMWUSD+N.MZNUSD, data = raw)
summary(fitN.ZARUSD2) #Adjusted R-squared:  0.9513

fitN.ZARUSD3=lm(N.ZARUSD~N.Oil+N.Au+N.Cu+N.AOAUSD+N.NGNUSD+N.BWPUSD+N.ZMWUSD, data = raw)
summary(fitN.ZARUSD3) #Adjusted R-squared:  0.9476

fitN.ZARUSD4=lm(N.ZARUSD~N.Oil+N.Cu+N.AOAUSD+N.NGNUSD+N.BWPUSD+N.ZMWUSD, data = raw)
summary(fitN.ZARUSD4) #Adjusted R-squared:  0.9434

fitN.ZARUSD5=lm(N.ZARUSD~N.Oil+N.Au+N.Cu+N.Coc, data = raw)
summary(fitN.ZARUSD5) #Adjusted R-squared:  0.4984

# Lag1 models

fitN.ZARUSD6=lm(N.ZARUSD~L1N.Oil+L1N.Au+L1N.Cu+L1N.Coc+L1N.ZARUSD+L1N.AOAUSD+L1N.NGNUSD+L1N.GHSUSD+L1N.BWPUSD+L1N.MADUSD+L1N.XOFUSD+L1N.ZMWUSD+L1N.MZNUSD, data = raw)
summary(fitN.ZARUSD6) #Adjusted R-squared:  0.9737

fitN.ZARUSD7=lm(N.ZARUSD~L1N.Oil+L1N.Au+L1N.Cu+L1N.ZARUSD+L1N.BWPUSD+L1N.ZMWUSD, data = raw)
summary(fitN.ZARUSD7) #Adjusted R-squared:  0.9727 

fitN.ZARUSD8=lm(N.ZARUSD~L1N.Au+L1N.Cu+L1N.BWPUSD+L1N.ZMWUSD, data = raw)
summary(fitN.ZARUSD8) #Adjusted R-squared:  0.9728

# Lag2 models

fitN.ZARUSD9=lm(N.ZARUSD~L1N.Oil+L1N.Au+L1N.Cu+L1N.Coc+L1N.ZARUSD+L1N.AOAUSD+L1N.NGNUSD+L1N.GHSUSD+L1N.BWPUSD+L1N.MADUSD+L1N.XOFUSD+L1N.ZMWUSD+L1N.MZNUSD+L2N.Oil+L2N.Au+L2N.Cu+L2N.Coc+L2N.ZARUSD+L2N.AOAUSD+L2N.NGNUSD+L2N.GHSUSD+L2N.BWPUSD+L2N.MADUSD+L2N.XOFUSD+L2N.ZMWUSD+L2N.MZNUSD, data = raw)
summary(fitN.ZARUSD9) #Adjusted R-squared:  0.974

#-------------------------------

fitN.ZARUSD10=lm(N.ZARUSD~L2N.Oil+L2N.Au+L2N.Cu+L2N.AOAUSD+L2N.BWPUSD+L2N.ZMWUSD, data = raw)
summary(fitN.ZARUSD10) #Adjusted R-squared:  0.9015

fitN.ZARUSD11=lm(N.ZARUSD~L2N.Au+L2N.Cu+L2N.AOAUSD+L2N.BWPUSD+L2N.ZMWUSD, data = raw)
summary(fitN.ZARUSD11) #Adjusted R-squared:  0.9005

# Lag3 models

fitN.ZARUSD12=lm(N.ZARUSD~L3N.Oil+L3N.Au+L3N.Cu+L3N.Coc+L3N.AOAUSD+L3N.NGNUSD+L3N.GHSUSD+L3N.BWPUSD+L3N.MADUSD+L3N.XOFUSD+L3N.ZMWUSD+L3N.MZNUSD, data = raw)
summary(fitN.ZARUSD12) #Adjusted R-squared:  0.8897

fitN.ZARUSD13=lm(N.ZARUSD~L3N.Oil+L3N.Au+L3N.Cu+L3N.AOAUSD+L3N.GHSUSD+L3N.BWPUSD+L3N.MADUSD+L3N.ZMWUSD, data = raw)
summary(fitN.ZARUSD13) #Adjusted R-squared:  0.8887

fitN.ZARUSD14=lm(N.ZARUSD~L3N.Oil+L3N.Au+L3N.Cu+L3N.AOAUSD+L3N.BWPUSD+L3N.MADUSD, data = raw)
summary(fitN.ZARUSD14) #Adjusted R-squared:  0.8822

fitN.ZARUSD15=lm(N.ZARUSD~L3N.Oil+L3N.Cu+L3N.BWPUSD+L3N.MADUSD, data = raw)
summary(fitN.ZARUSD15) #Adjusted R-squared:  0.8816

fitN.ZARUSD16=lm(N.ZARUSD~L3N.Cu+L3N.BWPUSD+L3N.MADUSD, data = raw)
summary(fitN.ZARUSD16) #Adjusted R-squared:  0.8804

# Lag4 models...

library(ranger)
library(xgboost)
library(ggplot2)
library(MASS)
library(latex2exp)
library(ipw)
library(cowplot)
library(tidyverse)
library(gbm)
library(DoubleML)
library(mlr3)
library(mlr3learners)


set.seed(192748)
# simulate ground truth
n <- 100000
# generate sigma matrix
p <- 4
Sigma <- matrix(0,p,p)
for (i in 1:p){
  for(j in 1:p){
    Sigma[i,j] <- 0.5^(abs(i-j))
  }
}
X <- mvrnorm( n = n, mu = rep(0,p), Sigma = Sigma )
# treatment
D <- rbinom(n,1,plogis( X[,1] - 0.5*X[,2] +0.25*X[,3] + 0.1*X[,4])) 
EY <- 21 + D + 27.4*X[,1] + 13.7*X[,2] + 13.7*X[,3] + 13.7*X[,4]
EY1 <- 21 + 1 + 27.4*X[,1] + 13.7*X[,2] + 13.7*X[,3] + 13.7*X[,4]
EY0 <- 21 + 0 + 27.4*X[,1] + 13.7*X[,2] + 13.7*X[,3] + 13.7*X[,4]
Y <- rbinom(n,1, plogis(EY))
Y1 <- rbinom(n,1, plogis(EY1))
Y0 <- rbinom(n,1, plogis(EY0))

# generate subgroups
A1<- X[,1] > quantile(X[,1],0.1)
E.A1 <- sum(Y1[A1])/sum(A1)
A2 <- (X[,2] > quantile(X[,2],0.1)) &(X[,1] < quantile(X[,2],0.9))
E.A2 <- sum(Y1[A2])/sum(A2)
A3 <- (X[,3] + X[,4]) > -2
E.A3 <- sum(Y1[A3])/sum(A3)
A4 <- X[,4] > -1
E.A4 <- sum(Y1[A4])/sum(A4)

truePsi <- c(E.A1,E.A2,E.A3,E.A4)
truePsi


## helper functions ---------------------------------
## GLM
ml_g_glm = lrn("regr.glmnet")
ml_m_glm = lrn("classif.glmnet")


# Random forest learners
ml_g_rf = lrn("regr.ranger")
ml_m_rf = lrn("classif.ranger")

# Xgboost learners
ml_g_xgb = lrn("regr.xgboost", objective = "reg:squarederror")
ml_m_xgb = lrn("classif.xgboost", objective = "binary:logistic",
               eval_metric = "logloss")

# self-specified score
p1_f <- function(y,d,g0_hat,g1_hat,m_hat,smpls){
  res <- d*(y-g1_hat)/m_hat + g1_hat
  res.list <- list(psi_a = rep(-1,length(y)), psi_b = res)
  return(res.list)
} 

# moderate subgroups
Estimator <- function(n){
  # Generate Data
  p <- 4
  Sigma <- matrix(0,p,p)
  for (i in 1:p){
    for(j in 1:p){
      Sigma[i,j] <- 0.5^(abs(i-j))
    }
  }
  X <- mvrnorm( n = n, mu = rep(0,p), Sigma = Sigma )
  colnames(X) <- c("X1","X2","X3","X4")
  # treatment
  D <- rbinom(n,1,plogis( X[,1] - 0.5*X[,2] +0.25*X[,3] + 0.1*X[,4])) 
  EY <- 21 + D + 27.4*X[,1] + 13.7*X[,2] + 13.7*X[,3] + 13.7*X[,4]
  Y <- rbinom(n,1, plogis(EY))
  
  # generate subgroups indices
  A1<- X[,1] > quantile(X[,1],0.1)
  
  A2 <- (X[,2] > quantile(X[,2],0.1)) &(X[,2] < quantile(X[,2],0.9))
  
  A3 <- (X[,3] + X[,4]) > -2
  
  A4 <- X[,4] > -1
  
  P1 <- sum(A1)/length(A1)
  P2<- sum(A2)/length(A2)
  P3<- sum(A3)/length(A3)
  P4 <- sum(A4)/length(A4)
  
  
  # misspecified covariates
  Z1 <- exp(X[,1]/2) 
  Z2 <- X[,2]/(1+exp(X[,1])) + 10
  Z3 <- (X[,1]*X[,3]/25 + 0.6)^3 
  Z4 <- (X[,2]+ X[,4] +20)^2
  Z <- cbind(Z1,Z2,Z3,Z4)

  
  dat <- data.frame(cbind(Y,Z,D))
  dml_data_df <- double_ml_data_from_data_frame(dat,
                                                y_col='Y',
                                                d_cols='D',
                                                x_cols=c('Z1','Z2','Z3','Z4'))
 # Model
  dml_irm_forest = DoubleMLIRM$new(dml_data_df,
                                   #score = "ATE",
                                   score = p1_f,
                                   ml_g = ml_g_rf,
                                   ml_m = ml_m_rf)
  # Estimation
  dml_irm_forest$fit()
  
  DML.A1 <- sum(dml_irm_forest$psi_b * A1)/ sum(A1)
  DML.A2<- sum(dml_irm_forest$psi_b * A2)/ sum(A2)
  DML.A3 <- sum(dml_irm_forest$psi_b * A3)/ sum(A3)
  DML.A4 <- sum(dml_irm_forest$psi_b * A4)/ sum(A4)
  
  E.A1.IC <- A1/P1*(dml_irm_forest$psi_b )
  E.A2.IC <- A2/P2*(dml_irm_forest$psi_b )
  E.A3.IC <- A3/P3*(dml_irm_forest$psi_b )
  E.A4.IC <- A4/P4*(dml_irm_forest$psi_b )
  
  cor.mat <- cor(cbind(E.A1.IC,E.A2.IC,E.A3.IC,E.A4.IC))
  Z <- mvrnorm( n, mu = rep(0,4), Sigma = cor.mat )
  kappa<- quantile(apply(Z,1, max),0.975)
  
  cov.mat <- cov(cbind(E.A1.IC,E.A2.IC,E.A3.IC,E.A4.IC))
  CI.length <- kappa * c(sqrt(cov.mat[1,1]),sqrt(cov.mat[2,2]),
                         sqrt(cov.mat[3,3]),sqrt(cov.mat[4,4]))/sqrt(n)
  
  
  DML.est <- c(DML.A1,DML.A2,DML.A3,DML.A4)
  LB <- DML.est-CI.length
  
  cover <- NULL
  for(b in 1:4){
    cover[b] <- (LB[b]<truePsi[b])
  }
  
  
  return(c(DML.A1,DML.A2,DML.A3,DML.A4))
}

Niter <-1000
size <- c(1500,2000,2500,3000)
size <- c(1500)


result <- matrix(NA,ncol = 4*1,nrow = length(size))
SDresult <- matrix(NA,ncol = 4*1,nrow = length(size))
length.res <- matrix(NA,ncol = 4*1,nrow = length(size))


for (j in 1: length(size)){
  print(size[j])
  res <- matrix(NA,ncol = 4*1,nrow = Niter)
  
  for (i in 1:Niter){
    res[i,] <- Estimator(size[j])
  }
  
  result[j,] <- colMeans(res,na.rm = TRUE)
  SDresult[j,] <- apply(res,2, function(x) sd(x,na.rm = TRUE))

  
}



bias <- matrix(NA,nrow = length(size), ncol = 1)
sd <- matrix(NA,nrow = length(size), ncol = 1)
length<- matrix(NA,nrow = length(size), ncol = 1)
coverage<- matrix(NA,nrow = length(size), ncol = 1)


for (i in 1:length(size)){
  bias[i,] <- c(sum(abs(result[i,1:4] - truePsi)))
                #sum(abs(result[i,5:8] - truePsi))

  sd[i,] <- c(sum(abs(SDresult[i,1:4])))
              #sum(abs(SDresult[i,5:8]))

  
 # coverage[i,] <- c(mean(result[i,1:4]))
  
}




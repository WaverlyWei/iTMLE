library(ranger)
library(xgboost)
library(ggplot2)
library(MASS)
library(latex2exp)
library(ipw)
library(cowplot)
library(tidyverse)
library(gbm)


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
# generate subgroups
percentile <- seq(0,1,0.1) # interval case
C <- quantile(X[,1],probs = percentile) # subgroup threshold
truePsi <- NULL
for (i in 1:10){
  u1 <- sum(Y1*(X[,1]>=C[i] &X[,1]<=C[i+1] ))/sum(X[,1]>=C[i] &X[,1]<=C[i+1] )
  truePsi[i] <- mean(u1)
}

truePsi


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
  # estimate CDF 
  f <- ecdf(X[,1])
  C <- seq(0,1,0.1) # interval case
  C <- quantile(X[,1],probs = C) # subgroup threshold
  
  
  # misspecified covariates
  Z1 <- exp(X[,1]/2) 
  Z2 <- X[,2]/(1+exp(X[,1])) + 10
  Z3 <- (X[,1]*X[,3]/25 + 0.6)^3 
  Z4 <- (X[,2]+ X[,4] +20)^2
  Z <- cbind(Z1,Z2,Z3,Z4)
  
  psModel <- glm(D~X,family = "binomial")
  #psModel <- ranger(x=X,y=D)
  e1 <- predict(psModel,data = as.data.frame(X),type="response")
  #e1 <- e1$predictions
  
  ## GB -----------------
  #psModel <- xgboost(data = X, label = D, nrounds  = 500)
  #psModel <- gbm(D~.,data = data.frame(cbind(D,X)),distribution = "bernoulli")
  #e1 <- predict(psModel,newdata = data.frame(X),type="response")
  
  muModel <- glm(Y~ D+Z,family = "binomial")
  #muModel <- ranger(x=cbind(D,Z),y=Y)
  mu <- predict(muModel,data = cbind(D,Z),type="response")
  #mu <- mu$predictions
  
  ## GB---------------------
  #muModel <- xgboost(data=cbind(D,Z),label=Y,nrounds = 50)
  #muModel <- gbm(Y~.,data = data.frame(cbind(Y,D,Z)),distribution = "bernoulli")
  #mu <- predict(muModel,newdata = data.frame(cbind(D,Z)),type="response")
  
  
  mat1 <- data.frame(cbind(D,Z))
  mat1[,1] <- rep(1,nrow(mat1))
  #mu1 <- predict(muModel,data = mat1,type="response")
  #mu1 <- mu1$predictions
  
  ## GB ----------------
  mu1 <- predict(muModel,newdata = mat1,type="response")
  
  mu1[mu1==1] <- 0.9999
  mu1[mu1==0] <- 0.0001
  mu[mu==1] <- 0.9999
  mu[mu==0] <- 0.0001
  e1[e1==1] <- 0.9999
  e1[e1==0] <- 0.0001
  
  
  # iterative tmle =======================================
  mu1.tmle <- mu1
  phi_all <- NULL
  S_all <- NULL
  
  Iter <- 300
  neg.log.like.old <- -100
  #delta <- delta0 <- 0.00001 # delta for mis-outcome model
  delta <- delta0 <- 0.000001 # delta for mis-propensity model
  
  temp <- matrix(NA,ncol = 2,nrow = Iter)
  
  for (j in 1:Iter){
    
    for (i in 1:10){
      P <- f(C[i+1]) - f(C[i])
      # prep for multi-subgroup
      phi_all[[i]] <-  (X[,1]>=C[i] & X[,1] <=C[i+1])/P * D/e1*(Y-mu1.tmle)
      S_all[[i]] <- (X[,1]>=C[i] & X[,1] <=C[i+1])/P * D/e1
    }
    
    correction<- sqrt(mean(phi_all[[1]])^2 +mean(phi_all[[2]])^2+ 
                        mean(phi_all[[3]])^2+mean(phi_all[[4]])^2+
                        mean(phi_all[[5]])^2+
                        +mean(phi_all[[6]])^2+
                        mean(phi_all[[7]])^2+mean(phi_all[[8]])^2+
                        mean(phi_all[[9]])^2+mean(phi_all[[10]])^2)
    
    S <- sum(S_all[[1]]*mean(phi_all[[1]]) + S_all[[2]]*mean(phi_all[[2]])+
               S_all[[3]]*mean(phi_all[[3]]) + S_all[[4]]*mean(phi_all[[4]])+
               S_all[[5]]*mean(phi_all[[5]]) + S_all[[6]]*mean(phi_all[[6]])+
               S_all[[7]]*mean(phi_all[[7]]) + S_all[[8]]*mean(phi_all[[8]])+
               S_all[[9]]*mean(phi_all[[9]]) + S_all[[10]]*mean(phi_all[[10]]))/correction
    
    #S <- S * max(unlist(lapply(S_all, var)))*2
    
    neg.log.like.new <- sum(Y* (qlogis(mu1.tmle) + delta * S) -
                              log(1+exp(qlogis(mu1.tmle) + delta *S)))
    
    if(neg.log.like.new > neg.log.like.old){
      mu1.tmle <- plogis(qlogis(mu1.tmle) + delta*S)
      neg.log.like.old <- neg.log.like.new
    }else{
      delta <- -delta0
    }
    
    sub.est <- NULL
    for (m in 1:10){
      sub.est[m] <- sum((X[,1]>=C[m] & X[,1] <=C[m+1])*mu1.tmle,na.rm = TRUE)/sum((X[,1]>=C[m] & X[,1] <=C[m+1]))
    }
    
    temp[j,]<- c(sum(abs(sub.est-truePsi)),neg.log.like.old)
    
  }
  
  #plot(temp[,1],temp[,2])
  
  mustar <- mu1.tmle
  
  E.TMLE <- NULL
  for (m in 1:10){
    E.TMLE[m] <- sum((X[,1]>=C[m] & X[,1] <=C[m+1])*mustar,na.rm = TRUE)/sum((X[,1]>=C[m] & X[,1] <=C[m+1]))
  }
  
  E.TMLE.IC <- matrix(NA,nrow =n,ncol =10)
  for (m in 1:10){
    E.TMLE.IC[,m] <- (X[,1]>=C[m] & X[,1] <=C[m+1])/sum((X[,1]>=C[m] & X[,1] <=C[m+1]))*(D/e1*(Y-mustar)+mustar)
  }
  
  cor.mat <- cor(E.TMLE.IC)
  Z <- mvrnorm( n, mu = rep(0,10), Sigma = cor.mat )
  kappa<- quantile(apply(Z,1, max),0.975)
  
  cov.mat <- cov(E.TMLE.IC)
  CI.length <- kappa * c(sqrt(cov.mat[1,1]),sqrt(cov.mat[2,2]),
                                            sqrt(cov.mat[3,3]),sqrt(cov.mat[4,4]),
                                            sqrt(cov.mat[5,5]),
                                            sqrt(cov.mat[6,6]),
                                            sqrt(cov.mat[7,7]),sqrt(cov.mat[8,8]),
                                            sqrt(cov.mat[9,9]),sqrt(cov.mat[10,10]))/sqrt(n)

  LB <- NULL
  TMLE.est <- E.TMLE
  for(b in 1:10){
    LB[b] <- ((TMLE.est[b] - CI.length[b])<truePsi[b])
    #LB[b] <- TMLE.est[b] - CI.length[b]
  }
  # ====================================================
  

  ## DR
  E.DR <- NULL
  for (m in 1:10){
    E.DR[m] <- sum((X[,1]>=C[m] & X[,1] <=C[m+1])*(D/e1*(Y-mu1)+mu1),na.rm = TRUE)/sum((X[,1]>=C[m] & X[,1] <=C[m+1]))
  }
  
  E.DR.IC <- matrix(NA,nrow =n,ncol =10)
  for (m in 1:10){
    E.DR.IC[,m] <- (X[,1]>=C[m] & X[,1] <=C[m+1])/sum((X[,1]>=C[m] & X[,1] <=C[m+1]))*(D/e1*(Y-mu1)+mu1)
  }
  

  # identity matrix
  Sigma <- matrix(0,10,10)
  diag(Sigma) <- 1
  
  Z.DR <- mvrnorm( n, mu = rep(0,10), Sigma = Sigma )
  kappa.DR <- quantile(apply(Z.DR,1, max),0.975)
  
  cov.mat.DR <- cov(E.DR.IC)
  CI.length.DR <- kappa.DR * c(sqrt(cov.mat.DR[1,1]),
                               sqrt(cov.mat.DR[2,2]),
                               sqrt(cov.mat.DR[3,3]),
                               sqrt(cov.mat.DR[4,4]),
                               sqrt(cov.mat.DR[5,5]),
                               sqrt(cov.mat.DR[6,6]),
                               sqrt(cov.mat.DR[7,7]),
                               sqrt(cov.mat.DR[8,8]),
                               sqrt(cov.mat.DR[9,9]),
                               sqrt(cov.mat.DR[10,10]))/sqrt(n)
  
  LB.DR <- NULL
  DR.est <- E.DR
  for(b in 1:10){
    LB.DR[b] <- ((DR.est[b] - CI.length.DR[b]) < truePsi[b])
    #LB.DR[b] <- DR.est[b] - CI.length.DR[b]
  }
  
  
  # -------------------------------------------
  
  return(c(E.TMLE,E.DR))
}



EstimatorGLM <- function(n){
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
  # estimate CDF 
  f <- ecdf(X[,1])
  C <- seq(0,1,0.1) # interval case
  C <- quantile(X[,1],probs = C) # subgroup threshold
  
  
  # misspecified covariates
  Z1 <- exp(X[,1]/2) 
  Z2 <- X[,2]/(1+exp(X[,1])) + 10
  Z3 <- (X[,1]*X[,3]/25 + 0.6)^3 
  Z4 <- (X[,2]+ X[,4] +20)^2
  Z <- cbind(Z1,Z2,Z3,Z4)
  cbind(Z1,Z2,Z3,Z4)

  muModel <- glm(Y~ D+Z,family = "binomial")
  mu <- predict(muModel,data = cbind(D,Z),type="response")
  
  
  mat1 <- data.frame(cbind(D,Z))
  mat1[,1] <- rep(1,nrow(mat1))
  mu1 <- predict(muModel,data = mat1,type="response",se.fit = TRUE)
  se.mu1 <- mean(mu1$se.fit)
  mu1 <- mu1$fit
  

  E.GLM <- NULL
  for(m in 1:10){
    E.GLM[m] <- sum((X[,1]>=C[m] & X[,1] <=C[m+1])*mu1,na.rm = TRUE)/sum((X[,1]>=C[m] & X[,1] <=C[m+1]))
  }
  
  
  Sigma <- matrix(0,10,10)
  diag(Sigma) <- 1
  
  Z.DR <- mvrnorm( n, mu = rep(0,10), Sigma = Sigma )
  kappa.DR <- quantile(apply(Z.DR,1, max),0.975)
  
  LB.glm <- E.GLM - se.mu1*1.96/sqrt(n)
  
  cover <- NULL
  for(b in 1:10){
    cover[b] <- (LB.glm[b] < truePsi[b])
  }
  
 # return(E.GLM)
  return(cover)
}

Niter <- 1000
size <- c(1500,2000,2500,3000)
result <- matrix(NA,ncol = 10*2,nrow = length(size))
SDresult <- matrix(NA,ncol = 10*2,nrow = length(size))
length.res <- matrix(NA,ncol = 10*2,nrow = length(size))


for (j in 1: length(size)){
  print(size[j])
  res <- matrix(NA,ncol = 10*2,nrow = Niter)
  
  for (i in 1:Niter){
    #res[i,] <- EstimatorGLM(size[j])
    res[i,] <- Estimator(size[j])
  }
  
  result[j,] <- colMeans(res,na.rm = TRUE)
  SDresult[j,] <- apply(res,2, function(x) sd(x,na.rm = TRUE))
}


bias <- matrix(NA,nrow = length(size), ncol = 1)
sd <- matrix(NA,nrow = length(size), ncol = 1)
coverage<- matrix(NA,nrow = length(size), ncol = 1)

for (i in 1:length(size)){
  bias[i,] <- c(sum(abs(result[i,1:10] - truePsi)))
                #sum(abs(result[i,11:20] - truePsi)))

  sd[i,] <- c(sum(abs(SDresult[i,1:10])))
              #sum(abs(SDresult[i,11:20])))
 
  # coverage[i,] <- c(mean(result[i,1:10]))
                   # mean(result[i,11:20]))
   
}


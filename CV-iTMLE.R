library(ranger)
library(xgboost)
library(ggplot2)
library(MASS)
library(latex2exp)
library(ipw)
library(cowplot)
library(tidyverse)
library(gbm)
library(caret)


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


# moderate subgroups
Estimator <- function(n,folds){
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
  
  # misspecified covariates
  Z1 <- exp(X[,1]/2) 
  Z2 <- X[,2]/(1+exp(X[,1])) + 10
  Z3 <- (X[,1]*X[,3]/25 + 0.6)^3 
  Z4 <- (X[,2]+ X[,4] +20)^2
  Z <- cbind(Z1,Z2,Z3,Z4)
  
  
  ## create fold list
  fold.list <- createFolds(Y, k = folds)
  
  mu1.tmle.all <- matrix(NA,nrow = folds,ncol = 4)
  ci.length.tmle.all <- matrix(NA,nrow = folds,ncol = 4)
  
  for(j in 1:folds)
  { 
    Y.train <- Y[-fold.list[[j]]]
    D.train <- D[-fold.list[[j]]]
    X.train <- X[-fold.list[[j]],]
    Z.train <- Z[-fold.list[[j]],]
    
    Y.val <- Y[fold.list[[j]]]
    D.val <- D[fold.list[[j]]]
    X.val <- X[fold.list[[j]],]
    Z.val <- Z[fold.list[[j]],]
  
    A1.val <-  A1[fold.list[[j]]]
    A2.val <-  A2[fold.list[[j]]]
    A3.val <-  A3[fold.list[[j]]]
    A4.val <-  A4[fold.list[[j]]]
  
    #psModel <- glm(D.train~X.train,family = "binomial")
    psModel <- ranger(x=X.train,y=D.train)
    e1.val <- predict(psModel,data = as.data.frame(X.val),type="response")
    e1.val <- e1.val$predictions
  
  ## GB -----------------
  #psModel <- xgboost(data = X.train, label = D.train, nrounds  = 500)
  #psModel <- gbm(D.train~.,data = data.frame(cbind(D.train,X.train)),distribution = "bernoulli")
  #e1.val <- predict(psModel,newdata = data.frame(X.val),type="response")
  
    train.DZ <- cbind(D.train,Z.train)
    colnames(train.DZ)[1] <-"D"
    val.DZ <- cbind(D.val,Z.val)
    colnames(val.DZ)[1] <-"D"
    muModel <- ranger(x=train.DZ,y=Y.train)
    #muModel <- glm(Y.train~ train.DZ,family = "binomial")
    mu.val <- predict(muModel,data = as.data.frame(val.DZ),type="response")
    mu.val <- mu.val$predictions
  
  ## GB---------------------
  #muModel <- xgboost(data=cbind(D,Z),label=Y,nrounds = 50)
  
    #train.YDZ <- cbind(Y.train,D.train,Z.train)
    #colnames(train.YDZ)[1:2] <-c("Y","D")
    #val.DZ <- cbind(D.val,Z.val)
    #colnames(val.DZ)[1] <-"D"
    #muModel <- gbm(Y~.,data = data.frame(train.YDZ),distribution = "bernoulli")
    #mu.val <- predict(muModel,newdata = data.frame(val.DZ),type="response")

    val.mat1 <- cbind(D.val,Z.val)
    val.mat1[,1] <- rep(1,nrow(val.mat1))
    colnames(val.mat1)[1] <-"D"
    mu1.val <- predict(muModel,data = as.data.frame(val.mat1),type="response")
    mu1.val <- mu1.val$predictions
    
  #mu1.val <- predict(muModel,newdata = as.data.frame(val.mat1),type="response")
  #mu1.val <- mu1.val$predictions
  
 
  mu1.val[mu1.val==1] <- 0.9999
  mu1.val[mu1.val==0] <- 0.0001
  mu.val[mu.val==1] <- 0.9999
  mu.val[mu.val==0] <- 0.0001
  e1.val[e1.val==1] <- 0.9999
  e1.val[e1.val==0] <- 0.0001
  
  
  # P1.val <- sum(A1.val)/length(A1.val)
  # P2.val<- sum(A2.val)/length(A2.val)
  # P3.val<- sum(A3.val)/length(A3.val)
  # P4.val <- sum(A4.val)/length(A4.val)
  
  P1.val <- sum(A1)/length(A1)
  P2.val<- sum(A2)/length(A2)
  P3.val<- sum(A3)/length(A3)
  P4.val <- sum(A4)/length(A4)
  
  # iterative tmle =======================================
  S1<- A1.val/P1.val * D.val/e1.val
  S2<- A2.val/P2.val * D.val/e1.val
  S3 <- A3.val/P3.val * D.val/e1.val
  S4<- A4.val/P4.val * D.val/e1.val
  
  mu1.tmle <- mu1.val
  
  Iter <- 300
  neg.log.like.old <- -100
  delta <- delta0 <- 0.00001
  
  temp <- matrix(NA,ncol = 2,nrow = Iter)
  
  for (k in 1:Iter){
    
    phi.1 <- A1.val/P1.val * D.val/e1.val * (Y.val-mu1.tmle)
    phi.2 <- A2.val/P2.val * D.val/e1.val * (Y.val-mu1.tmle)
    phi.3 <- A3.val/P3.val * D.val/e1.val * (Y.val-mu1.tmle)
    phi.4 <- A4.val/P4.val * D.val/e1.val * (Y.val-mu1.tmle)
    
    correction<- sqrt(mean(phi.1)^2 +mean(phi.2)^2+ mean(phi.3)^2+mean(phi.4)^2)
    
    S <- sum(S1 * mean(phi.1) + S2 *mean(phi.2) + S3*mean(phi.3) + S4 * mean(phi.4))/correction
    
    neg.log.like.new <- sum(Y.val* (qlogis(mu1.tmle) + delta * S) -
                              log(1+exp(qlogis(mu1.tmle) + delta *S)))
    
    if(neg.log.like.new > neg.log.like.old){
      #mu1.tmle <- plogis(logit.mu1 + epsilon1*S)
      mu1.tmle <- plogis(qlogis(mu1.tmle) + delta*S)
      #delta.old <- delta.new
      neg.log.like.old <- neg.log.like.new
    }else{
      delta <- -delta0
    }
    
    #temp[j,] <- c(mean(mu1.tmle),neg.log.like.new)
    
    # temp[k,]<- c(sum(abs(c(sum(A1.val*mu1.tmle,na.rm = TRUE)/sum(A1.val),
    #                        sum(A2.val*mu1.tmle,na.rm = TRUE)/sum(A2.val),
    #                        sum(A3.val*mu1.tmle,na.rm = TRUE)/sum(A3.val),
    #                        sum(A4.val*mu1.tmle,na.rm = TRUE)/sum(A4.val))-truePsi)),neg.log.like.old)
    # 
  }
  
  #plot(temp[,1],temp[,2])
  
  
  mustar <- mu1.tmle
  
  E.A1 <-  sum(A1.val*mustar,na.rm = TRUE)/sum(A1.val)
  E.A2 <-  sum(A2.val*mustar,na.rm = TRUE)/sum(A2.val)
  E.A3 <-  sum(A3.val*mustar,na.rm = TRUE)/sum(A3.val)
  E.A4 <-  sum(A4.val*mustar,na.rm = TRUE)/sum(A4.val)
  
  mu1.tmle.all[j,] <- c(E.A1,E.A2,E.A3,E.A4)
  
  E.A1.IC <- A1.val/P1.val*(D.val/e1.val*(Y.val-mustar)+mustar-mean(mustar))
  E.A2.IC <- A2.val/P2.val*(D.val/e1.val*(Y.val-mustar)+mustar-mean(mustar))
  E.A3.IC <- A3.val/P3.val*(D.val/e1.val*(Y.val-mustar)+mustar-mean(mustar))
  E.A4.IC <- A4.val/P4.val*(D.val/e1.val*(Y.val-mustar)+mustar-mean(mustar))
  
  cor.mat <- cor(cbind(E.A1.IC,E.A2.IC,E.A3.IC,E.A4.IC))
  Z <- mvrnorm( n, mu = rep(0,4), Sigma = cor.mat )
  kappa<- quantile(apply(Z,1, max),0.975)
  
  cov.mat <- cov(cbind(E.A1.IC,E.A2.IC,E.A3.IC,E.A4.IC))
  CI.length <- kappa * c(sqrt(cov.mat[1,1]),sqrt(cov.mat[2,2]),
                         sqrt(cov.mat[3,3]),sqrt(cov.mat[4,4]))/sqrt(n)
  
  ci.length.tmle.all[j,] <- CI.length
  
  # ====================================================
  }
  
  
  # ## DR
  # E.A1.DR <- sum(A1*(D/e1*(Y-mu1)+mu1),na.rm = TRUE)/sum(A1)
  # E.A2.DR <- sum(A2*(D/e1*(Y-mu1)+mu1),na.rm = TRUE)/sum(A2)
  # E.A3.DR <- sum(A3*(D/e1*(Y-mu1)+mu1),na.rm = TRUE)/sum(A3)
  # E.A4.DR <- sum(A4*(D/e1*(Y-mu1)+mu1),na.rm = TRUE)/sum(A4)
  
  #c(E.A1.DR,E.A2.DR,E.A3.DR,E.A4.DR)-truePsi
  # -------------------------------------------
  
  #return(colMeans(mu1.tmle.all))
  
  LB <- colMeans(mu1.tmle.all)-colMeans(ci.length.tmle.all)
  
  cover <- NULL
  TMLE.est <- colMeans(mu1.tmle.all)
  for(b in 1:4){
    cover[b] <- (LB[b]<truePsi[b])
    #LB[b] <- TMLE.est[b] - CI.length[b]
  }
  
  return(c(TMLE.est,cover))
}

Niter <- 1000
size <- c(1500,2000,2500,3000)


result <- matrix(NA,ncol = 4*2,nrow = length(size))
SDresult <- matrix(NA,ncol = 4*2,nrow = length(size))
length.res <- matrix(NA,ncol = 4*2,nrow = length(size))


for (j in 1: length(size)){
  print(size[j])
  res <- matrix(NA,ncol = 4*2,nrow = Niter)
  
  for (i in 1:Niter){
    res[i,] <- Estimator(size[j],folds = 2)
  }
  
  result[j,] <- colMeans(res,na.rm = TRUE)
  SDresult[j,] <- apply(res,2, function(x) sd(x,na.rm = TRUE))

}



bias <- matrix(NA,nrow = length(size), ncol = 1)
sd <- matrix(NA,nrow = length(size), ncol = 1)
coverage<- matrix(NA,nrow = length(size), ncol = 1)

for (i in 1:length(size)){
  bias[i,] <- c(sum(abs(result[i,1:4] - truePsi)))

  sd[i,] <- c(sum(abs(SDresult[i,1:4])))
            
  coverage[i,] <- c(mean(result[i,5:8]))
}


## bias and variance 
bias[,1] <- sqrt(size) * bias[,1]
sd[,1] <- sqrt(size) * sd[,1]
length[,1] <- sqrt(size) * length[,1]



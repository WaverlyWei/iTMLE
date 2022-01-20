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
  mu1 <- predict(muModel,data = mat1,type="response")
  #mu1 <- mu1$predictions
  
  ## GB ----------------
  #mu1 <- predict(muModel,newdata = mat1,type="response")
  
  mu1[mu1==1] <- 0.9999
  mu1[mu1==0] <- 0.0001
  mu[mu==1] <- 0.9999
  mu[mu==0] <- 0.0001
  e1[e1==1] <- 0.9999
  e1[e1==0] <- 0.0001
  
  
  P1 <- sum(A1)/length(A1)
  P2 <- sum(A2)/length(A2)
  P3<- sum(A3)/length(A3)
  P4 <- sum(A4)/length(A4)
  
  # if(P1==0 | P2== 0 | P3==0|P4==0){
  #   return(c(NA,NA,NA,NA))
  # }
  
  
  # iterative tmle =======================================
  S1 <- A1/P1 * D/e1
  S2 <- A2/P2 * D/e1
  S3 <- A3/P3 * D/e1
  S4 <- A4/P4 * D/e1
  
  mu1.tmle <- mu1
  
  Iter <- 300
  neg.log.like.old <- -100
  #delta <- delta0 <- 0.00001  # delta for mis-outcome model
  delta <- delta0 <- 0.00000001 # delta for mis-propensity model
  
  temp <- matrix(NA,ncol = 2,nrow = Iter)
  
  for (j in 1:Iter){
    
    phi.1 <- A1/P1 * D/e1 * (Y-mu1.tmle)
    phi.2 <- A2/P2 * D/e1 * (Y-mu1.tmle)
    phi.3 <- A3/P3 * D/e1 * (Y-mu1.tmle)
    phi.4 <- A4/P4 * D/e1 * (Y-mu1.tmle)
    
    correction<- sqrt(mean(phi.1)^2 +mean(phi.2)^2+ mean(phi.3)^2+mean(phi.4)^2)
    
    S <- sum(S1 * mean(phi.1) + S2 *mean(phi.2) + S3*mean(phi.3) + S4 * mean(phi.4))/correction
    
    # S <- S1*phi.1/correction + 
    #   S2*phi.2/correction+
    #   S3*phi.3/correction+
    #   S4*phi.4/correction
    
    
    #logit.mu1[logit.mu1 == Inf] <- 36.044
    #logit.mu1[logit.mu1 == -Inf] <- -16.815
    
    # method 1: compute through glm
    # epsilon1 <- coef(glm(Y~-1 + offset(logit.mu1) + S, 
    #                      family="binomial"))
    # log.like.new <- mean(Y* (qlogis(mu1.tmle) + epsilon1 * S) - 
    #                        log(1+exp(qlogis(mu1.tmle) + epsilon1*S)))
    
    # method 2: update delta
    #delta.new <- delta.old* S
    
    neg.log.like.new <- sum(Y* (qlogis(mu1.tmle) + delta * S) -
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
    
    #mu1.tmle[mu1.tmle==1] <- 0.9999
    #mu1.tmle[mu1.tmle==0] <- 0.0001
    
    temp[j,]<- c(sum(abs(c(sum(A1*mu1.tmle,na.rm = TRUE)/sum(A1),
                           sum(A2*mu1.tmle,na.rm = TRUE)/sum(A2),
                           sum(A3*mu1.tmle,na.rm = TRUE)/sum(A3),
                           sum(A4*mu1.tmle,na.rm = TRUE)/sum(A4))-truePsi)),neg.log.like.old)

  }
  
  
  mustar <- mu1.tmle
  
  E.A1 <-  sum(A1*mustar,na.rm = TRUE)/sum(A1)
  E.A2 <-  sum(A2*mustar,na.rm = TRUE)/sum(A2)
  E.A3 <-  sum(A3*mustar,na.rm = TRUE)/sum(A3)
  E.A4 <-  sum(A4*mustar,na.rm = TRUE)/sum(A4)
  
  E.A1.IC <- A1/P1*(D/e1*(Y-mustar)+mustar-mean(mustar))
  E.A2.IC <- A2/P2*(D/e1*(Y-mustar)+mustar-mean(mustar))
  E.A3.IC <- A3/P3*(D/e1*(Y-mustar)+mustar-mean(mustar))
  E.A4.IC <- A4/P4*(D/e1*(Y-mustar)+mustar-mean(mustar))
  
  cor.mat <- cor(cbind(E.A1.IC,E.A2.IC,E.A3.IC,E.A4.IC))
  Z <- mvrnorm( n, mu = rep(0,4), Sigma = cor.mat )
  kappa<- quantile(apply(Z,1, max),0.975)

  cov.mat <- cov(cbind(E.A1.IC,E.A2.IC,E.A3.IC,E.A4.IC))
  CI.length <- kappa * c(sqrt(cov.mat[1,1]),sqrt(cov.mat[2,2]),
                         sqrt(cov.mat[3,3]),sqrt(cov.mat[4,4]))/sqrt(n)

  
  LB <- NULL
  TMLE.est <- c(E.A1,E.A2,E.A3,E.A4)
  for(b in 1:4){
    LB[b] <- ((TMLE.est[b] - CI.length[b])<truePsi[b])
  }
# ====================================================
 
  
  ## DR
  E.A1.DR <- sum(A1*(D/e1*(Y-mu1)+mu1),na.rm = TRUE)/sum(A1)
  E.A2.DR <- sum(A2*(D/e1*(Y-mu1)+mu1),na.rm = TRUE)/sum(A2)
  E.A3.DR <- sum(A3*(D/e1*(Y-mu1)+mu1),na.rm = TRUE)/sum(A3)
  E.A4.DR <- sum(A4*(D/e1*(Y-mu1)+mu1),na.rm = TRUE)/sum(A4)
  
  E.A1.DR.sd <- sd(A1/P1*(D/e1*(Y-mu1)+mu1-mean(mu1)),na.rm = TRUE)
  E.A2.DR.sd <- sd(A2/P2*(D/e1*(Y-mu1)+mu1-mean(mu1)),na.rm = TRUE)
  E.A3.DR.sd <- sd(A3/P3*(D/e1*(Y-mu1)+mu1-mean(mu1)),na.rm = TRUE)
  E.A4.DR.sd <- sd(A4/P4*(D/e1*(Y-mu1)+mu1-mean(mu1)),na.rm = TRUE)
  
  Sigma <- matrix(0,4,4)
  diag(Sigma) <- 1
  Z.DR <- mvrnorm( n, mu = rep(0,4), Sigma = Sigma )
  kappa.DR <- quantile(apply(Z.DR,1, max),0.975)
  CI.length.DR <- kappa.DR * c(E.A1.DR.sd,E.A2.DR.sd,E.A3.DR.sd,E.A4.DR.sd)/sqrt(n)
  
  LB.DR <- NULL
  DR.est <- c(E.A1.DR,E.A2.DR,E.A3.DR,E.A4.DR)
  #DR.sd <- c(E.A1.DR.sd,E.A2.DR.sd,E.A3.DR.sd,E.A4.DR.sd)
  for(b in 1:4){
    LB.DR[b] <- ((DR.est[b] - CI.length.DR[b]) < truePsi[b])
    #LB.DR[b] <- DR.est[b] - CI.length.DR[b]
  }
  
  
  
  #c(E.A1.DR,E.A2.DR,E.A3.DR,E.A4.DR)-truePsi
  # -------------------------------------------
  
  # update each subgroup separately ---------------------
  logit.mu1 <- qlogis(mu1)
  
  # epsilon1.1 <- coef(glm(Y~-1 + offset(logit.mu1) + S1, 
  #                      family="binomial"))
  # epsilon1.2 <- coef(glm(Y~-1 + offset(logit.mu1) + S2, 
  #                        family="binomial"))
  # epsilon1.3 <- coef(glm(Y~-1 + offset(logit.mu1) + S3, 
  #                        family="binomial"))
  # epsilon1.4 <- coef(glm(Y~-1 + offset(logit.mu1) + S4, 
  #                        family="binomial"))
  
  epsilon.all <- coef(glm(Y~-1 + offset(logit.mu1) + 
                            S1+ S2+ S3+ S4, 
                          family="binomial"))
  
  # mu1.single.1 <- plogis(logit.mu1 + epsilon1.1*S1)
  # mu1.single.2 <- plogis(logit.mu1 + epsilon1.2*S2)
  # mu1.single.3 <- plogis(logit.mu1 + epsilon1.3*S3)
  # mu1.single.4 <- plogis(logit.mu1 + epsilon1.4*S4)
  
  mu1.single.1 <- plogis(logit.mu1 + epsilon.all[1]*S1)
  mu1.single.2 <- plogis(logit.mu1 + epsilon.all[2]*S2)
  mu1.single.3 <- plogis(logit.mu1 + epsilon.all[3]*S3)
  mu1.single.4 <- plogis(logit.mu1 + epsilon.all[4]*S4)
  
  E.A1.single <-  sum(A1*mu1.single.1,na.rm = TRUE)/sum(A1)
  E.A2.single<-  sum(A2*mu1.single.2,na.rm = TRUE)/sum(A2)
  E.A3.single <-  sum(A3*mu1.single.3,na.rm = TRUE)/sum(A3)
  E.A4.single <-  sum(A4*mu1.single.4,na.rm = TRUE)/sum(A4)
  

  return(c(TMLE.est,DR.est))

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
  
  muModel <- glm(Y~ D+Z,family = "binomial")
  mu <- predict(muModel,data = cbind(D,Z),type="response",se.fit = TRUE)
  se.mu <- mean(mu$se.fit)
 
  mat1 <- data.frame(cbind(D,Z))
  mat1[,1] <- rep(1,nrow(mat1))
  mu1 <- predict(muModel,data = mat1,type="response",se.fit = TRUE)
  se.mu1 <- mean(mu1$se.fit)
  mu1 <- mu1$fit
  
  P1 <- sum(A1)/length(A1)
  P2 <- sum(A2)/length(A2)
  P3<- sum(A3)/length(A3)
  P4 <- sum(A4)/length(A4)
  
  
  E.A1.GLM <- sum(A1*mu1,na.rm = TRUE)/sum(A1)
  E.A2.GLM <- sum(A2*mu1,na.rm = TRUE)/sum(A2)
  E.A3.GLM<- sum(A3*mu1,na.rm = TRUE)/sum(A3)
  E.A4.GLM <- sum(A4*mu1,na.rm = TRUE)/sum(A4)
  
  Sigma <- matrix(0,4,4)
  diag(Sigma) <- 1
  Z.DR <- mvrnorm( n, mu = rep(0,4), Sigma = Sigma )
  kappa.DR <- quantile(apply(Z.DR,1, max),0.975)

  
  LB.glm <- c(E.A1.GLM,E.A2.GLM,E.A3.GLM,E.A4.GLM) -  kappa.DR*se.mu1/sqrt(n)
  cover <- NULL
  for(b in 1:4){
    cover[b] <- (LB.glm[b] < truePsi[b])
  }
  
  #return(c(E.A1.GLM,E.A2.GLM,E.A3.GLM,E.A4.GLM))
  return(cover)
  
}


Niter <- 200
size <- c(1500,2000,2500,3000)

result <- matrix(NA,ncol = 4*2,nrow = length(size))
SDresult <- matrix(NA,ncol = 4*2,nrow = length(size))
length.res <- matrix(NA,ncol = 4*2,nrow = length(size))


for (j in 1: length(size)){
  print(size[j])
  res <- matrix(NA,ncol = 4*1,nrow = Niter)
  
  for (i in 1:Niter){
    #res[i,] <- EstimatorGLM(size[j])
    res[i,] <- Estimator(size[j])
  }
  
  result[j,] <- colMeans(res,na.rm = TRUE)
  SDresult[j,] <- apply(res,2, function(x) sd(x,na.rm = TRUE))
  
}


bias <- matrix(NA,nrow = length(size), ncol = 2)
sd <- matrix(NA,nrow = length(size), ncol = 2)
coverage <- matrix(NA,nrow = length(size), ncol = 2)

for (i in 1:length(size)){
  bias[i,] <- c(sum(abs(result[i,1:4] - truePsi)),
               sum(abs(result[i,5:8] - truePsi)))

  sd[i,] <- c(sum(abs(SDresult[i,1:4])),
              sum(abs(SDresult[i,5:8])))
  
  coverage[i,] <- c(mean(result[i,1:4]),
                    mean(result[i,5:8]))

}




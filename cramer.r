library(cramer)

################################### Simulations ##############################################
##  step0: set dataset
r <- 1 # rotate over [1,4,8,12]
n.case <- 1 # rotate over [1,3]
n.sims <- 3 # number of simulations, 3 for demonstration

## step1: set hyperparameters
alpha <- 0.01 # set significance level = 0.01, can be changed
set.seed(42)

count <- 0
for(i in 1:n.sims){
    ## step2: load training and test data
    tr_file <- file.path(getwd(), paste0("data/simulation/sac",r,"/c",n.case,"_tr_", i, ".csv"))
    te_file <- file.path(getwd(), paste0("data/simulation/sac",r,"/c",n.case,"_te_", i, ".csv"))
    tr <- read.csv(file=tr_file,header=TRUE,sep=",",dec=".", stringsAsFactors=FALSE, fill=TRUE)
    te <- read.csv(file=te_file,header=TRUE,sep=",",dec=".", stringsAsFactors=FALSE, fill=TRUE)
    
    ## step3: extract features
    x_tr <- cbind(tr['X1'],tr['X2'])
    x_tr_mat <- as.matrix(x_tr)  
    x_te <- cbind(te['X1'],te['X2'])
    x_te_mat <- as.matrix(x_te) 
    
    ##step4: perform cramer test
    ct <- cramer.test(x_tr_mat,x_te_mat,sim="permutation")
    
    ## step5: count the number of rejecting null hypothesis
    if (ct[["p.value"]] < alpha){
        count <- count + 1
    }
}



#################################### Real datasets ##############################################
# Take `house_latitude` dataset for example. 

## step1: load training and test data
tr <- read.csv(file="./data/housing/house_latitude_tr.csv",header=TRUE,sep=",",dec=".", stringsAsFactors=FALSE, fill=TRUE)
te <- read.csv(file="./data/housing/house_latitude_te.csv",header=TRUE,sep=",",dec=".", stringsAsFactors=FALSE, fill=TRUE)

## step2: extract features
x_tr <- cbind(tr['housing_median_age'],tr['median_income'],tr['total_rooms'],tr['total_bedrooms'],tr['population'],tr['households'])
x_tr_mat <- as.matrix(x_tr)  
x_te <- cbind(te['housing_median_age'],te['median_income'],te['total_rooms'],te['total_bedrooms'],te['population'],te['households'])
x_te_mat <- as.matrix(x_te) 

##step3: perform cramer test
cramer.test(x_tr_mat,x_te_mat,sim="permutation")
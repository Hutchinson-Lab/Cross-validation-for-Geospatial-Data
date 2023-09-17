library(sp)
library(raster)
library(blockCV) #version 2.1.4

################################### Simulations ##############################################
##  step0: set dataset
dir.create("bcv")
n.sims <- 3 # number of simulations, 3 for demonstration, 100 in paper
r = 1 # rotate over [1,4,8,12]
c = 1 # rotate over [1,3]
dir.create(paste0("bcv/sac",r)) 

## step1: set hyperparameters
bs <- 20 # blocksize
k <- 9 # number of folds

for(i in 1:n.sims){
    ## step2: load training data
    sample_file <- file.path(getwd(), paste0("/data/sac",r,"/c",c,"_tr_", i, ".csv")) 
    sample_data <- read.csv(file=sample_file,header=TRUE,sep=",",dec=".", stringsAsFactors=FALSE, fill=TRUE)
    
    ## step3: set coords
    coords = sample_data[c("long", "lat")]
    
    ## step4: split training set by spatial blocking
    x_coords <- SpatialPointsDataFrame(coords = coords, data = sample_data[c("X1")])
    bcv_k <- spatialBlock(speciesData = x_coords, theRange = bs*111325, k=k, selection = "systematic", iteration = 10, xOffset = 0, yOffset = 0) 
    
    ## step5: output blocking fold id
    myfile <- file.path(getwd(), paste0("/bcv/sac",r,"/bcv_c",c,"_",i, ".csv"))
    write.csv(bcv_k[["foldID"]], file = myfile)
}



#################################### Real datasets ##############################################
# Take `house_latitude` dataset for example. 

## step1: set hyperparameters
bs <- 0.23 # blocksize 
k <- 10 # number of folds

## step2: load training data
file <- "./data/housing/house_latitude_tr.csv" #change the training set
data <- read.csv(file=file,header=TRUE, sep=",",dec=".", stringsAsFactors=FALSE, fill=TRUE)

## step3: set coords
xy = data[c("Longitude", "Latitude")]

## step4: split training set by spatial blocking
count <- SpatialPointsDataFrame(coords = xy, data = data[c("median_house_value")],proj4string=CRS("+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs"))
bcv <- spatialBlock(speciesData = count, theRange = bs*111325, k=k, selection = "systematic", iteration = 100, xOffset = 0, yOffset = 0) 

## step5: output blocking fold id
myfile <- file.path(getwd(), "house_latitude_b0.23.csv")  # change the output filename
write.csv(bcv[["foldID"]], file = myfile)

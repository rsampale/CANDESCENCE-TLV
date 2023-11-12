# before running, do use_virtualenv(virtualenv="r-reticulate",required=TRUE)
# before running, do use_virtualenv(virtualenv="/home/rsampale/.local/share/r-miniconda/envs/tf_new",required=TRUE)


if (tensorflow::tf$executing_eagerly())
  tensorflow::tf$compat$v1$disable_eager_execution()

setwd("/home/rsampale/repo/candescence-grace/src/tlv/1-vae-creation")
source("init.R")

# install.packages("bioimagetools")
require(tcltk)
library(keras)
library(bmp)
library(imager)
library(tidyverse)
library(abind)
library(ggpubr)

K <- keras::backend()

exp = "simple-vae-2"
numba = "simple-vae-2"
thresh = 0.25 # legacy feature, useless
# DATASET SEED (for creation of datasets)
dataset_seed = 5004
wash_included = FALSE

# Paths ----------------------

# MAKE TRAIN / VAL / TEST SETS

all_images_path <- "/home/data/refined/candescence/tlv/0.2-images_cut/all-final" #can make simpler

train_num <- 1600
val_num <- 400
test_num <- 400

set.seed(dataset_seed)
# comment out when not in use:
# create_samples(all_images_path, VAES, train_num, val_num, test_num, dataset_seed, incl_wash = wash_included, thresh, pattern = "bmp$|BMP$")

train_events <- file.path(VAES, paste0("train",  dataset_seed))
val_events <- file.path(VAES, paste0("val",  dataset_seed))
test_events <- file.path(VAES, paste0("test",  dataset_seed, "_", thresh))


keras_model_dir <-  file.path(VAES, "keras_models", exp)
test_plots_dir <- file.path(VAES, "test_plots")

output <- file.path(VAES, exp)

# input image dimensions
img_rows <- 135L
img_cols <- 135L
# color channels (1 = grayscale, 3 = RGB)
img_chns <- 1L

# Data preparation --------------------------------------------------------


image_tensor <- function( targets ) {
  tmp <-  lapply(targets, FUN = function(t) {
    tmp <- load.image(t)
    
    # DO CHECK AND DELETE COLUMN OF ROW IF NECESARY
    rows <- dim(tmp)[1]
    cols <- dim(tmp)[2]
    
    if (rows != 135 && cols != 135) {
      tmp <- imsub(tmp,x<136,y<136)
    }
    else if (cols != 135 && rows == 135) {
      tmp <- imsub(tmp,y<136)
    }
    else if (rows != 135 && cols == 135) {
      tmp <- imsub(tmp,x<136)
    }
    
    ## Make grayscale here ##
    tmp <- grayscale(tmp)
    
    ## NOISE REMOVAL ##
    # tmp<- imsharpen(tmp,amplitude=1,type="shock",edge=0.2)
    # tmp <- threshold(tmp,thr = "92%") 
    # tmp <- vanvliet(tmp,sigma=2,order=1) # gaussian filter, can capture edges pretty well but doesn't address background color issue
    
    return(tmp)
  }) 
  return(tmp)
}


files_x_train <- list.files(train_events, full.names = TRUE)

files_x_val <- list.files(val_events, full.names = TRUE)

files_x_test <- list.files(test_events, full.names = TRUE)

x_train <- image_tensor( files_x_train )
x_val  <- image_tensor( files_x_val )
x_test <- image_tensor( files_x_test )

dim(x_test)
# Step 1: Create a new list with 135 x 135 matrices
matrices_list <- lapply(x_train, function(cimg) {
  matrix_data <- cimg[, , 1, 1]  # Extract the 135 x 135 data from cimg
  return(matrix_data)
})
# Step 2: Convert the list of matrices into a 3D array
x_train <- simplify2array(matrices_list)
x_train <- aperm(x_train, c(3,1,2))
matrices_list <- lapply(x_val, function(cimg) {
  matrix_data <- cimg[, , 1, 1]  # Extract the 135 x 135 data from cimg
  return(matrix_data)
})
x_val <- simplify2array(matrices_list)
x_val <- aperm(x_val, c(3,1,2))
matrices_list <- lapply(x_test, function(cimg) {
  matrix_data <- cimg[, , 1, 1]  # Extract the 135 x 135 data from cimg
  return(matrix_data)
})
x_test <- simplify2array(matrices_list)
x_test <- aperm(x_test, c(3,1,2))


# Reshape input data to 1d vector (prod(135*135))
input_size <- dim(x_train)[2]*dim(x_train)[3]

x_train <- array_reshape(x_train,c(nrow(x_train),input_size))

x_val <- array_reshape(x_val,c(nrow(x_val),input_size))

x_test <- array_reshape(x_test,c(nrow(x_test),input_size))

#### Parameterization ####

set.seed(1)
visualization <- TRUE

# number of convolutional filters to use
filters <- 68L

# convolution kernel size
num_conv <- 5L

latent_dim <- 4L
intermediate_dim <-  34L
epsilon_std <- 1.0

# training parameters
batch_size <- 64L
eps <- 20L

# learning rate of optimizer (default is 0.001 for adam)
LR <- 0.0005


#### Model Construction ####

enc_input <- layer_input(shape = c(input_size))
layer_one <- layer_dense(enc_input, units=256, activation = "relu")
z_mean <- layer_dense(layer_one, latent_dim)
z_log_var <- layer_dense(layer_one, latent_dim)

encoder <- keras_model(enc_input, z_mean)
summary(encoder)

sampling <- function(arg){
  z_mean <- arg[, 1:(latent_dim)]
  z_log_var <- arg[, (latent_dim + 1):(2 * latent_dim)]
  epsilon <- k_random_normal(shape = c(k_shape(z_mean)[[1]]), mean=0)
  z_mean + k_exp(z_log_var/2)*epsilon
}

z <- layer_concatenate(list(z_mean, z_log_var)) %>% 
  layer_lambda(sampling)

decoder_layer <- layer_dense(units = 256, activation = "relu")
decoder_mean <- layer_dense(units = input_size, activation = "sigmoid")
h_decoded <- decoder_layer(z)
x_decoded_mean <- decoder_mean(h_decoded)

vae <- keras_model(enc_input, x_decoded_mean)

# custom loss function
vae_loss <- function(input, x_decoded_mean){
  xent_loss=(input_size/1.0)*loss_binary_crossentropy(input, x_decoded_mean)
  kl_loss=-0.5*k_mean(1+z_log_var-k_square(z_mean)-k_exp(z_log_var), axis=-1)
  xent_loss + kl_loss
}

#vae %>% compile(optimizer = "rmsprop", loss = vae_loss)
vae %>% compile(optimizer = optimizer_adam(learning_rate = LR), loss = vae_loss)  # better than rmsprop
#vae %>% compile(optimizer = optimizer_nadam(), loss = vae_loss)  # not bad. a bit compressed. compare with adam
#vae %>% compile(optimizer = optimizer_adagrad(), loss = vae_loss)  # also in contention. A bit compressed
#vae %>% compile(optimizer = optimizer_adadelta(), loss = vae_loss) # also good
summary(vae)

optimiza <- "adam" # could change this as a parameter

## build a digit generator that can sample from the learned distribution

dec_input <- layer_input(shape = latent_dim)
h_decoded_2 <- decoder_layer(dec_input)
x_decoded_mean_2 <- decoder_mean(h_decoded_2)
generator <- keras_model(dec_input, x_decoded_mean_2)
summary(generator)

#### Model Fitting ####


with(tensorflow::tf$device('GPU:8'), {
  vae %>% fit(
    x_train, x_train, 
    shuffle = TRUE,
    epochs = eps,
    batch_size = batch_size, 
    validation_data = list(x_val, x_val)
  )
})

gc()

#vae %>% model.save((file.path(save_keras_dir, "first_time"))
#vae %>% export_savedmodel(file.path(save_keras_dir, "first_time"), remove_learning_phase = FALSE)
# vae %>% save_model_weights_tf(keras_model_dir)



#### Visualizations ####

library(ggplot2)
library(grid)
library(bmp)
library(dplyr)
library(pals)
library(graphics)
library(ggimage)

# 
# 
# # -> prep the test data frames
# 
# x_train_encoded <- predict(encoder, x_train, batch_size = batch_size)
# x_val_encoded <- predict(encoder, x_val, batch_size = batch_size)
x_test_encoded <- predict(encoder, x_test, batch_size = batch_size)

x_test_encoded <- x_test_encoded %>% as_data_frame()



### CREATE SCATTERPLOT WITH ORIGINAL IMAGES AS POINTS ON LATENT SPACE ###
img_populated_df <- data.frame(x = x_test_encoded$V1,
                               y = x_test_encoded$V2,
                               image = files_x_test
)
# plot img scatter plot
p <- ggplot(img_populated_df, aes(x, y)) + geom_image(aes(image=image), size=.04)
print(p)

# Save the plot to the vae directory for that specific test
plot_filename <- paste0(exp,".png")
ggsave(filename=plot_filename,plot=p,path=test_plots_dir)

# Save the tibble (with all vectors) to refined, for later use in analysis
all_umaps_w_imgs <- cbind(x_test_encoded,files_x_test) 

all_umaps_w_imgs <- as_tibble(all_umaps_w_imgs)
save(all_umaps_w_imgs,file = file.path(TLV,"1.1-umap_tibbles",paste0(exp,"-umap_img_tibble.Rdata")))

# Clear all labels from the plot (COMMENT OUT IF NEEDED)
# p <- p + annotate("text", label = "", x = x_test_encoded$V1, y = x_test_encoded$V2)



### IMAGE RECONSTRUCTION VISUALIZATION (move to different file later) ###
if (visualization) {
  
  ### IMAGE RECONSTRUCTION COMPARISON ###
  n = 5
  test = x_test[0:n,]
  encoded_stuff <- predict(encoder, test)
  
  decoded_imgs = generator %>% predict(encoded_stuff)
  pred_images = array_reshape(decoded_imgs, dim=c(dim(decoded_imgs)[1], 135, 135))
  orig_images = array_reshape(test, dim=c(dim(test)[1], 135, 135))
  
  op = par(mfrow=c(n,2), mar=c(1,0,0,0))
  for (i in 1:n) 
  {
    plot(as.raster(orig_images[i,,])) # using cimg instead of raster because some values are negative
    plot(as.raster(pred_images[i,,]))
  }
  ### IMAGE INTERPOLATION ###
  
}


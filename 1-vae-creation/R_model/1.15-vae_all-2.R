# before running, do use_virtualenv(virtualenv="r-reticulate",required=TRUE)
# before running, do library(reticulate) then do use_condaenv("tf_new", required = TRUE)

library(reticulate)
use_condaenv("tf")

if (tensorflow::tf$executing_eagerly())
  tensorflow::tf$compat$v1$disable_eager_execution()


setwd("/home/rsampale/repo/candescence_master/projects/tlv/1-vae-creation")
source("init.R")

# install.packages("bioimagetools")
require(tcltk)
library(tensorflow)
library(keras)
library(bmp)
library(imager)
library(tidyverse)
library(abind)
library(ggpubr)

K <- keras::backend()

Sys.getenv(x="LD_LIBRARY_PATH")

exp = "discard7"
numba = "discard7"
thresh = 0.25 # legacy feature, useless
# DATASET SEED (for creation of datasets)
dataset_seed = 5005
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
    tmp <- vanvliet(tmp,sigma=2,order=1) # gaussian filter, can capture edges pretty well but doesn't address background color issue
    
    tmp <- as.array(tmp)
    return( array_reshape(tmp[,,1,1], c(img_rows, img_cols, img_chns), order = "F" ) )
  }) 
  return( abind( tmp, along = 0 ))
}


files_x_train <- list.files(train_events, full.names = TRUE)

files_x_val <- list.files(val_events, full.names = TRUE)

files_x_test <- list.files(test_events, full.names = TRUE)



x_train <- image_tensor( files_x_train )
x_val  <- image_tensor( files_x_val )
x_test <- image_tensor( files_x_test )

#cat("\n Dimensions of train: ", dim(x_train), 
#    "\t Dimensions of val: ", dim(x_val), 
#    "\t Dimensions of test: ", dim(x_test), "\n")

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
batch_size <- 75L
eps <- 15L

# learning rate of optimizer (default is 0.001 for adam)
LR <- 0.0003


#### Model Construction ####

original_img_size <- c(img_rows, img_cols, img_chns)

x <- layer_input(shape = c(original_img_size))

conv_1 <- layer_conv_2d(  #conv2d_20
  x,
  filters = img_chns, # number of output filters (dimensions) in the convolution
  kernel_size = c(2L, 2L), # height and width of the convolution window
  strides = c(1L, 1L), # strides of the convolution along the height and width
  padding = "same", # same means padding with zeroes evenly around the input (since strides = 1 output same size as input)
  activation = "relu" # which activation function to use
)

conv_2 <- layer_conv_2d(  #convd_21
  conv_1,
  filters = filters,
  kernel_size = c(2L, 2L),
  strides = c(2L, 2L),
  padding = "same",
  activation = "relu"
)

conv_3 <- layer_conv_2d( #convd_22
  conv_2,
  filters = filters,
  kernel_size = c(num_conv, num_conv),
  strides = c(1L, 1L),
  padding = "same",
  activation = "relu"
)

conv_4 <- layer_conv_2d( #convd_23
  conv_3,
  filters = filters,
  kernel_size = c(num_conv, num_conv),
  strides = c(1L, 1L),
  padding = "same",
  activation = "relu"
)

flat <- layer_flatten(conv_4)
hidden <- layer_dense(flat, units = intermediate_dim, activation = "relu")

z_mean <- layer_dense(hidden, units = latent_dim)
z_log_var <- layer_dense(hidden, units = latent_dim)

sampling <- function(args) {
  z_mean <- args[, 1:(latent_dim)]
  z_log_var <- args[, (latent_dim + 1):(2 * latent_dim)]
  
  epsilon <- k_random_normal(
    shape = c(k_shape(z_mean)[[1]]),
    mean = 0.,
    stddev = epsilon_std
  )
  z_mean + k_exp(z_log_var) * epsilon
}

z <- layer_concatenate(list(z_mean, z_log_var)) %>% layer_lambda(sampling)

#output_shape <- c(batch_size, 14L, 14L, filters)
# output_shape <- c(batch_size, 64L, 64L, filters) 
#output_shape <- c(batch_size, 128L, 128L, filters)
output_shape <- c(batch_size, 68L, 68L, filters)


decoder_hidden <- layer_dense(units = intermediate_dim, activation = "relu")
decoder_upsample <- layer_dense(units = prod(output_shape[-1]), activation = "relu")

decoder_reshape <- layer_reshape(target_shape = output_shape[-1])
decoder_deconv_1 <- layer_conv_2d_transpose(
  filters = filters,
  kernel_size = c(num_conv, num_conv),
  strides = c(1L, 1L),
  padding = "same",
  activation = "relu"
)

decoder_deconv_2 <- layer_conv_2d_transpose(
  filters = filters,
  kernel_size = c(num_conv, num_conv),
  strides = c(1L, 1L),
  padding = "same",
  activation = "relu"
)

decoder_deconv_3_upsample <- layer_conv_2d_transpose(
  filters = filters,
  kernel_size = c(3L, 3L),
  strides = c(2L, 2L),
  padding = "valid",
  activation = "relu"
)

decoder_mean_squash <- layer_conv_2d( 
  filters = img_chns,
  kernel_size = c(2L, 2L),
  strides = c(1L, 1L),
  padding = "valid",
  activation = "sigmoid"
)

decoder_mean_squash_crp <- layer_cropping_2d( #added cropping layer to reduce dimensions by 1 pix on height and width
  cropping = list(c(1L,0L),c(1L,0L)),
  data_format= NULL
)

hidden_decoded <- decoder_hidden(z)
up_decoded <- decoder_upsample(hidden_decoded)
reshape_decoded <- decoder_reshape(up_decoded)
deconv_1_decoded <- decoder_deconv_1(reshape_decoded)
deconv_2_decoded <- decoder_deconv_2(deconv_1_decoded)
x_decoded_relu <- decoder_deconv_3_upsample(deconv_2_decoded)
x_decoded_mean_squash <- decoder_mean_squash(x_decoded_relu)
x_decoded_mean_squash_crp <- decoder_mean_squash_crp(x_decoded_mean_squash) #added

# custom loss function
vae_loss3 <- function(x, x_decoded_mean_squash_crp) { # works with ranges (6 to -4)
  beta = 1.0
  x <- k_flatten(x)
  x_decoded_mean_squash_crp <- k_flatten(x_decoded_mean_squash_crp)
  xent_loss <- loss_binary_crossentropy(x, x_decoded_mean_squash_crp) 
  kl_loss <- -0.5 * k_sum(1 + z_log_var - k_square(z_mean) - k_square(k_exp(z_log_var)), axis = -1)
  
  k_mean(xent_loss + kl_loss)
}

## variational autoencoder
vae <- keras_model(x, x_decoded_mean_squash_crp) #changed
#vae %>% compile(optimizer = "rmsprop", loss = vae_loss)
vae %>% compile(optimizer = keras$optimizers$legacy$Adam(learning_rate = LR), loss = vae_loss3)  # better than rmsprop
#vae %>% compile(optimizer = optimizer_nadam(), loss = vae_loss)  # not bad. a bit compressed. compare with adam
#vae %>% compile(optimizer = optimizer_adagrad(), loss = vae_loss)  # also in contention. A bit compressed
#vae %>% compile(optimizer = optimizer_adadelta(), loss = vae_loss) # also good


optimiza <- "adam" # could change this as a parameter

summary(vae)


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

## encoder: model to project inputs on the latent space
encoder <- keras_model(x, z_mean)

## generator: reconstruct image from latent representation
decoder_input <- layer_input(shape = latent_dim)
d_hidden_decoded <- decoder_hidden(decoder_input)
d_up_decoded <- decoder_upsample(d_hidden_decoded)
d_reshape_decoded <- decoder_reshape(d_up_decoded)
d_deconv_1_decoded <- decoder_deconv_1(d_reshape_decoded)
d_deconv_2_decoded <- decoder_deconv_2(d_deconv_1_decoded)
d_x_decoded_relu <- decoder_deconv_3_upsample(d_deconv_2_decoded)
d_x_decoded_mean_squash <- decoder_mean_squash(d_x_decoded_relu)
d_x_decoded_mean_squash_crp <- decoder_mean_squash_crp(d_x_decoded_mean_squash)
generator <- keras_model(decoder_input, d_x_decoded_mean_squash_crp)


gc()

#vae %>% model.save((file.path(save_keras_dir, "first_time"))
#vae %>% export_savedmodel(file.path(save_keras_dir, "first_time"), remove_learning_phase = FALSE)
vae %>% save_model_weights_tf(keras_model_dir)



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
  test = x_test[0:n,,,]
  test = array_reshape(test, c(nrow(test),img_rows,img_cols,img_chns)) #adding the img_chns dimension - kind of useless but encoder input needs it..
  encoded_stuff <- predict(encoder, test) # if using encoder, get different coordinates - z_encoder gives same coordinates for all images
  
  decoded_imgs = generator %>% predict(encoded_stuff)
  
  op = par(mfrow=c(n,2), mar=c(1,0,0,0))
  for (i in 1:n) 
  {
    plot(as.cimg(test[i,,,])) # using cimg instead of raster because some values are negative
    plot(as.cimg(decoded_imgs[i,,,]))
  }
  
  
  ### IMAGE INTERPOLATION ###
  
}


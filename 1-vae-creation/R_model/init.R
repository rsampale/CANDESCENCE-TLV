# BiocManager::install("ComplexHeatmap")
# library(ComplexHeatmap)
library(circlize)
library(reticulate); library(tidyverse); library(ggpubr)
library(cvms)
library(broom)    # tidy()
library(rlist); library(feather); library(imager)
library("readxl")

set.seed(1)
CANDESCENCE="/home/data/refined/candescence"
GRACE=file.path(CANDESCENCE, "grace")
TLV=file.path(CANDESCENCE, "tlv")
VAES=file.path(TLV, "vaes")

current_grace_annotation_file <- "grace_library_annotations_july15"
grace <- readRDS(file.path(CANDESCENCE, "grace", paste0(current_grace_annotation_file, ".rds")))


short_filename <- function( fnames ) {
  f <- str_split( fnames, pattern = "/" )
  return( unlist( lapply( f , "[[", length(f[[1]]) )) )  }

scale_bboxes <- function( bboxes, 
                          orig_height =  1040, orig_width = 1408,
                          target_height=800, target_width =800) {
  
  b <- apply(bboxes, MARGIN=1, FUN=function(x) 
    c( 
      max(1, floor(x[1]/orig_width *target_width)),
      max(1, floor(x[2]/orig_height*target_height)), 
      floor(x[3]/orig_width *target_width),
      floor(x[4]/orig_height*target_height)))
  return(t(b))
}


convert_to_pickle_format <- function(res, code) {
  output <- list()
  for (i in 1:length(res)) {
    if (res[[i]]$Skipped) next
    nd <- length(output) + 1
    current <- res[[i]]
    
    tmp <- list()
    tmp$filename <- current$`External ID`
    tmp$width <- 1408
    tmp$height <- 1040
    
    obj <- current[4][[1]][[1]]
    
    labels <- c()
    bboxes <- matrix(0, nrow = length(obj), ncol = 4)
    for (j in 1:length(obj)) {
      kurrent <- obj[[j]]
      labels[j] <- as.numeric(code[kurrent$value])
      bboxes[j, 1] <- as.numeric(kurrent$bbox$left)
      bboxes[j, 3] <- as.numeric(kurrent$bbox$left + kurrent$bbox$width)
      # bboxes[j, 2] <-
      #   as.numeric(tmp$height - (kurrent$bbox$top + kurrent$bbox$height))
      # bboxes[j, 4] <- as.numeric(tmp$height - kurrent$bbox$top)
      bboxes[j, 2] <- as.numeric(kurrent$bbox$top)
      bboxes[j, 4] <- as.numeric(kurrent$bbox$top + kurrent$bbox$height)
    }
    tmp$ann <- list()
    
    tmp$ann$bboxes <- scale_bboxes(bboxes)
    tmp$width <- 800
    tmp$height <- 800
    tmp$width <- as.integer(tmp$width)
    tmp$height <- as.integer(tmp$height)
    
    tmp$ann$labels <- as.array(labels)
    output[[nd]] <- tmp
  } # end of for i
  return(output)
}




convert_to_pickle_format_macro <- function(res, code) {
  output <- list()
  for (i in 1:length(res)) {
    nd <- length(output) + 1
    current <- res[[i]]
    
    tmp <- list()
    tmp$filename <- current$`External ID`
    tmp$width <- 1408
    tmp$height <- 1040
    
    obj <- current[4][[1]][[1]]
    
    labels <- c()
    bboxes <- matrix(0, nrow = length(obj), ncol = 4)
    for (j in 1:length(obj)) {
      kurrent <- obj[[j]]
      labels[j] <- as.numeric(code[kurrent$value])
      bboxes[j, 1] <- as.numeric(kurrent$bbox$left) - tmp$width
      bboxes[j, 3] <- as.numeric(kurrent$bbox$left + kurrent$bbox$width) - tmp$width
      # bboxes[j, 2] <-
      #   as.numeric(tmp$height - (kurrent$bbox$top + kurrent$bbox$height))
      # bboxes[j, 4] <- as.numeric(tmp$height - kurrent$bbox$top)
      bboxes[j, 2] <- as.numeric(kurrent$bbox$top)
      bboxes[j, 4] <- as.numeric(kurrent$bbox$top + kurrent$bbox$height)
    }
    tmp$ann <- list()
    
    tmp$ann$bboxes <- scale_bboxes(bboxes)
    tmp$width <- 800
    tmp$height <- 800
    tmp$width <- as.integer(tmp$width)
    tmp$height <- as.integer(tmp$height)
    
    tmp$ann$labels <- as.array(labels)
    output[[nd]] <- tmp
  } # end of for i
  return(output)
}


make_unique_by_iou <- function( hallucin, upper_bound ){
  
  all_files <- unique(hallucin[["short_filename"]])
  final <- hallucin[-c(1:nrow(hallucin)),]
  
  for (i in 1:length(all_files)) {
    current_file <- all_files[i]
    hall <- hallucin %>% filter( short_filename == current_file)
    if (nrow(hall) < 2) { 
      final <- bind_rows(final, hall)
      next
    }
    
    ious <- matrix( nrow=nrow(hall), ncol = nrow(hall), data = 0)
    
    for (j in 1:(nrow(hall)-1)) {
      for (k in (j+1):nrow(hall)) {
        
        A <- c( hall[["bbox_1"]][j], hall[["bbox_2"]][j], hall[["bbox_3"]][j], hall[["bbox_4"]][j] )
        B <- c( hall[["bbox_1"]][k], hall[["bbox_2"]][k], hall[["bbox_3"]][k], hall[["bbox_4"]][k] )
        
        # x-dimension   
        xl <- max( A[1], B[1] )
        xr <- min( A[3], B[3] )
        if (xr <= xl) next
        
        yh <-min( A[2], B[2])
        yl <- max( A[4], B[4])
        if (yh >= yl) next
        
        num <- (xr - xl) * (yl - yh)
        denom <- num + ( (A[3]- A[1]) * (A[4]-A[2]) ) + ( (B[3]-B[1]) * (B[4]-B[4]) )
        
        ious[j, k] <-   num / denom
      } # end of k
    } # end of j
    
    to_remove <- c()
    while (max(ious) > upper_bound) {
      loc <- which(ious == max(ious), arr.ind = TRUE)
      to_remove <- c(to_remove, loc[1])
      ious[loc[1], ] <- 0
      ious[ , loc[1]] <- 0
    }
    
    if (length(to_remove) > 0) print( hall[to_remove, ] )
    
    ifelse(length(to_remove) > 0, final <- bind_rows(final, hall[-to_remove,]), final <- bind_rows(final, hall))
    
  }
  return(final)
}

reformat_grace_annotations <- function( ) {
  
  raw <-  read_excel(file.path(CANDESCENCE, "grace", paste0(current_grace_annotation_file, ".xlsx") ))
  raw <- raw %>%
      relocate( `Plate`, Position, `orf19 name`, Common, `Feature Name`, `Description`, 
                `Replicate 1 Macrophages`, `Replicate 1 TC conditions`, 
                `Replicate 2 Macrophages`, `Replicate 2 TC conditions`,
                `S.cerevisiae homologue`, `S. cerevisiae KO phenotype` )
  raw <- raw %>%
      rename( plate=`Plate`, position=Position, orf=`orf19 name`, common=Common, feature_name=`Feature Name`, 
              description=`Description`, 
              rep1_macro=`Replicate 1 Macrophages`, rep1_TC=`Replicate 1 TC conditions`, 
              rep2_macro=`Replicate 2 Macrophages`, rep2_TC=`Replicate 2 TC conditions`,
              sc_homologue=`S.cerevisiae homologue`, sc_ko_pheno=`S. cerevisiae KO phenotype` ) 
  
  raw$plate <- as.integer(str_split(raw$plate, pattern="Plate ", simplify=TRUE)[,2])
  raw <- raw %>% separate( col=position, into=c("row", "column"), sep=1 )
  raw$column <- as.character(as.integer(raw$column))
  
  grace <- raw
  saveRDS(grace, file.path(CANDESCENCE, "grace", paste0(current_grace_annotation_file, ".rds")))
}

calculate_area_via_diagnonal <- function( t,l, b, r, width=10 ) {
  adj <- sqrt( width^2 / 2 )  
  if ((b-t) < width) 
    if ((r-l) < width) return( (b-t)*(r-l) )  else return( width * (r-l) )
        
  if ((r-l) < width) return( width * (b-t) )
  
  tri <- ((b-t-adj) * (r-l-adj))/2
  return( ((b-t) * (r-l)) - 2*(tri) )
}

create_samples <- function(all_imgs_path, dest, train_number, val_number, test_number, ds_seed, incl_wash, thresh_num, pattern = "bmp$|BMP$"){
  ####################################################################
  # all_imgs_path = path to folder with all images to select                                 
  # 
  # dest = path to destination folder
  #
  # X_number = percentage or number of images to select. If value is 
  #   between 0 and 1 percentage of files is assumed, if value greater than 1, 
  #   number of files is assumed                                               
  #                                                                            
  # pattern = file extension to select. By default it selects bmp files. For   
  #   other type of files replace bmp and BMP by the desired extension         
  ####################################################################
  
  # Get file list with full path and file names
  files <- list.files(all_imgs_path, full.names = TRUE, pattern = pattern)
  
  file_names <- list.files(all_imgs_path, pattern = pattern)
  
  # Remove wash images if argument is set
  if (!incl_wash) {
    # Find indices of files that do NOT contain "wash"
    non_wash_indices <- grep("wash", files, invert = TRUE, value = FALSE)
    non_wash_indices_names <- grep("wash", file_names, invert = TRUE, value = FALSE)
    
    # Filter files to exclude those with "wash" in their name
    files <- files[non_wash_indices]
    file_names <- file_names[non_wash_indices]
  }
  
  # Give error if numbers don't add up
  if (train_number <=1 | val_number <= 1 | test_number <= 1) {
    stopifnot(train_number + val_number + test_number == 1)
  } else {
    stopifnot(train_number + val_number + test_number <= length(files))
  }
  
  # Select the desired % or number of file by simple random sampling 
  randomize <- sample(seq(files))
  files2analyse <- files[randomize]
  names2analyse <- file_names[randomize]
  
  if(train_number <= 1){
    train_size <- floor(train_number * length(files))
    val_size <- floor(val_number * length(files))
    test_size <- floor(test_number * length(files))
  }else{
    train_size <- train_number
    val_size <- val_number
    test_size <- test_number
  }
  # commented = for DEBUG
  # print(paste(train_size,val_size,test_size,sep = " "))
  # print(files2analyse[(1:(train_size+val_size+test_size))])
  
  train_files2analyse <- files2analyse[(1:train_size)]
  train_names2analyse <- names2analyse[(1:train_size)]
  # print(train_files2analyse)
  val_files2analyse <- files2analyse[((train_size+1):(train_size+val_size))]
  val_names2analyse <- names2analyse[((train_size+1):(train_size+val_size))]
  # print(val_files2analyse)
  test_files2analyse <- files2analyse[((train_size+val_size+1):(train_size+val_size+test_size))]
  test_names2analyse <- names2analyse[((train_size+val_size+1):(train_size+val_size+test_size))]
  # print(test_files2analyse)
  
  # Create folder to output
  train_results_folder <- paste0(dest, '/train', ds_seed)
  dir.create(train_results_folder, recursive=FALSE)
  val_results_folder <- paste0(dest, '/val', ds_seed)
  dir.create(val_results_folder, recursive=FALSE)
  test_results_folder <- paste0(dest, '/test', ds_seed,"_",thresh_num)
  dir.create(test_results_folder, recursive=FALSE)
  
  # Write csv with file names
  write.table(train_names2analyse, file = paste0(dest, "/",ds_seed,"train_selected_files.csv"),
              col.names = "Files", row.names = FALSE)
  write.table(val_names2analyse, file = paste0(dest, "/",ds_seed,"val_selected_files.csv"),
              col.names = "Files", row.names = FALSE)
  write.table(test_names2analyse, file = paste0(dest, "/",ds_seed,"test_selected_files.csv"),
              col.names = "Files", row.names = FALSE)
  
  file.copy(train_files2analyse,train_results_folder)
  file.copy(val_files2analyse,val_results_folder)
  file.copy(test_files2analyse,test_results_folder)
}




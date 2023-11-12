library(imager)
library(magick)

# FUNCTION
create_save_subimages <- function(image_crop,name) {
    rows = 1:8
    cols = 1:12
    x_step <- (dim(image_crop)[1])/12
    y_step <- (dim(image_crop)[2])/8
    top_points_x = c(0)
    left_points_y =  c(0)
    
    for(i in cols) {
        new_point = top_points_x[i] + x_step
        top_points_x = c(top_points_x, new_point)
    }
    for(i in rows) {
        new_point = left_points_y[i] + y_step
        left_points_y = c(left_points_y, new_point)
    }
    
    # VISUALIZE GRID:
    # for (i in top_points_x) {
    #     vert_line <- (Xc(image_crop) %inr% c(i,i + 1)) & (Yc(image_crop) %inr% c(0,dim(image_crop)[2]))
    #     highlight(vert_line)
    # }
    # for (i in left_points_y) {
    #     vert_line <- (Xc(image_crop) %inr% c(0,dim(image_crop)[1])) & (Yc(image_crop) %inr% c(i,i+1))
    #     highlight(vert_line)
    # }
    
    ## TESTING: ITERATE THROUGH MY (small) LISTS TO SEE THE ELEMENTS
    # for(i in top_points_x) {
    #     print(i)
    # }
    # for(j in left_points_y) {
    #     print(j)
    # }
    ## END TESTING
    img_c = 0
    for (i in 1: 8) {
        for (j in 1: 12) {
            tempimg = imsub(image_crop,x %inr% c(top_points_x[j],top_points_x[j+1]),y %inr% c(left_points_y[i],left_points_y[i+1]))
            img_c = img_c + 1
            # bmp(filename = paste("/home/rsampale/repo/candescence-grace/src/tlv/0-image-prep/testing/",name,"-r",i,"-c",j,".bmp",sep = ""))
            # plot(tempimg,axes=FALSE)
            # dev.off()
            imager::save.image(im=tempimg,file=paste("/home/data/refined/candescence/tlv/0.2-images_cut/all-final/",name,"-r",i,"-c",j,".bmp",sep = ""))
        }
    }
}

image_files <- list.files(path = "/home/data/refined/candescence/tlv/0.1-images_whole/wash-final",pattern="*.bmp",full.names = TRUE,recursive = FALSE)

for (i in 1:length(image_files)) {
    img <- load.image(image_files[i])
    
    # COMMENTED = OPTIONAL VIEWING OF DIMENSIONS AND GRID:
    
    # #get dimensions of image
    # dim(img)
    # #plot the image to view it:
    # plot(img)
    # 
    # # CROP AND CHOP BIG (2560 x 1920) IMAGES
    # # visualize crop border:
    # crop_bounds <- (Xc(img) %inr% c(473,2092)) & (Yc(img) %inr% c(413,1493))
    # highlight(crop_bounds)
    # #crop
    img.crop <- imsub(img,x %inr% c(473,2092),y %inr% c(413,1493))
    # plot(img.crop)
    # # Get vertical and horizontal grid lines
    # dim(img.crop)
    image_name <- tools::file_path_sans_ext(basename(image_files[i]))
    
    create_save_subimages(img.crop,image_name)
}

# FOR COPYING:
# /home/data/refined/candescence/tlv/0.2-images_cut/day-5-batch1/
# /home/data/refined/candescence/tlv/0.1-images_whole/day-5-batch1








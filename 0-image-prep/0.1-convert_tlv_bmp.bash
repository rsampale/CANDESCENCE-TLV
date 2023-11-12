# run command from raw image directory with same directory name as target, e.g ~/data/raw/candescence/tlv_raw_images/day-X-batchX, and change
# final directory in path option of the command to desired directory

mogrify -path /home/data/refined/candescence/tlv/0.1-images_whole/wash-final -format bmp *.png

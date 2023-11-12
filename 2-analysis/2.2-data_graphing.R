source("/home/rsampale/repo/candescence-grace/src/tlv/1-vae-creation/init.R")
setwd("/home/rsampale/repo/candescence-grace/src/tlv/2-analysis")

# NAME OF FINAL EXPERIMENT / DATA USED IN THE ANALYSIS:
experiment <- "simple-vae-2"

# LOAD TIBBLES (from VAE run and from 2.1-read_tsv_data):
load(file.path(TLV,"1.1-umap_tibbles",paste0(experiment,"-umap_img_tibble.Rdata")))
load(file.path(TLV,"2.1-read_tsv_data","pre_umap_master.Rdata"))

# giving them better names for this stage
umaps_with_imgs <- all_umaps_w_imgs # for final_2 and later, use all_umaps_w_imgs, otherwise use img_populated_df
meta_table <- master_table

# For final_2+, do umapping to convert V1-Vx into x and y coordinates  *** POSSIBLY MOVE TO 1.1 END SO PLOTS ARE SAVED CORRECTLY ***
library(umap)
vectors_only <- select(umaps_with_imgs, matches("^V[1-9]|^V10$"))
rest <- select(umaps_with_imgs, -matches("^V[1-9]|^V10$"))
set.seed(777)

umapped_vecs <- umap(vectors_only, n_components = 2)
umap_coords <- as_tibble(umapped_vecs$layout) %>% rename(x = V1, y = V2)

umaps_with_imgs <- bind_cols(umap_coords,rest)
umaps_with_imgs <- rename(umaps_with_imgs, image = files_x_test)

#### Extend umaps_with_imgs to add new columns, for MEDIUM, REPLICATE, DAY ####


library(tidyr)
library(dplyr)
library(stringr)

pre_master_separated <- umaps_with_imgs
pre_master_separated <- pre_master_separated %>%
  mutate(file_name = image) 

pre_master_separated <- separate(pre_master_separated, file_name, into = c("path", "plate", "growth_medium", "day", "rep_position"), sep = "_")
# fix plate and rep-pos columns
pre_master_separated <- pre_master_separated %>%
  mutate(plate = str_extract(plate, "(?<=P[l]?)\\d+"))
pre_master_separated <- pre_master_separated %>% mutate(plate = as.numeric(plate)) # turn character plate numbers into actual doubles
pre_master_separated <- pre_master_separated %>%
  separate(rep_position, into = c("replicate", "position"), sep = "-",extra="merge")

# Change Pwt into plate "-1"
# Change spdr medium into "spider"
# Change ctrl medium into "control"
pre_master_separated <- pre_master_separated %>%
  mutate(plate = ifelse(is.na(plate), -1, plate),
         growth_medium = ifelse(growth_medium == "spdr", "spider",
                                ifelse(growth_medium == "ctrl", "control", growth_medium)))

# turn the "rx-cy" position format into the one consistent with METADATA:
get_row_col <- function(pos) {
  # extract the "rX-cY" part of the file name using regular expressions
  rc_str <- sub("r([0-9]+)-c([0-9]+)\\.bmp", "\\1-\\2", pos)
  # split the string into separate row and column components
  rc <- strsplit(rc_str, "-")[[1]]
  # convert the row and column values to numbers
  row <- LETTERS[as.numeric(rc[1])]
  col <- as.numeric(rc[2])
  # return the row and column as a string
  paste0(row, col)
}
pre_master_separated$position <- sapply(pre_master_separated$position, get_row_col)
# view the resulting tibble
pre_master_separated



#### Merge tibbles into one master tibble ####

MASTER <- inner_join(pre_master_separated, meta_table, by = c("plate","position"))
MASTER <- select(MASTER, -path) # remove useless "path" column
# NOTE: NOT EVERY PLATE AND POSITION WAS DOCUMENTED ON THE METADATA TABLE
#       "Pwt"/-1 plates are also not documented, and are thus not added to the MASTER        


#### Generate GRAPHS from the data ####
library(ggplot2)
library(grid)
library(bmp)
library(pals)
library(graphics)
library(ggimage)
library(ggforce)
require(car)

# Scatter with image overlay
image_overlay <- ggplot(MASTER, aes(x, y)) + geom_image(aes(image=image), size=.04)
print(image_overlay)

# Scatter with coloring based on MEDIUM
distinct_colors <- c("control" = "#E41A1C", "RPMI" = "#377EB8", "serum" = "#4DAF4A", "spider" = "#984EA3", "YPD" = "#FF7F00")
medium <- ggplot(MASTER, aes(x, y, color = growth_medium)) + geom_point(size=2) + scale_color_manual(values=distinct_colors) + theme(
  legend.text = element_text(size = 14),          # Bigger legend text
  legend.title = element_text(size = 16),         # Bigger legend title
  legend.key.size = unit(1.5, "cm")                # Bigger legend key
) + guides(color = guide_legend(override.aes = list(size = 4)))
print(medium)
# Scatter with coloring based on DAY
day <- ggplot(MASTER, aes(x, y, color = day)) + geom_point(size = 2) + theme(
  legend.text = element_text(size = 14),          # Bigger legend text
  legend.title = element_text(size = 16),         # Bigger legend title
  legend.key.size = unit(1.5, "cm")                # Bigger legend key
) + guides(color = guide_legend(override.aes = list(size = 4)))
print(day)
# Scatter with coloring based on ISOLATE TYPE
filtered_iso_MASTER <- MASTER %>%
  filter(!is.na(isolate_type) & isolate_type != "unknown")
iso_type <- ggplot(filtered_iso_MASTER, aes(x, y, color = isolate_type)) + geom_point()
print(iso_type)
# Scatter with coloring based on GEOGRAPHY (COUNTRY)
geo_country <- ggplot(MASTER, aes(x, y, color = geography_country)) + geom_point()
print(geo_country)
# Scatter with coloring based on GEOGRAPHY (GENERAL)
geo_general <- ggplot(MASTER, aes(x, y, color = geography_general)) + geom_point()
print(geo_general)


# Testing circles: *** EXPERIMENTAL AND DOESN'T WORK WELL - CIRCLES TOO LARGE ***
unique_geography_general <- unique(MASTER$geography_general)
circle_list <- list()
threshold <- 0.0001
for (value in unique_geography_general) {
  if (is.na(value)) {
    subset_df <- filter(MASTER, is.na(geography_general))
  } else {
    subset_df <- filter(MASTER, geography_general == value)
  }
  distances <- sqrt((subset_df$x - mean(subset_df$x))^2 + (subset_df$y - mean(subset_df$y))^2)
  center <- c(mean(subset_df$x[distances <= threshold]), mean(subset_df$y[distances <= threshold]))
  radius <- max(sqrt((subset_df$x[distances <= threshold] - center[1])^2 + (subset_df$y[distances <= threshold] - center[2])^2))
  circle <- list(value = value, center = center, radius = radius)
  circle_list[[length(circle_list) + 1]] <- circle
}

geo_general <- geo_general + geom_point() +
  lapply(circle_list, function(circle) {
    geom_circle(aes(x0 = circle$center[1], y0 = circle$center[2], r = circle$radius, color = circle$value),
                fill = NA)
  })
print(geo_general)

# Testing ellipsoid with percentage of points inside:  *** EXPERIMENTAL ***
unique_geography_general <- unique(MASTER$geography_general)
for (value in unique_geography_general) {
  if (is.na(value)) {
    subset_df <- filter(MASTER, is.na(geography_general))
  } else {
    subset_df <- filter(MASTER, geography_general == value)
  }
  with(subset_df ,dataEllipse(x, y, levels=0.8))
}

# library(ggalt)
new_medium <- medium + stat_ellipse(aes(fill = growth_medium, group = growth_medium), geom = "polygon", level = 0.90, alpha = 0.1) +
  theme(
    legend.text = element_text(size = 14),          # Bigger legend text
    legend.title = element_text(size = 16),         # Bigger legend title
    legend.key.size = unit(1.5, "cm")                # Bigger legend key
  ) + guides(color = guide_legend(override.aes = list(size = 4)))
print(new_medium)

# Testing showing means only


# Scatter with BROAD NICHE
broad_niche <- ggplot(MASTER, aes(x, y, color = broad_niche)) + geom_point()
print(broad_niche)
source("/home/rsampale/repo/candescence-grace/src/tlv/1-vae-creation/init.R")
setwd("/home/rsampale/repo/candescence-grace/src/tlv/2-analysis")

library(readr)

Orig_file_name = "Calb_Master_10062022.tsv" # Change to whatever your file is called

METADATA = file.path(TLV,"data_files",Orig_file_name)

master_table <- read_tsv(
  file=METADATA,
)

# Fix column formatting (replace . with _ ) and make lowercase:
colnames(master_table) <- gsub(x=colnames(master_table),pattern="\\.",replacement="_")
colnames(master_table)<-tolower(colnames(master_table))

# Convert data frame to tibble and save it to refined directory for further data analysis
master_table <- as_tibble(master_table)
save(master_table,file = file.path(TLV,"2.1-read_tsv_data","pre_umap_master.Rdata"))

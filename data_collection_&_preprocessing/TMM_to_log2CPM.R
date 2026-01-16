library(edgeR)

pathlist <- c("./Prat_et_al/TMM_rna_seq.txt",
              "./Gide_et_al/TMM_rna_seq.txt",
              "./Mariathasan/rna_expr_TMM.txt",
              "./Jung/rna_expr_TMM.txt")

pathlist <-c("./PratNSCLC/rna_expr_TMM.txt")

for(path in pathlist){
  # Read counts
  counts <- read.delim(path)
  
  # Create DGEList and normalize
  dge <- DGEList(counts = counts)
  dge <- calcNormFactors(dge, method="TMM")
  
  # Compute log2 CPM
  logCPM <- cpm(dge, log=TRUE, prior.count=1)
  
  # Create output path in the same folder
  out_folder <- dirname(path)
  out_path <- file.path(out_folder, "logcpm.txt")
  
  # Write logCPM to file
  write.table(logCPM, file = out_path, sep = "\t", quote = FALSE, col.names = NA)
}

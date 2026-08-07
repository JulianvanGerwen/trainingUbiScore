##### Background #####
# Turns the raw, unprocessed features in allLys_HsRevi_feats.csv (features for every
# lysine in the proteome) into the processed feature matrix used for training and for
# scoring sites. Mirrors the "Deal with missing values" / "Process features" steps of
# 11_training_models.Rmd.

library(caret)


##### Feature sets #####

# The features used to train the final model / score all sites
defaultFeatures <- c(
  "PRIDE_gold",
  "FinalBoxProcNum",
  "HotspotNew_bool",
  "mutR_am_pathogenicity_all",
  "interface_label",
  "NSP_p_q3_H",
  "NSP_p_q3_C",
  "NSP_rsa",
  "NSP_disorder",
  "PSP_boolOtherLysMod",
  "PSP_numAdj_5",
  "numQuant_all",
  "numReg_notPrtsm",
  "UNIPROT_in_region",
  "UNIPROT_in_domain"
)

# Features that get log-transformed by logTransformFeatures()
defaultLogFeatures <- c(
  "mutR_am_pathogenicity_all",
  "PSP_numAdj_5",
  "numQuant_all",
  "numReg_notPrtsm"
)

# Features that get mean-imputed by meanImputeFeatures()
defaultMeanImputeFeatures <- c("FinalBoxProcNum")


##### Load raw features #####

# Loads defaultPredSites.csv - the default set of sites to score/train on - as a character
# vector of uniprot_site.
loadDefaultPredSites <- function(dataDir = "data"){
  predSites <- read.csv(file.path(dataDir, "defaultPredSites.csv"))
  predSites$uniprot_site
}

# Loads allLys_HsRevi_feats.csv (features for every lysine in the proteome), sets rownames
# to uniprot_site, and subsets to `sites` (rownames) and `features` (columns). `sites`
# defaults to loadDefaultPredSites().
loadRawFeatures <- function(dataDir = "data", features = defaultFeatures, sites = NULL){
  allLysFeats <- read.csv(file.path(dataDir, "allLys_HsRevi_feats.csv"))
  rownames(allLysFeats) <- allLysFeats$uniprot_site
  if (is.null(sites)) sites <- loadDefaultPredSites(dataDir = dataDir)
  missing <- setdiff(sites, rownames(allLysFeats))
  if (length(missing) > 0) {
    warning(length(missing), " sites are not present in allLys_HsRevi_feats and will be dropped: ",
            paste(head(missing, 5), collapse = ", "))
    sites <- intersect(sites, rownames(allLysFeats))
  }
  allLysFeats[sites, features]
}


##### Processing steps #####

# Mean-impute NAs in `cols`, using the mean of each column's non-NA values. Columns not
# present in `data` (e.g. excluded via `features`) are silently skipped.
meanImputeFeatures <- function(data, cols = defaultMeanImputeFeatures){
  cols <- intersect(cols, colnames(data))
  for (col in cols) {
    vals <- data[, col]
    meanVal <- mean(vals, na.rm = TRUE)
    data[which(is.na(vals)), col] <- meanVal
  }
  data
}

# Drop rows with a missing value in any column
filterCompleteFeatures <- function(data){
  data[rowSums(is.na(data)) == 0, ]
}

# Convert logical columns to 0/1, as required by log-transforming/centering/scaling below
binariseLogicalFeatures <- function(data){
  for (col in colnames(data)) {
    if (is.logical(data[, col])) data[, col] <- as.numeric(data[, col])
  }
  data
}

# Log-transform `cols`. Columns containing zeros use log(1 + x) instead of log(x), to avoid
# -Inf. Columns not present in `data` (e.g. excluded via `features`) are silently skipped.
logTransformFeatures <- function(data, cols = defaultLogFeatures){
  cols <- intersect(cols, colnames(data))
  for (col in cols) {
    if (any(data[, col] == 0, na.rm = TRUE)) {
      data[, col] <- log(1 + data[, col])
    } else {
      data[, col] <- log(data[, col])
    }
  }
  data
}

# Center and scale every column (mean 0, sd 1)
centerScaleFeatures <- function(data){
  proc <- caret::preProcess(data, method = c("center", "scale"))
  predict(proc, data)
}


##### Full pipeline #####

# Loads allLys_HsRevi_feats.csv and runs the full processing pipeline:
# subset to `sites` -> mean-impute -> drop incomplete rows -> binarise logicals ->
# log-transform -> center/scale
# sites/meanImputeFeats/logFeats/centerScale let you customise or skip each step.
# raw = TRUE skips all processing, returning the subsetted-but-untouched features.
prepareFeatures <- function(dataDir = "data",
                            features = defaultFeatures,
                            sites = NULL,
                            meanImputeFeats = defaultMeanImputeFeatures,
                            logFeats = defaultLogFeatures,
                            centerScale = TRUE,
                            raw = FALSE){
  # Dedupe sites up front - a repeated site would otherwise produce duplicate rownames in
  # `data`, and any later rowname-based subsetting (e.g. buildTrainingData()) would then
  # return more rows for that one site than intended.
  if (!is.null(sites)) sites <- unique(sites)
  data <- loadRawFeatures(dataDir = dataDir, features = features, sites = sites)
  if (raw) return(data)
  data <- meanImputeFeatures(data, cols = meanImputeFeats)
  data <- filterCompleteFeatures(data)
  data <- binariseLogicalFeatures(data)
  data <- logTransformFeatures(data, cols = logFeats)
  if (centerScale) data <- centerScaleFeatures(data)
  data
}

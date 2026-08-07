##### Background #####
# User-facing entry point for retraining the model. Builds on the functions in
# model_training.R and feature_processing.R - source both before this one.


##### Load default data #####

# Loads defaultLabelData.csv and returns the default trainSites list directly (see
# runTrainingPipeline()): every site with response "X1" as positive, every site with
# response "X0" as negative. Feature values for training come from `datFeats` (see
# feature_processing.R's prepareFeatures()), so the two stay consistent.
loadDefaultTrainSites <- function(dataDir = "data"){
  labelData <- read.csv(file.path(dataDir, "defaultLabelData.csv"), row.names = 1)
  list(
    pos = rownames(labelData)[labelData$response == "X1"],
    neg = rownames(labelData)[labelData$response == "X0"]
  )
}


##### Build a training set from datFeats #####

# Builds a response + features data.frame by pulling posSites/negSites rows out of datFeats.
# posSites/negSites: character vectors of rownames(datFeats) - the positive (X1) and
# negative (X0) training sites.
buildTrainingData <- function(datFeats, posSites, negSites){
  overlap <- intersect(posSites, negSites)
  if (length(overlap) > 0) {
    stop("Sites cannot be in both posSites and negSites: ", paste(head(overlap, 5), collapse = ", "))
  }
  missingPos <- setdiff(posSites, rownames(datFeats))
  if (length(missingPos) > 0) {
    warning(length(missingPos), " posSites are not present in datFeats and will be dropped: ",
            paste(head(missingPos, 5), collapse = ", "))
    posSites <- setdiff(posSites, missingPos)
  }
  missingNeg <- setdiff(negSites, rownames(datFeats))
  if (length(missingNeg) > 0) {
    warning(length(missingNeg), " negSites are not present in datFeats and will be dropped: ",
            paste(head(missingNeg, 5), collapse = ", "))
    negSites <- setdiff(negSites, missingNeg)
  }
  response <- factor(rep(c("X1", "X0"), c(length(posSites), length(negSites))), levels = c("X1", "X0"))
  cbind(response = response, datFeats[c(posSites, negSites), ])
}


##### Full training pipeline #####

# datFeats: data.frame of features for all sites (rownames "<uniprot>_<site>"). Final
#   predictions are made for every row of this data.frame.
# trainSites: list(pos = ..., neg = ...) of rowname vectors of datFeats, giving the positive
#   (X1) and negative (X0) training sites. See loadDefaultTrainSites() for the default.
# numReps: number of repeated protein-stratified k-fold train-test splits to generate.
#   Ignored if customSplits is supplied
# k: number of folds per repeat
# customSplits: optional pre-built train-test splits (as produced by makeFoldsProtstrat() /
#   makeRepeatedProtstratSplits()), used instead of auto-generating splits from numReps/k
# splitSeed: base seed used when auto-generating splits
# trainSeed: base seed used for model training
# returnAllPredsLst: if TRUE, also return each split's model's individual predictions on
#   datFeats (allPredsDf is already their per-site median across splits)
#
# Returns a list (as produced by trainAndMakePreds_logreg()):
#   AUCdf            - AUC of every train-test split, on its own held-out test set
#   modsPredsTest_df - the underlying test-set predictions behind AUCdf
#   allPredsDf       - median predicted score for every site in datFeats, across all splits
#   mods             - the trained model for each split
#   allPredsLst      - only present if returnAllPredsLst = TRUE
runTrainingPipeline <- function(datFeats,
                                trainSites,
                                numReps = 1,
                                k = 5,
                                customSplits = NULL,
                                splitSeed = 123,
                                trainSeed = 80,
                                returnAllPredsLst = FALSE){
  trainDat <- buildTrainingData(datFeats = datFeats, posSites = trainSites$pos, negSites = trainSites$neg)

  if (is.null(customSplits)) {
    trainTestSplits <- makeRepeatedProtstratSplits(data = trainDat, numReps = numReps, k = k, seed = splitSeed)
  } else {
    trainTestSplits <- customSplits
  }

  trainAndMakePreds_logreg(
    trainTestSplits = trainTestSplits,
    datFeats = datFeats,
    seed = trainSeed,
    returnAllPredsLst = returnAllPredsLst
  )
}

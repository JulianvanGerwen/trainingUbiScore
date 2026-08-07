##### Background #####
# Core engine for training the weighted logistic regression classifier:
# evaluation metrics, protein-stratified train-test splitting, model training,
# and prediction extraction. See pipeline.R for the user-facing entry point.

library(dplyr)
library(purrr)
library(caret)
library(pROC)
library(yardstick)


##### Evaluation metrics #####

# summaryFunction for caret::trainControl(). Adds ROC AUC (positive class = lev[1])
# to the default Accuracy/Sensitivity/Specificity metrics.
accROCSummary <- function(data, lev = NULL, model = NULL){
  if (is.character(data$obs)) {
    data$obs <- factor(data$obs, levels = lev)
  }
  acc <- caret::postResample(data[, "pred"], data[, "obs"])

  if (length(lev) > 2) {
    stop(paste("Your outcome has", length(lev), "levels. accROCSummary() only supports two-class outcomes."))
  }
  if (!all(levels(data[, "pred"]) == lev)) {
    stop("levels of observed and predicted data do not match")
  }
  rocObject <- try(pROC::roc(data$obs, data[, lev[1]], direction = ">", quiet = TRUE), silent = TRUE)
  rocAUC <- if (inherits(rocObject, "try-error")) NA else rocObject$auc

  out <- c(rocAUC,
           caret::sensitivity(data[, "pred"], data[, "obs"], lev[1]),
           caret::specificity(data[, "pred"], data[, "obs"], lev[2]))
  names(out) <- c("ROC", "Sens", "Spec")
  c(acc, out)
}

# ROC AUC from a data.frame with an observed-class column (factor, levels X0/X1) and a
# predicted-probability-of-X1 column.
ROCAUC_fromPredictor <- function(data, obsCol, predCol){
  data[, c("obs", "pred")] <- data[, c(obsCol, predCol)]
  if (typeof(data$obs) == "logical") {
    data$obs <- factor(make.names(as.numeric(data$obs)), levels = c("X1", "X0"))
  }
  AUCDat <- yardstick::roc_auc(data, truth = "obs", "pred")
  as.numeric(AUCDat[1, ".estimate"])
}


##### Feature/response prep #####

combine_FeatResp <- function(feats, resp){
  comb <- cbind(resp, feats)
  colnames(comb)[1] <- "response"
  comb
}

# Weights that give the positive and negative classes equal total influence
caseWeights_balanced <- function(resp){
  tab <- table(resp)
  unname(0.5 / tab[as.character(resp)])
}


##### Train-test splitting #####

# 5-fold cross-validation control used inside model training, scored with accROCSummary
fitControl_CV <- caret::trainControl(
  method = "cv", number = 5,
  classProbs = TRUE,
  savePredictions = TRUE,
  summaryFunction = accROCSummary
)

# Train-test split stratified by response class and by protein (so no protein appears in
# both the train and test set of a fold). `data` must have a "response" column with values
# "X0"/"X1", and rownames formatted "<uniprot>_<site>".
makeFoldsProtstrat <- function(seed = 123, k = 5, data){
  data$uniprot <- gsub("_.+$", "", rownames(data))

  respMax <- function(x) "X1" %in% x
  protResp <- data[, c("uniprot", "response")] %>%
    group_by(uniprot) %>%
    summarise(response = respMax(response))

  set.seed(seed)
  protFolds <- caret::createFolds(protResp$response, k = k, returnTrain = TRUE)

  splits <- lapply(protFolds, function(trainIdx){
    trainProt <- protResp$uniprot[trainIdx]
    testProt  <- protResp$uniprot[-trainIdx]
    list(
      trainDat = subset(data, uniprot %in% trainProt) %>% dplyr::select(-uniprot),
      testDat  = subset(data, uniprot %in% testProt) %>% dplyr::select(-uniprot)
    )
  })
  names(splits) <- paste0("Fold", seq_along(splits))
  splits
}

# Repeat makeFoldsProtstrat() numReps times with a distinct seed each time, giving every
# fold across every repeat a unique name (e.g. "Rep1Fold1", "Rep2Fold1", ...).
makeRepeatedProtstratSplits <- function(data, numReps = 1, k = 5, seed = 123){
  allSplits <- list()
  for (j in seq_len(numReps)) {
    splits <- makeFoldsProtstrat(seed = seed + j, k = k, data = data)
    names(splits) <- paste0("Rep", j, names(splits))
    allSplits <- c(allSplits, splits)
  }
  allSplits
}


##### Model training #####

modTrain_LogReg_weighted <- function(feats, resp, fitControl, seed,
                                     weightFunc = caseWeights_balanced){
  weights <- weightFunc(resp)
  FeatResp_data <- combine_FeatResp(feats = feats, resp = resp)
  set.seed(seed)
  caret::train(response ~ ., data = FeatResp_data,
               method = "glm", family = "binomial",
               metric = "Accuracy", trControl = fitControl,
               weights = weights)
}


##### Predictions and results #####

# Applies trained models to their held-out test sets and to `datFeats`.
# Returns:
#   modsPredsTest_df - test-set predictions, one row per (split, held-out site)
#   AUCdf            - AUC within each split's test set
#   allPredsDf       - median predicted score per site in datFeats, across all splits
makePreds_fromMods <- function(mods, trainTestSplits, datFeats, returnAllPredsLst = FALSE){
  resultsLst <- list()

  modsPredsTest <- map2(mods, names(mods), function(mod, fold){
    testDat <- trainTestSplits[[fold]]$testDat
    predictions <- predict(mod, newdata = testDat[, -which(colnames(testDat) == "response")], type = "prob")
    data.frame(X1 = predictions[, "X1"], obs = testDat$response,
               fold = fold, uniprot_site = rownames(testDat))
  })
  modsPredsTest_df <- purrr::reduce(modsPredsTest, rbind) %>% mutate(Resample = fold)
  resultsLst$modsPredsTest_df <- modsPredsTest_df

  AUCs <- modsPredsTest_df %>% split(.$Resample) %>%
    map(~ROCAUC_fromPredictor(data = ., obsCol = "obs", predCol = "X1")) %>% unlist
  resultsLst$AUCdf <- data.frame(Resample = names(AUCs), AUC = AUCs)

  allPredsLst <- map2(mods, names(mods), function(mod, fold){
    predictions <- predict(mod, newdata = datFeats, type = "prob")
    data.frame(X1 = predictions[, "X1"], uniprot_site = rownames(datFeats))
  })
  allPredsLstDf <- map_dfr(allPredsLst, function(x) x, .id = "fold")
  resultsLst$allPredsDf <- allPredsLstDf %>%
    group_by(uniprot_site) %>%
    summarise(X1 = median(X1, na.rm = TRUE))

  if (returnAllPredsLst) resultsLst$allPredsLst <- allPredsLst
  resultsLst
}

# Trains one weighted logistic regression model per train-test split, then scores every
# model against its own test set and against `datFeats`.
# Returns a list: AUCdf, modsPredsTest_df, allPredsDf, mods
trainAndMakePreds_logreg <- function(trainTestSplits, datFeats, seed = 80, returnAllPredsLst = FALSE){
  mods <- vector("list", length(trainTestSplits))
  for (j in seq_along(trainTestSplits)) {
    trainDat <- trainTestSplits[[j]]$trainDat
    mods[[j]] <- modTrain_LogReg_weighted(
      feats = trainDat[, -which(colnames(trainDat) == "response")],
      resp = trainDat$response,
      fitControl = fitControl_CV, seed = seed + j,
      weightFunc = caseWeights_balanced
    )
  }
  names(mods) <- names(trainTestSplits)

  resultsLst <- makePreds_fromMods(mods = mods, trainTestSplits = trainTestSplits,
                                   datFeats = datFeats, returnAllPredsLst = returnAllPredsLst)
  resultsLst$mods <- mods
  resultsLst
}

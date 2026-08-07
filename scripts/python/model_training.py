"""Core engine for training the weighted logistic regression classifier: protein-stratified
train-test splitting, model training, and prediction extraction.

Python port of scripts/model_training.R. The R version trains via caret::train(), which
also runs an inner 5-fold cross-validation to report Accuracy/ROC/Sens/Spec (see
accROCSummary()/fitControl_CV in the R script) - but method="glm" has no hyperparameters to
tune, so that inner CV never shapes or selects the final model; it only populates
mod$resample/mod$results fields that trainAndMakePreds_logreg() never reads. This port
skips reproducing that inner-CV machinery entirely and fits scikit-learn's
LogisticRegression directly - the outer protein-stratified test-set evaluation (auc_df /
test_preds_df), which IS what gets used, is unaffected.

For the weighted fit itself: R's glm(weights = w) maximises the weighted log-likelihood
sum(w_i * loglik_i) with no regularisation. The closest scikit-learn equivalent is
LogisticRegression(C=float("inf")).fit(X, y, sample_weight=w) - unregularised weighted MLE,
optimised with LBFGS instead of R's IRLS. Both are exact optimisers of the same convex
objective, so coefficients should match closely (small differences only from convergence
tolerance), even though the two frameworks disagree on standard errors/inference - which
this pipeline never uses anyway.
"""

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold


##### Class weights #####

def case_weights_balanced(resp):
    """Per-sample weights giving the positive ("X1") and negative ("X0") classes equal total influence."""
    resp = pd.Series(resp)
    counts = resp.value_counts()
    return (0.5 / resp.map(counts)).to_numpy()


##### Train-test splitting #####

def make_folds_protstrat(data, k=5, seed=123):
    """Train-test split stratified by response class and by protein (so no protein appears
    in both the train and test set of a fold). `data` must have a "response" column with
    values "X0"/"X1", and an index formatted "<uniprot>_<site>".

    Uses sklearn's StratifiedKFold on protein-level labels in place of R's
    caret::createFolds() - both target balanced class proportions per fold, but the exact
    fold membership won't be identical between the two implementations even with the same
    seed, since the algorithms and RNG streams differ.
    """
    uniprot = pd.Series(data.index, index=data.index).str.replace(r"_.+$", "", regex=True)
    protein_resp = (
        pd.DataFrame({"uniprot": uniprot, "response": data["response"].to_numpy()})
        .groupby("uniprot")["response"]
        .apply(lambda x: bool((x == "X1").any()))
    )

    proteins = protein_resp.index.to_numpy()
    labels = protein_resp.to_numpy()
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)

    splits = {}
    for i, (train_idx, test_idx) in enumerate(skf.split(proteins, labels), start=1):
        train_proteins = set(proteins[train_idx])
        test_proteins = set(proteins[test_idx])
        splits[f"Fold{i}"] = {
            "train_dat": data[uniprot.isin(train_proteins)],
            "test_dat": data[uniprot.isin(test_proteins)],
        }
    return splits


def make_repeated_protstrat_splits(data, num_reps=1, k=5, seed=123):
    """Repeat make_folds_protstrat() num_reps times with a distinct seed each time, giving
    every fold across every repeat a unique name (e.g. "Rep1Fold1", "Rep2Fold1", ...)."""
    all_splits = {}
    for j in range(1, num_reps + 1):
        for name, split in make_folds_protstrat(data, k=k, seed=seed + j).items():
            all_splits[f"Rep{j}{name}"] = split
    return all_splits


##### Model training #####

def train_logreg_weighted(train_dat, seed=None):
    """Fits an unregularised, balanced-class-weighted logistic regression - see the module
    docstring for how this maps onto R's caret::train(..., method="glm", family="binomial").
    `seed` is accepted for API parity with the R version; LBFGS is deterministic, so it has
    no effect here.
    """
    X = train_dat.drop(columns="response")
    y = (train_dat["response"] == "X1").astype(int)
    weights = case_weights_balanced(train_dat["response"])
    # C=inf (rather than the deprecated penalty=None) switches off L2 regularisation:
    # as C -> inf the penalty term's contribution to the objective vanishes, leaving plain
    # weighted MLE.
    model = LogisticRegression(C=float("inf"), max_iter=1000, random_state=seed)
    model.fit(X, y, sample_weight=weights)
    return model


##### Predictions and results #####

def make_preds_from_models(models, train_test_splits, dat_feats, return_all_preds_lst=False):
    """Applies trained models to their held-out test sets and to `dat_feats`.
    Returns a dict with:
      test_preds_df - test-set predictions, one row per (split, held-out site)
      auc_df        - AUC within each split's test set
      all_preds_df  - median predicted score per site in dat_feats, across all splits
    """
    results = {}

    test_preds = []
    for fold, model in models.items():
        test_dat = train_test_splits[fold]["test_dat"]
        # Reindex to model.feature_names_in_ rather than trusting column order: unlike R's
        # formula interface (which matches by name), sklearn only warns - it does not
        # realign - if predict-time columns are ordered differently than at fit time, which
        # would otherwise silently score every prediction against the wrong features.
        X_test = test_dat.drop(columns="response")[model.feature_names_in_]
        test_preds.append(pd.DataFrame({
            "X1": model.predict_proba(X_test)[:, 1],
            "obs": test_dat["response"].to_numpy(),
            "fold": fold,
            "uniprot_site": test_dat.index,
        }))
    test_preds_df = pd.concat(test_preds, ignore_index=True)
    test_preds_df["Resample"] = test_preds_df["fold"]
    results["test_preds_df"] = test_preds_df

    aucs = {
        fold: roc_auc_score((d["obs"] == "X1").astype(int), d["X1"])
        for fold, d in test_preds_df.groupby("Resample")
    }
    results["auc_df"] = pd.DataFrame({"Resample": list(aucs.keys()), "AUC": list(aucs.values())})

    all_preds = []
    for fold, model in models.items():
        all_preds.append(pd.DataFrame({
            "X1": model.predict_proba(dat_feats[model.feature_names_in_])[:, 1],
            "uniprot_site": dat_feats.index,
            "fold": fold,
        }))
    all_preds_df = pd.concat(all_preds, ignore_index=True)
    results["all_preds_df"] = all_preds_df.groupby("uniprot_site")["X1"].median().reset_index()

    if return_all_preds_lst:
        results["all_preds_lst"] = all_preds
    return results


def train_and_make_preds_logreg(train_test_splits, dat_feats, seed=80, return_all_preds_lst=False):
    """Trains one weighted logistic regression model per train-test split, then scores
    every model against its own test set and against `dat_feats`.
    Returns a dict: auc_df, test_preds_df, all_preds_df, models
    """
    models = {
        fold: train_logreg_weighted(split["train_dat"], seed=seed + j)
        for j, (fold, split) in enumerate(train_test_splits.items(), start=1)
    }

    results = make_preds_from_models(
        models, train_test_splits, dat_feats, return_all_preds_lst=return_all_preds_lst
    )
    results["models"] = models
    return results

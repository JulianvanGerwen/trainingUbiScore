"""User-facing entry point for retraining the model. Python port of scripts/pipeline.R -
builds on model_training.py (import that module first, e.g. by adding scripts/python to
sys.path).
"""

import os
import warnings

import pandas as pd

from model_training import make_repeated_protstrat_splits, train_and_make_preds_logreg


##### Load default data #####

def load_default_train_sites(data_dir="data"):
    """Loads defaultLabelData.csv and returns the default train_sites dict directly (see
    run_training_pipeline()): every site with response "X1" as positive, every site with
    response "X0" as negative. Feature values for training come from `dat_feats` (see
    feature_processing.py's prepare_features()), so the two stay consistent.
    """
    label_data = pd.read_csv(os.path.join(data_dir, "defaultLabelData.csv"), index_col=0)
    return {
        "pos": label_data.index[label_data["response"] == "X1"].tolist(),
        "neg": label_data.index[label_data["response"] == "X0"].tolist(),
    }


##### Build a training set from dat_feats #####

def build_training_data(dat_feats, pos_sites, neg_sites):
    """Builds a response + features DataFrame by pulling pos_sites/neg_sites rows out of
    dat_feats. pos_sites/neg_sites: lists of dat_feats.index values - the positive (X1) and
    negative (X0) training sites.
    """
    overlap = set(pos_sites) & set(neg_sites)
    if overlap:
        raise ValueError(f"Sites cannot be in both pos_sites and neg_sites: {sorted(overlap)[:5]}")

    missing_pos = set(pos_sites) - set(dat_feats.index)
    if missing_pos:
        warnings.warn(f"{len(missing_pos)} pos_sites are not present in dat_feats and will be "
                      f"dropped: {', '.join(sorted(missing_pos)[:5])}")
        pos_sites = [s for s in pos_sites if s not in missing_pos]

    missing_neg = set(neg_sites) - set(dat_feats.index)
    if missing_neg:
        warnings.warn(f"{len(missing_neg)} neg_sites are not present in dat_feats and will be "
                      f"dropped: {', '.join(sorted(missing_neg)[:5])}")
        neg_sites = [s for s in neg_sites if s not in missing_neg]

    sites = list(pos_sites) + list(neg_sites)
    response = ["X1"] * len(pos_sites) + ["X0"] * len(neg_sites)
    train_dat = dat_feats.loc[sites].copy()
    train_dat.insert(0, "response", response)
    return train_dat


##### Full training pipeline #####

def run_training_pipeline(dat_feats, train_sites, num_reps=1, k=5, custom_splits=None,
                          split_seed=123, train_seed=80, return_all_preds_lst=False):
    """dat_feats: DataFrame of features for all sites (index = "<uniprot>_<site>"). Final
      predictions are made for every row of this DataFrame.
    train_sites: dict(pos=[...], neg=[...]) of dat_feats.index values, giving the positive
      (X1) and negative (X0) training sites. See load_default_train_sites() for the default.
    num_reps: number of repeated protein-stratified k-fold train-test splits to generate.
      Ignored if custom_splits is supplied
    k: number of folds per repeat
    custom_splits: optional pre-built train-test splits (as produced by
      make_folds_protstrat() / make_repeated_protstrat_splits()), used instead of
      auto-generating splits from num_reps/k
    split_seed: base seed used when auto-generating splits
    train_seed: base seed used for model training
    return_all_preds_lst: if True, also return each split's model's individual predictions
      on dat_feats (all_preds_df is already their per-site median across splits)

    Returns a dict (as produced by train_and_make_preds_logreg()):
      auc_df        - AUC of every train-test split, on its own held-out test set
      test_preds_df - the underlying test-set predictions behind auc_df
      all_preds_df  - median predicted score for every site in dat_feats, across all splits
      models        - the trained model for each split
      all_preds_lst - only present if return_all_preds_lst = True
    """
    train_dat = build_training_data(dat_feats, train_sites["pos"], train_sites["neg"])

    if custom_splits is None:
        train_test_splits = make_repeated_protstrat_splits(train_dat, num_reps=num_reps, k=k, seed=split_seed)
    else:
        train_test_splits = custom_splits

    return train_and_make_preds_logreg(
        train_test_splits, dat_feats, seed=train_seed, return_all_preds_lst=return_all_preds_lst
    )

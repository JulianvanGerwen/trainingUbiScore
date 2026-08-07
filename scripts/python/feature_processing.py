"""Turns the raw, unprocessed features in allLys_HsRevi_feats.csv (features for every
lysine in the proteome) into the processed feature matrix used for training and for
scoring sites.

Python port of scripts/feature_processing.R; mirrors the "Deal with missing values" /
"Process features" steps of 11_training_models.Rmd. centre_scale_features() deliberately
does not use sklearn's StandardScaler: scikit-learn scales by the population standard
deviation (ddof=0), while R's sd()/preProcess() - and pandas' own .std() default - use the
sample standard deviation (ddof=1). Matching R's behaviour just means using pandas directly.
"""

import os
import warnings

import numpy as np
import pandas as pd

##### Feature sets #####

# The features used to train the final model / score all sites
DEFAULT_FEATURES = [
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
    "UNIPROT_in_domain",
]

# Features that get log-transformed by log_transform_features()
DEFAULT_LOG_FEATURES = [
    "mutR_am_pathogenicity_all",
    "PSP_numAdj_5",
    "numQuant_all",
    "numReg_notPrtsm",
]

# Features that get mean-imputed by mean_impute_features()
DEFAULT_MEAN_IMPUTE_FEATURES = ["FinalBoxProcNum"]


##### Load raw features #####

def load_default_pred_sites(data_dir="data"):
    """Loads defaultPredSites.csv - the default set of sites to score/train on - as a list
    of uniprot_site."""
    pred_sites = pd.read_csv(os.path.join(data_dir, "defaultPredSites.csv"))
    return pred_sites["uniprot_site"].tolist()


def load_raw_features(data_dir="data", features=DEFAULT_FEATURES, sites=None):
    """Loads allLys_HsRevi_feats.csv (features for every lysine in the proteome), indexes
    by uniprot_site, and subsets to `sites` (rows) and `features` (columns). `sites`
    defaults to load_default_pred_sites().
    """
    all_lys_feats = pd.read_csv(os.path.join(data_dir, "allLys_HsRevi_feats.csv"))
    all_lys_feats = all_lys_feats.set_index("uniprot_site")

    if sites is None:
        sites = load_default_pred_sites(data_dir=data_dir)

    missing = sorted(set(sites) - set(all_lys_feats.index))
    if missing:
        warnings.warn(
            f"{len(missing)} sites are not present in allLys_HsRevi_feats and will be "
            f"dropped: {', '.join(missing[:5])}"
        )
        sites = [s for s in sites if s not in missing]

    return all_lys_feats.loc[sites, features]


##### Processing steps #####

def mean_impute_features(data, cols=DEFAULT_MEAN_IMPUTE_FEATURES):
    """Mean-impute NAs in `cols`, using the mean of each column's non-NA values. Columns not
    present in `data` (e.g. excluded via `features`) are silently skipped."""
    data = data.copy()
    cols = [c for c in cols if c in data.columns]
    for col in cols:
        data[col] = data[col].fillna(data[col].mean())
    return data


def filter_complete_features(data):
    """Drop rows with a missing value in any column."""
    return data.dropna(axis=0, how="any")


def binarise_logical_features(data):
    """Convert boolean columns to 0/1, as required by log-transforming/centering/scaling below."""
    data = data.copy()
    bool_cols = data.select_dtypes(include="bool").columns
    data[bool_cols] = data[bool_cols].astype(int)
    return data


def log_transform_features(data, cols=DEFAULT_LOG_FEATURES):
    """Log-transform `cols`. Columns containing zeros use log(1 + x) instead of log(x), to
    avoid -inf. Columns not present in `data` (e.g. excluded via `features`) are silently
    skipped."""
    data = data.copy()
    cols = [c for c in cols if c in data.columns]
    for col in cols:
        if (data[col] == 0).any():
            data[col] = np.log1p(data[col])
        else:
            data[col] = np.log(data[col])
    return data


def center_scale_features(data):
    """Center and scale every column (mean 0, sd 1), using the sample standard deviation
    (ddof=1) - see the module docstring for why this isn't sklearn's StandardScaler."""
    return (data - data.mean()) / data.std()


##### Full pipeline #####

def prepare_features(data_dir="data", features=DEFAULT_FEATURES, sites=None,
                      mean_impute_feats=DEFAULT_MEAN_IMPUTE_FEATURES,
                      log_feats=DEFAULT_LOG_FEATURES, center_scale=True, raw=False):
    """Loads allLys_HsRevi_feats.csv and runs the full processing pipeline:
    subset to `sites` -> mean-impute -> drop incomplete rows -> binarise logicals ->
    log-transform -> center/scale
    sites/mean_impute_feats/log_feats/center_scale let you customise or skip each step.
    raw=True skips all processing, returning the subsetted-but-untouched features.
    """
    # Dedupe sites up front - a repeated site would otherwise produce a duplicate index
    # label in `data`, and pandas expands EVERY later .loc[] lookup for that label to all
    # of its matching rows - silently inflating row counts in e.g. build_training_data().
    if sites is not None:
        sites = list(dict.fromkeys(sites))  # preserves order, unlike set()
    data = load_raw_features(data_dir=data_dir, features=features, sites=sites)
    if raw:
        return data
    data = mean_impute_features(data, cols=mean_impute_feats)
    data = filter_complete_features(data)
    data = binarise_logical_features(data)
    data = log_transform_features(data, cols=log_feats)
    if center_scale:
        data = center_scale_features(data)
    return data

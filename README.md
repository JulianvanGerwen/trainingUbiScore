# trainingUbiScore

This repository allows the user to retrain the ubi-site positional importance score for
a custom set of human lysines, provided that these are in the Uniprot reviewed human proteome.

The pipeline is implemented twice, in parallel: once in R (the original implementation)
and once in Python (a port with equivalent functionality). Pick whichever you prefer -
they read the same input data and produce the same kind of output.

## Repository layout
- `data/` - input features and default site lists (see the files loaded in
  `scripts/feature_processing.R`/`scripts/python/feature_processing.py` and
  `scripts/pipeline.R`/`scripts/python/pipeline.py`)
- `scripts/*.R` - R implementation (feature processing, model training, top-level pipeline)
- `scripts/python/*.py` - Python port of the above (same file names, snake_case functions)
- `training_pipeline.Rmd` - R walkthrough/example
- `training_pipeline.ipynb` - Python walkthrough/example (mirrors the Rmd)

## Dependencies

### R
`dplyr`, `purrr`, `caret`, `pROC`, `yardstick`

### Python
`pandas`, `numpy`, `scikit-learn`

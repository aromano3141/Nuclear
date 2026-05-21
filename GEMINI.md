Act as an expert Senior Research Engineer in Tabular Machine Learning, Geostatistics, and Explainable AI (XAI). You are tasked with completely rewriting `Nuclear_Tropical_Analysis.py` to transform it into a publication-ready comparative benchmarking study. 

Given that tree-based architectures vastly outperform deep learning on this sparse environmental dataset (Random Forest achieves an R² of ~0.591 while the Neural Network collapses to -0.334), you will completely strip out the PyTorch multi-task neural network architecture. In its place, you will construct a highly rigorous, robustly validated Tree Ensemble pipeline (Random Forest, XGBoost, and LightGBM) integrated with SHAP interpretability and spatial validation.

Maintain historical paths (`./Results_Tropical/`), use random seed `42` across all models for strict reproducibility, and implement the precise structural requirements outlined below.

================================================================================
CORE ARCHITECTURAL & METHODOLOGICAL REFACTORING
================================================================================

1. RESOLVE ISOTOPE BLINDNESS & RESTRUCTURE FEATURE SPACE
- Context: The previous model lacked the direct identity of the radionuclide/element being predicted, forcing identical predictions for vastly different isotopes within the same family.
- Requirements:
  a. Move 'Target' into `self.cat_cols` within the `DataPipeline` class so it is explicitly processed as an input feature.
  b. For models requiring numeric feature arrays (like Random Forest and XGBoost), use a unified `OneHotEncoder` or a clean `LabelEncoder` fallback strategy. For LightGBM, explicitly cast categorical string features to the `category` pandas dtype to leverage its optimized native categorical splits.

2. ELIMINATE COVARIANCE FABRICATION FROM GLOBAL MICE IMPUTATION
- Context: Global linear MICE imputation creates severe artifacts, inventing spurious soil-isotope correlations and flipping signs (e.g., Cs-137 vs Organic Matter).
- Requirements:
  a. Remove `IterativeImputer` entirely.
  b. Implement an Isotope-Stratified Median Imputation strategy within `process_features`. Group the soil data strictly by the `Target` feature. Compute the median value for each soil parameter within each specific isotope group using ONLY the training fold data.
  c. Apply these isotope-specific medians to fill missing values in both the training and testing segments. If a specific isotope group is completely missing a feature in the training partition, apply the global training median as a secure baseline fallback.
  d. Ensure LightGBM is evaluated twice: once on this imputed feature matrix, and once on the raw, unimputed data to scientifically test whether native missing-data tree split algorithms outperform explicit imputation.

3. ELEVATE VALIDATION RIGOR: GEOGRAPHIC GROUP K-FOLD CROSS-VALIDATION
- Context: Random train/test splits lead to data leakage if samples from identical geographic coordinates or sites populate both folds, artificially inflating model performance.
- Requirements:
  a. Replace `train_test_split` with a rigorous `GroupKFold` cross-validation strategy using `5` splits.
  b. Define the grouping key as the geographic proxy column: `df['Country']` (or `df['Site']` if country-level splits create massive, unmanageable class imbalances). 
  c. Train, cross-validate, and evaluate all models out-of-fold. Proving the model can accurately predict radionuclide transfer vectors in a completely unseen geographic territory is a core requirement for high-tier environmental journals.

4. IMPLEMENT THE ENSEMBLE BENCHMARKING SUITE
- Requirements: Build a unified training, cross-validation, and hyperparameter configuration engine for three production-grade architectures:
  a. **Random Forest Regressor** (Scikit-Learn): Tune with `n_estimators=150`, `max_depth=12`, and `min_samples_leaf=4`.
  b. **XGBoost Regressor** (`xgboost`): Configure using the modern hist-based algorithm (`tree_method='hist'`), `n_estimators=200`, `learning_rate=0.05`, and `max_depth=6`.
  c. **LightGBM Regressor** (`lightgbm`): Configure with `n_estimators=200`, `learning_rate=0.05`, `num_leaves=31`. Run this model natively on the unimputed dataset to serve as a baseline comparison against the imputed tree runs.

5. INTEGRATE EXPLAINABLE AI (XAI) VIA SHAP VALUES
- Context: Journals reject pure black-box models. Adding SHAP values translates predictive scores into verifiable, biogeochemical rules.
- Requirements:
  a. Install/import `shap`. Once the top-performing model is identified across the cross-validation rounds, instantiate a `TreeExplainer` on that model using the validation split feature space.
  b. Compute the global SHAP values for all core soil attributes (`pH`, `OM`, `Clay`, `CEC`) along with plant characteristics (`PFT`).
  c. Generate and save a crisp SHAP beeswarm plot to `MAIN_FIG_DIR / 'Fig5_SHAP_Summary.png'` showing comprehensive feature impacts.
  d. Export an automated text file to `STATS_DIR / 'SHAP_feature_rankings.txt'` listing features ranked by mean absolute SHAP value.

================================================================================
STATISTICAL EXPORTS & GRAPHICAL PIPELINE
================================================================================
- **Metrics Harmonization:** Evaluate all out-of-fold predictions on the original log-scale (reversing any Yeo-Johnson transforms via `target_transformer.inverse_transform` to maintain parity).
- **Comprehensive Summary File:** Export a structured, comprehensive metrics breakdown to `STATS_DIR / 'ensemble_vs_baseline_comparison.csv'`. This table must track:
  * Model Name | Scope (Global, Alkali, Metals, Actinides) | R² | RMSE | MAE | Sample Size (N)
- **Plots Generation:** Update the plotting scripts to output:
  * `Fig1_Correlations.png`: Pre-imputation Spearman correlation heatmap.
  * `Fig2_PCA.png`: Robust PCA loading patterns.
  * `Fig3_Soil_Dependence.png`: Raw bivariate regression lines for key target elements.
  * `Fig4_Ensemble_Performance.png`: Scatter plots showing Predicted vs. Actual values for the best ensemble architecture across each chemical family.

================================================================================
CODE SANITY CHECK
================================================================================
Provide the complete, self-contained Python script. Do not use code truncation, pseudocode placeholders, or missing block comments. Ensure all scikit-learn, xgboost, lightgbm, and shap imports are explicitly handled at the top of the file, and that the code handles edge cases where an entire geographic fold contains zero instances of rare Actinide elements gracefully without crashing.
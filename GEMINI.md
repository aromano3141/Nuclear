Act as an expert Research Engineer in AI and Geostatistics. I need you to refactor an environmental machine learning pipeline analyzing the IAEA MODARIA II tropical radionuclide dataset. A journal reviewer has requested a major revision based on specific architectural, validation, and geostatistical bottlenecks. 

Below is the exact code to modify, followed by the specific structural upgrades you must implement. 

=========================================
CURRENT CODEBASE (Nuclear_Tropical_Analysis.py)

=========================================
REQUIRED REFACTORING TASKS
=========================================

1. IMPLEMENT BASELINE BENCHMARKING MODELS
- To justify using a Deep Learning network, you must bench it against standard baseline models.
- In main(), after splitting data, train a Baseline Ridge Regression model and a Baseline Random Forest Regressor (or XGBoost) using the same continuous and encoded categorical features.
- Evaluate these baseline models on the test set (inverse-transforming targets just like the neural network evaluation).
- Calculate global and family-specific R² and RMSE scores for these baselines.
- Export a comprehensive model comparison to './Results_Tropical/Statistics/nn_vs_baseline.csv' detailing performance across the Neural Network, Ridge, and Random Forest models.

2. ADD IMPUTATION SENSITIVITY CHECK DIAGNOSTICS
- The reviewer is concerned that the global MICE (IterativeImputer) may be injecting artificial covariance structures into small isotope subsets (e.g., Sr-90 vs K-40).
- Before conducting soil parameter imputation, calculate the raw, unimputed Pearson and Spearman correlation matrices for the targets plotted in Section 3 (Cs-137, Sr-90, Ra-226, K-40) against the raw soil parameters ('pH', 'OM', 'Clay', 'CEC').
- Export these raw correlations along with their exact sample sizes (N) into a new diagnostic file: './Results_Tropical/Statistics/S03_raw_soil_dependence_stats.csv'.
- This allows a reviewer to directly compare the pre-imputation vs. post-imputation correlation trends to verify that MICE did not fabricate linear relationships.

3. LOG SPARSITY METRIC FOR ROBUST PCA
- In Section 2, before executing 'RobustPCA.iterative_svd(X_sparse)', compute the exact sparsity (percentage of missing/NaN entries) of the 'X_sparse' matrix.
- Print this percentage to the terminal during execution (e.g., "  → X_sparse Matrix Sparsity: XX.X%") and append this scalar metric to the final master statistics dictionary saved under 'stats_data'.

4. ELIMINATE CATEGORICAL DATA LEAKAGE
- In 'DataPipeline.process_features', the LabelEncoder is currently globally fit on the entire dataframe: 'LabelEncoder().fit(self.df[col].astype(str))'. This introduces subtle data leakage.
- Refactor this to fit the LabelEncoder strictly on the training indices 'tr_idx'. 
- To handle potential unseen categories in the test set, implement a robust fallback strategy (e.g., mapping unseen categories to an 'Unknown' token or forcing them to the most frequent class) before running '.transform()' on 'te_idx'.

5. ARCHITECTURAL TRANSPARENCY & CLARIFICATION
- Because each sample row contains a single target element family index, the network utilizes a '.gather()' layer during training, updating only one head's loss path at a time per sample.
- Add descriptive code comments inside 'MTL_CR_NN' and the training loop clarifying that this setup operates technically as an "Isotope-Conditioned Shared-Representation Trunk Network" rather than traditional simultaneous multi-label Multi-Task Learning. 

6. CODE CLEANUP
- Remove the redundant code block at the end of the script that writes the exact same 'stats_data' dictionary out twice to two separate filenames in the same directory. Keep only the write to 'S00_master_statistics.json'.

=========================================
OUTPUT REQUIREMENT
=========================================
Please provide the fully updated, self-contained Python script. Ensure all matplotlib figure saves, directory paths, and random seeds remain identical to preserve historical continuity where applicable.
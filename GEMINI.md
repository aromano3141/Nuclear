You are an expert Senior Machine Learning Engineer specializing in tabular neural networks, deep geostatistics, and hyperparameter optimization in PyTorch.

### OBJECTIVE
Further optimize the existing `Nuclear_Tropical_Analysis.py` script to push the Multi-Task Learning (MTL) model's global $R^2$ performance beyond the current 0.490 baseline. You will achieve this by implementing five high-efficiency, domain-specific deep learning enhancements without altering the fundamental data architecture or file output locations.

### CURRENT IMPLEMENTATION CONTEXT
The current codebase executes the following pipeline:
- Features: `feat_cols = ['pH', 'OM', 'Clay', 'Sand', 'Silt', 'CEC']` plus encoded Plant Functional Types (PFTs) and macro-climate proxies.
- Target Variable: Transformed via `df['log_CR'] = np.log10(df['CR'].clip(lower=1e-10))`.
- Neural Network Trunk (`CR_NN`): Uses a dense shared architecture containing a `nn.BatchNorm1d(64)` layer following the first hidden projection.
- Optimizer: `optim.Adam(model.parameters(), lr=0.001)` running across a flat 300-epoch loop.
- Multi-Task Head Loss: Computes a masked Mean Squared Error (MSE) loss across three specialized element heads (Alkali/Alkaline Earth, Heavy Metals, Natural Actinides) and combines them with equal weight.

---

### OPTIMIZATION TASKS TO IMPLEMENT

Please refactor the script to add the following five optimization upgrades:

#### 1. Domain-Specific Soil Chemical Interactions
Before passing the feature array to the `RobustScaler`, engineer two classic geostatistical interaction indicators within the Pandas dataframe to give the network explicit chemical context:
- Effective CEC Ratio: `df['CEC_Clay_Ratio'] = df['CEC'] / (df['Clay'] + 1e-5)` (acts as a proxy for structural clay mineral species and ion-exchange affinity).
- pH-Dependent Organic Carbon Availability: `df['pH_OM_Interaction'] = df['pH'] * df['OM']` (captures protonation changes in soil organic matter binding sites).
Append these two variables directly into your active `feat_cols` array.

#### 2. Replace BatchNorm with LayerNorm in the Shared Trunk
In `class CR_NN(nn.Module)`, swap out `nn.BatchNorm1d(64)` and replace it with `nn.LayerNorm(64)`. 
*Reasoning to preserve in code comments:* Batch Normalization tracks global running mean and variance metrics that become highly unstable during masked loss passes, because the missing data masking patterns fluctuate randomly from batch to batch. Layer Normalization operates independently per sample row, neutralizing this noise.

#### 3. Upgrade Target Scaling via a PowerTransformer
- Remove the baseline `np.log10` target transformation pass.
- Instead, import and initialize Scikit-Learn's `PowerTransformer(method='yeo-johnson')`.
- Fit and transform the concentration ratio (`CR`) arrays dynamically per task head to enforce true Gaussian normality. Ensure you store the transformation parameters properly so you can inverse-transform the model's test predictions (`y_pred_te`) back into their original scales when calculating final global and family-specific $R^2$ and RMSE evaluations.

#### 4. Implement Inverse-Frequency Loss Weighting
Currently, heavily sampled elements like K-40 (499 records) and Cs-137 (309 records) dominate the multi-task gradient updates compared to rare targets. 
- In the training loop, compute static weights for the three chemical family losses based on the inverse of their target sample densities (or an explicitly defined frequency balance factor).
- Update the loss calculation line to apply these weights dynamically: `loss = (w_alkali * loss_alkali) + (w_metals * loss_metals) + (w_actinides * loss_actinides)`.

#### 5. Add a Learning Rate Scheduler and Weight Decay
- Modify the optimizer instantiation to include a mild weight decay to mitigate overfitting: `optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)`.
- Import `ReduceLROnPlateau` from `torch.optim.lr_scheduler`.
- Configure the scheduler with `mode='min', factor=0.1, patience=15`.
- Integrate `scheduler.step(val_loss)` at the end of the training epoch loop using a validation split loss to allow the network to settle smoothly into deep local minima during the late stages of training.

---

### OUTPUT EXPECTATIONS
Return the complete, production-ready, refactored Python script. Ensure all logging steps, data loading logic, visualization functions (`Fig1` through `Fig4`), and JSON master statistics outputs remain fully intact, updated to handle the new `PowerTransformer` arrays and feature shapes seamlessly.
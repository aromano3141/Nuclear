"""
================================================================================
IAEA MODARIA II TROPICAL DATASET ANALYSIS - REFACTORED PRODUCTION GRADE
================================================================================

A production-grade refactoring of the Tropical Radioecological dataset analysis.
Implements isotope-stratified group imputation, advanced feature engineering, 
and an Isotope-Conditioned Multi-Task Learning (MTL) Neural Network.

Upgrades:
  1. Resolved Isotope Blindness via dedicated Target embeddings in the trunk.
  2. Fixed covariance distortion using Isotope-Stratified Group Imputation.
  3. Established baseline feature parity (OHE for Ridge, full metadata for RF).
  4. Comprehensive audit and export of model comparison metrics.

Author: Senior Research Engineer (AI/Geostatistics)
Output → ./Results_Tropical/
================================================================================
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import pearsonr, spearmanr
from sklearn.preprocessing import StandardScaler, RobustScaler, LabelEncoder, PowerTransformer, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
import warnings, json, copy, os
from datetime import datetime
from pathlib import Path

warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Reproducibility
np.random.seed(42)
torch.manual_seed(42)

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'figure.dpi': 150, 'savefig.dpi': 300, 'font.size': 10,
    'axes.labelsize': 11, 'axes.titlesize': 12, 'legend.fontsize': 9,
    'figure.max_open_warning': 50,
})

# ============================================================================
# CONFIGURATION & MAPPINGS
# ============================================================================

BASE_DIR = Path('.')
OUTPUT_DIR   = BASE_DIR / 'Results_Tropical'
MAIN_FIG_DIR = OUTPUT_DIR / 'Main_Figures'
SUPP_FIG_DIR = OUTPUT_DIR / 'Supplementary'
STATS_DIR    = OUTPUT_DIR / 'Statistics'

PFT_MAP = {
    'Rice': 'Cereal Grains', 'Maize': 'Cereal Grains', 'Corn': 'Cereal Grains', 'Millet': 'Cereal Grains',
    'Sorghum': 'Cereal Grains', 'Wheat': 'Cereal Grains', 'Paddy grass': 'Cereal Grains', 'Sedge grass': 'Cereal Grains',
    'Jasmine rice': 'Cereal Grains',
    'Yam': 'Root/Tuber Crops', 'Cassava': 'Root/Tuber Crops', 'Potato': 'Root/Tuber Crops', 
    'Round yam': 'Root/Tuber Crops', 'Sweet potato': 'Root/Tuber Crops', 'Taro': 'Root/Tuber Crops',
    'Greater galangal': 'Root/Tuber Crops', 'Bush carrot': 'Root/Tuber Crops', 'Radish': 'Root/Tuber Crops',
    'Arum': 'Root/Tuber Crops', 'Cocoyam': 'Root/Tuber Crops', 'Long yam': 'Root/Tuber Crops',
    'Lettuce': 'Leafy Vegetables', 'Cabbage': 'Leafy Vegetables', 'Waterleaf': 'Leafy Vegetables', 
    'Spinach': 'Leafy Vegetables', 'Amaranth': 'Leafy Vegetables', 'Red spinach': 'Leafy Vegetables',
    'Red amaranth': 'Leafy Vegetables', 'Pui shak': 'Leafy Vegetables', 'Helencha': 'Leafy Vegetables',
    'Cucumber': 'Fruit Vegetables', 'Tomato': 'Fruit Vegetables', 'Eggplant': 'Fruit Vegetables',
    'Brinjal': 'Fruit Vegetables', "Ladies' fingers": 'Fruit Vegetables', 'Pumpkin': 'Fruit Vegetables',
    'Cacao': 'Tree Fruits', 'Cocoa': 'Tree Fruits', 'Passionfruit': 'Tree Fruits', 'Pawpaw': 'Tree Fruits', 
    'Papaya': 'Tree Fruits', 'Mango': 'Tree Fruits', 'Banana': 'Tree Fruits', 
    'Orange': 'Tree Fruits', 'Coconut': 'Tree Fruits', 'Kakadu plum': 'Tree Fruits',
    'Bush apple': 'Tree Fruits', 'Bush plum (green)': 'Tree Fruits', 'Bush plum (black)': 'Tree Fruits',
    'Noni': 'Tree Fruits', 'Cluster fig': 'Tree Fruits', 'Bush fig': 'Tree Fruits',
    'Sand palm': 'Tree Fruits', 'Gooseberry': 'Tree Fruits', 'White currant': 'Tree Fruits',
    'Black currant': 'Tree Fruits', 'Pineapple': 'Tree Fruits', 'Mandarin': 'Tree Fruits', 'Lemon': 'Tree Fruits',
    'Teak': 'Woody Perennials', 'Acacia': 'Woody Perennials', 'Rubber': 'Woody Perennials', 'Tea': 'Woody Perennials',
    'Bean': 'Legumes', 'Lentil': 'Legumes'
}

CLIMATE_MAP = {
    'Nigeria': 'Humid Tropical', 'Ghana': 'Humid Tropical', 'Malaysia': 'Humid Tropical', 
    'Thailand': 'Humid Tropical', 'Vietnam': 'Humid Tropical', 'Benin': 'Humid Tropical', 
    'Cameroon': 'Humid Tropical', 'Tanzania': 'Humid Tropical', 'Indonesia': 'Humid Tropical', 
    'Sri Lanka': 'Humid Tropical', 'French Polynesia': 'Humid Tropical', 'Philippines': 'Humid Tropical',
    'India': 'Seasonally Dry Tropical', 'Brazil': 'Seasonally Dry Tropical', 
    'Australia': 'Arid Tropical', 'Marshall Islands': 'Humid Tropical', 
    'Cuba': 'Humid Tropical', 'Honduras': 'Humid Tropical', 'Ecuador': 'Humid Tropical', 
    'Peru': 'Humid Tropical', 'Bangladesh': 'Humid Tropical'
}

CHEMICAL_FAMILIES = {
    'Alkali': ['Cs-134', 'Cs-137', 'Sr-85', 'Sr-90', 'K-40'],
    'Metals': ['Pb', 'Cd', 'As', 'Cu', 'Zn', 'Cr', 'Ni', 'Co', 'Mn', 'Fe', 'Zn-65', 'Co-60', 'Sb', 'Se', 'Hg', 'Na', 'P', 'Rb', 'V', 'Sm', 'Sc', 'La', 'Ba'],
    'Actinides': ['U-238', 'Th-232', 'Ra-226', 'Ra-228', 'Pb-210', 'U-234', 'Th-230', 'Th-228', 'Po-210', 'Pu-239,240', 'Am-241']
}

TARGET_TO_FAMILY = {t: f for f, ts in CHEMICAL_FAMILIES.items() for t in ts}

# ============================================================================
# DATA PIPELINE CLASS
# ============================================================================

class DataPipeline:
    def __init__(self, csv_path):
        self.csv_path = csv_path
        self.df = None
        self.feature_scaler = RobustScaler()
        self.target_transformer = PowerTransformer(method='yeo-johnson')
        self.numeric_soil_cols = ['pH', 'OM', 'Clay', 'Sand', 'Silt', 'CEC']
        self.cat_cols = ['Target', 'PFT', 'Climate', 'Compartment']
        self.le_dict = {}
        self.fam_list = ['Alkali', 'Metals', 'Actinides']

    def load_and_preprocess(self):
        print(f"  Loading data from {self.csv_path}...")
        df_raw = pd.read_csv(self.csv_path)
        
        def safe_numeric(s):
            if s.dtype == 'object':
                s = s.astype(str).str.strip().str.replace(',', '.')
            return pd.to_numeric(s, errors='coerce')

        for col in self.numeric_soil_cols + ['CR']:
            df_raw[col] = safe_numeric(df_raw[col])

        df_raw['Target'] = df_raw['Radionuclide'].fillna(df_raw['Element'])
        df = df_raw.dropna(subset=['CR']).copy()
        
        df['log_CR'] = np.log10(df['CR'].clip(lower=1e-10))
        df['PFT'] = df['Common name'].map(PFT_MAP).fillna('Other')
        df['Climate'] = df['Country'].map(CLIMATE_MAP).fillna('Other')
        df['Family'] = df['Target'].map(TARGET_TO_FAMILY).fillna('Metals')
        df['Family_Idx'] = df['Family'].map({f: i for i, f in enumerate(self.fam_list)})
        
        self.df = df
        return df

    def split_data(self, test_size=0.2):
        print(f"  Splitting data (test_size={test_size})...")
        indices = np.arange(len(self.df))
        tr_idx, te_idx = train_test_split(indices, test_size=test_size, random_state=42)
        return tr_idx, te_idx

    def process_features(self, tr_idx, te_idx):
        print("  Running Leakage-Free Preprocessing...")
        
        # 1. Isotope-Stratified Group Imputation
        df_imp = self.df.copy()
        global_medians = self.df.iloc[tr_idx][self.numeric_soil_cols].median()
        
        for col in self.numeric_soil_cols:
            target_medians = self.df.iloc[tr_idx].groupby('Target')[col].median()
            for target in self.df['Target'].unique():
                mask = self.df['Target'] == target
                fill_val = target_medians.get(target, global_medians[col])
                if pd.isna(fill_val): fill_val = global_medians[col]
                df_imp.loc[mask, col] = df_imp.loc[mask, col].fillna(fill_val)
            df_imp[col] = df_imp[col].fillna(global_medians[col])

        # Normalization
        df_imp['pH'] = df_imp['pH'].clip(0, 14)
        for col in ['OM', 'Clay', 'Sand', 'Silt', 'CEC']:
            df_imp[col] = df_imp[col].clip(lower=0)
        
        texture_sum = df_imp[['Sand', 'Silt', 'Clay']].sum(axis=1)
        mask = texture_sum > 0
        df_imp.loc[mask, ['Sand', 'Silt', 'Clay']] = df_imp.loc[mask, ['Sand', 'Silt', 'Clay']].div(texture_sum[mask], axis=0) * 100

        for col in self.numeric_soil_cols:
            self.df[f'{col}_imputed'] = df_imp[col]

        # 2. Domain-Specific Interactions
        self.df['CEC_Clay_Ratio'] = self.df['CEC_imputed'] / (self.df['Clay_imputed'] + 1e-5)
        self.df['pH_OM_Interaction'] = self.df['pH_imputed'] * self.df['OM_imputed']
        
        # Leakage-free interaction fallback
        for col in ['CEC_Clay_Ratio', 'pH_OM_Interaction']:
            median_val = self.df.iloc[tr_idx][col].median()
            self.df[col] = self.df[col].fillna(median_val)
        
        # 3. Categorical Encoding (Leakage-Free)
        self.le_dict = {}
        X_cat_list = []
        for col in self.cat_cols:
            le = LabelEncoder()
            train_vals = self.df.iloc[tr_idx][col].astype(str)
            le.fit(train_vals)
            self.le_dict[col] = le
            most_freq = train_vals.mode()[0]
            full_vals = self.df[col].astype(str).copy()
            full_vals[~full_vals.isin(le.classes_)] = most_freq
            X_cat_list.append(le.transform(full_vals))
        X_cat = np.column_stack(X_cat_list)
        
        # 4. Continuous Scaling
        cont_cols = [f'{col}_imputed' for col in self.numeric_soil_cols] + ['CEC_Clay_Ratio', 'pH_OM_Interaction']
        X_cont_raw = self.df[cont_cols].values
        self.feature_scaler.fit(X_cont_raw[tr_idx])
        X_cont = self.feature_scaler.transform(X_cont_raw)
        
        # 5. Target Transformation
        y_raw = self.df[['CR']].values
        self.target_transformer.fit(y_raw[tr_idx])
        y_scaled = self.target_transformer.transform(y_raw)
        self.df['CR_scaled'] = y_scaled
        
        return X_cont, X_cat, y_scaled

# ============================================================================
# MTL NEURAL NETWORK CLASS
# ============================================================================

class MTL_CR_NN(nn.Module):
    def __init__(self, n_cont, cat_sizes, n_tasks=3, emb_dim=4):
        super().__init__()
        # Target Embedding
        self.target_emb = nn.Embedding(cat_sizes[0], emb_dim * 2)
        # Other embeddings (PFT, Climate, Compartment)
        self.embs = nn.ModuleList([nn.Embedding(s, emb_dim) for s in cat_sizes[1:]])
        
        total_in = n_cont + (emb_dim * 2) + (len(cat_sizes) - 1) * emb_dim
        
        self.shared = nn.Sequential(
            nn.Linear(total_in, 128), nn.LayerNorm(128), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(128, 64), nn.GELU()
        )
        self.heads = nn.ModuleList([
            nn.Sequential(nn.Linear(64, 32), nn.GELU(), nn.Linear(32, 1))
            for _ in range(n_tasks)
        ])
        
    def forward(self, x_cont, x_cat):
        t_emb = self.target_emb(x_cat[:, 0])
        other_embs = [emb(x_cat[:, i+1]) for i, emb in enumerate(self.embs)]
        
        x = torch.cat([x_cont, t_emb] + other_embs, dim=1)
        shared_out = self.shared(x)
        return torch.cat([head(shared_out) for head in self.heads], dim=1)

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    print("=" * 80)
    print("TROPICAL RADIONUCLIDE TRANSFER ANALYSIS - REFACTORED PIPELINE")
    print("=" * 80)

    for d in [MAIN_FIG_DIR, SUPP_FIG_DIR, STATS_DIR]:
        if d.exists():
            for f in d.glob('*'):
                if f.is_file(): f.unlink()
        d.mkdir(parents=True, exist_ok=True)

    pipeline = DataPipeline('iaea-modaria-ii-tropical-dataset.csv')
    df = pipeline.load_and_preprocess()
    
    # Pre-Imputation Diagnostics
    raw_soil_params = ['pH', 'OM', 'Clay', 'CEC']
    raw_targets = ['Cs-137', 'Sr-90', 'Ra-226', 'K-40']
    raw_stats = []
    for target in raw_targets:
        for param in raw_soil_params:
            sub = df[(df['Target'] == target) & (df[param].notna()) & (df['log_CR'].notna())]
            if len(sub) > 5:
                r_p, _ = pearsonr(sub[param], sub['log_CR'])
                r_s, _ = spearmanr(sub[param], sub['log_CR'])
                raw_stats.append({'Target': target, 'Param': param, 'Pearson_r': r_p, 'Spearman_r': r_s, 'N': len(sub)})
    pd.DataFrame(raw_stats).to_csv(STATS_DIR / 'S03_raw_soil_dependence_stats.csv', index=False)

    tr_idx, te_idx = pipeline.split_data()
    X_cont, X_cat, y_scaled = pipeline.process_features(tr_idx, te_idx)

    # Section 1: Correlation Analysis
    corr_cols = [f'{c}_imputed' for c in pipeline.numeric_soil_cols] + ['log_CR']
    corr_df = df[corr_cols].dropna()
    if len(corr_df) > 10:
        spearman_mat = corr_df.corr(method='spearman')
        spearman_mat.to_csv(STATS_DIR / 'S01_corr_spearman.csv')
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(spearman_mat, annot=True, cmap='coolwarm', center=0, ax=ax)
        fig.savefig(MAIN_FIG_DIR / 'Fig1_Correlations.png')

    # Section 4: Neural Network
    tr_df = df.iloc[tr_idx]
    fam_counts = tr_df['Family_Idx'].value_counts().sort_index()
    weights = 1.0 / (fam_counts + 1e-5)
    weights = weights / weights.sum() * len(pipeline.fam_list)
    fam_weights = torch.FloatTensor(weights.values)

    def to_t(idx):
        return (torch.FloatTensor(X_cont[idx]), torch.LongTensor(X_cat[idx]), 
                torch.FloatTensor(y_scaled[idx]).view(-1, 1),
                torch.LongTensor(df.iloc[idx]['Family_Idx'].values))

    tr_x_cont, tr_x_cat, tr_y, tr_fam = to_t(tr_idx)
    te_x_cont, te_x_cat, te_y, te_fam = to_t(te_idx)

    model = MTL_CR_NN(X_cont.shape[1], [len(pipeline.le_dict[c].classes_) for c in pipeline.cat_cols])
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=15)
    criterion = nn.MSELoss(reduction='none')

    print("  Training Neural Network...")
    train_losses, test_losses = [], []
    for epoch in range(600):
        model.train(); optimizer.zero_grad()
        out = model(tr_x_cont, tr_x_cat)
        
        # Calculate per-family loss for reporting
        with torch.no_grad():
            tr_preds = out.gather(1, tr_fam.view(-1, 1))
            family_losses = {}
            for i, fam in enumerate(pipeline.fam_list):
                f_mask = (tr_fam == i)
                if f_mask.sum() > 0:
                    f_loss = criterion(tr_preds[f_mask], tr_y[f_mask]).mean().item()
                    family_losses[fam] = f_loss
        
        raw_loss = criterion(out.gather(1, tr_fam.view(-1, 1)), tr_y)
        loss = (raw_loss * fam_weights[tr_fam].view(-1, 1)).mean()
        loss.backward(); optimizer.step()
        train_losses.append(loss.item())
        
        model.eval()
        with torch.no_grad():
            te_out = model(te_x_cont, te_x_cat)
            te_loss = criterion(te_out.gather(1, te_fam.view(-1, 1)), te_y).mean()
            test_losses.append(te_loss.item())
        
        scheduler.step(te_loss)
        
        if (epoch + 1) % 50 == 0:
            fam_str = " | ".join([f"{k}: {v:.4f}" for k, v in family_losses.items()])
            print(f"    Epoch {epoch+1:3d} | Total Loss: {loss.item():.4f} | Val: {te_loss.item():.4f} | {fam_str}")

    model.eval()
    with torch.no_grad():
        preds_scaled = model(te_x_cont, te_x_cat).gather(1, te_fam.view(-1, 1)).numpy()
        preds_cr = pipeline.target_transformer.inverse_transform(preds_scaled).flatten()
        preds = np.log10(np.clip(preds_cr, 1e-10, None))
        actual = df.iloc[te_idx]['log_CR'].values

    # Section 5: Baseline Benchmarking
    print(f"\n{'='*80}\nSECTION 5: BASELINE BENCHMARKING (INCLUDING 3 TREE MODELS)\n{'='*80}")
    
    # Prepare flat features for tree-based baselines
    X_rf = np.column_stack([X_cont, X_cat, df['Family_Idx'].values])
    
    # Pre-calculate common inverse transform helper
    def inv_log_cr(scaled_preds):
        cr_preds = pipeline.target_transformer.inverse_transform(scaled_preds.reshape(-1, 1)).flatten()
        return np.log10(np.clip(cr_preds, 1e-10, None))

    y_tr_flat = y_scaled[tr_idx].flatten()

    # 1. Decision Tree (Tree Model 1)
    print("  Fitting Decision Tree...")
    dt = DecisionTreeRegressor(random_state=42)
    dt.fit(X_rf[tr_idx], y_tr_flat)
    dt_preds = inv_log_cr(dt.predict(X_rf[te_idx]))
    print(f"    Decision Tree Fit Complete. Test R2: {r2_score(actual, dt_preds):.3f}")

    # 2. Random Forest (Tree Model 2)
    print("  Fitting Random Forest (Verbose)...")
    rf = RandomForestRegressor(n_estimators=100, random_state=42, verbose=1)
    rf.fit(X_rf[tr_idx], y_tr_flat)
    rf_preds = inv_log_cr(rf.predict(X_rf[te_idx]))

    # 3. Gradient Boosting (Tree Model 3)
    print("  Fitting Gradient Boosting (Iterative Verbose)...")
    gbr = GradientBoostingRegressor(n_estimators=100, random_state=42, verbose=1)
    gbr.fit(X_rf[tr_idx], y_tr_flat)
    gbr_preds = inv_log_cr(gbr.predict(X_rf[te_idx]))

    # 4. Ridge (Mathematical Baseline)
    print("\n  Fitting Ridge Regression...")
    ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    X_cat_ohe = ohe.fit_transform(X_cat[tr_idx])
    X_te_cat_ohe = ohe.transform(X_cat[te_idx])
    X_tr_ridge = np.column_stack([X_cont[tr_idx], X_cat_ohe])
    X_te_ridge = np.column_stack([X_cont[te_idx], X_te_cat_ohe])
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_tr_ridge, y_tr_flat)
    ridge_preds = inv_log_cr(ridge.predict(X_te_ridge))

    def get_metrics(a, p, current_fams=None):
        res = {}; m = np.isfinite(a) & np.isfinite(p)
        res['Global'] = (r2_score(a[m], p[m]), np.sqrt(mean_squared_error(a[m], p[m])))
        for i, fam in enumerate(pipeline.fam_list):
            f_m = (current_fams == i) & m
            if f_m.sum() > 5: res[fam] = (r2_score(a[f_m], p[f_m]), np.sqrt(mean_squared_error(a[f_m], p[f_m])))
        return res

    nn_m = get_metrics(actual, preds, te_fam.numpy())
    dt_m = get_metrics(actual, dt_preds, te_fam.numpy())
    rf_m = get_metrics(actual, rf_preds, te_fam.numpy())
    gb_m = get_metrics(actual, gbr_preds, te_fam.numpy())
    rd_m = get_metrics(actual, ridge_preds, te_fam.numpy())
    
    comp_data = []
    models_to_compare = [
        ('Neural Network', nn_m), ('Decision Tree', dt_m), 
        ('Random Forest', rf_m), ('Gradient Boosting', gb_m), ('Ridge Regression', rd_m)
    ]
    for model_name, metrics_dict in models_to_compare:
        for scope, (r2, rmse) in metrics_dict.items():
            comp_data.append({'Model': model_name, 'Scope': scope, 'R2': r2, 'RMSE': rmse})
    
    comparison_df = pd.DataFrame(comp_data)
    comparison_df.to_csv(STATS_DIR / 'nn_vs_baseline.csv', index=False)
    print("\nFinal Model Comparison (Global Scope):")
    print(comparison_df[comparison_df['Scope'] == 'Global'].to_string(index=False))

    print("\nREFACTORING COMPLETE. Analysis results saved to Results_Tropical/")

if __name__ == "__main__": main()

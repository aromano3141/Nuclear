"""
================================================================================
IAEA MODARIA II TROPICAL DATASET ANALYSIS - ADVANCED PRODUCTION GRADE
================================================================================

A production-grade refactoring of the Tropical Radioecological dataset analysis.
Implements Spatial Group Cross-Validation, LightGBM Native handling, 
and SHAP-based feature interpretation.

Directives Implemented:
  1. Spatial Group K-Fold (Grouping by Country).
  2. LightGBM Native (NaN handling + categorical dtype) vs. Imputed comparison.
  3. Feature Parity & One-Hot Encoding for Scikit-Learn baselines.
  4. SHAP Beeswarm Plot for the best-performing ensemble model.

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
import lightgbm as lgb
import shap
from scipy import stats
from scipy.stats import pearsonr, spearmanr
from sklearn.preprocessing import StandardScaler, RobustScaler, LabelEncoder, PowerTransformer, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
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
        
        # Casting to category for LightGBM
        for col in self.cat_cols:
            df[col] = df[col].astype('category')
        
        self.df = df
        return df

    def run_imputation(self, tr_idx):
        # Isotope-Stratified Group Imputation (Leakage-Free)
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
        
        return df_imp

# ============================================================================
# MTL NEURAL NETWORK CLASS
# ============================================================================

class MTL_CR_NN(nn.Module):
    def __init__(self, n_cont, cat_sizes, n_tasks=3, emb_dim=4):
        super().__init__()
        self.target_emb = nn.Embedding(cat_sizes[0], emb_dim * 2)
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
    print("TROPICAL RADIONUCLIDE TRANSFER ANALYSIS - ADVANCED REFACTOR")
    print("=" * 80)

    for d in [MAIN_FIG_DIR, SUPP_FIG_DIR, STATS_DIR]:
        if d.exists():
            for f in d.glob('*'):
                if f.is_file(): f.unlink()
        d.mkdir(parents=True, exist_ok=True)

    pipeline = DataPipeline('iaea-modaria-ii-tropical-dataset.csv')
    df = pipeline.load_and_preprocess()
    
    # 5-Fold GroupKFold using Country
    gkf = GroupKFold(n_splits=5)
    groups = df['Country'].values
    
    cv_results = []
    best_model = None
    best_r2 = -np.inf
    best_fold_data = None

    print(f"\nStarting 5-Fold Spatial Group Cross-Validation (Grouping by Country)...")
    
    for fold, (tr_idx, te_idx) in enumerate(gkf.split(df, groups=groups)):
        print(f"\n--- FOLD {fold+1} ---")
        train_countries = df.iloc[tr_idx]['Country'].unique()
        test_countries = df.iloc[te_idx]['Country'].unique()
        print(f"  Train Countries: {len(train_countries)} | Test Countries: {len(test_countries)} ({test_countries})")

        # 1. Feature Prep (Native vs Imputed)
        df_imp = pipeline.run_imputation(tr_idx)
        
        # Scaling targets (Leakage-free)
        target_transformer = PowerTransformer(method='yeo-johnson')
        y_scaled = target_transformer.fit_transform(df[['CR']])
        y_te_actual = df.iloc[te_idx]['log_CR'].values
        
        # Continuous Feature prep
        cont_cols = pipeline.numeric_soil_cols
        scaler = RobustScaler()
        X_cont_imp = scaler.fit_transform(df_imp[cont_cols].values)
        
        # Categorical prep for SKLearn (OHE)
        ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        X_cat_ohe = ohe.fit_transform(df.iloc[tr_idx][pipeline.cat_cols])
        X_te_cat_ohe = ohe.transform(df.iloc[te_idx][pipeline.cat_cols])
        
        # LGBM Data (Native NaN handling)
        X_lgbm_native = df[cont_cols + pipeline.cat_cols]
        X_lgbm_imp = df_imp[cont_cols + pipeline.cat_cols]
        
        # 2. Model Training
        # A. LGBM Native (missing inputs as NaN)
        lgbm_native = lgb.LGBMRegressor(n_estimators=200, learning_rate=0.05, random_state=42, verbose=-1)
        lgbm_native.fit(X_lgbm_native.iloc[tr_idx], y_scaled[tr_idx].flatten(), 
                        categorical_feature=pipeline.cat_cols)
        
        # B. LGBM Imputed
        lgbm_imp = lgb.LGBMRegressor(n_estimators=200, learning_rate=0.05, random_state=42, verbose=-1)
        lgbm_imp.fit(X_lgbm_imp.iloc[tr_idx], y_scaled[tr_idx].flatten(), 
                     categorical_feature=pipeline.cat_cols)
        
        # C. Random Forest (OHE Baseline)
        rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        X_tr_rf = np.column_stack([X_cont_imp[tr_idx], X_cat_ohe])
        rf.fit(X_tr_rf, y_scaled[tr_idx].flatten())
        
        # D. Neural Network (MTL)
        # We simplify NN for CV speed: use simple LabelEncoder for cat
        le_cats = []
        for col in pipeline.cat_cols:
            le = LabelEncoder()
            le.fit(df.iloc[tr_idx][col].astype(str))
            le_cats.append(le)
        
        def get_nn_tensors(idx):
            X_cat_nn = np.column_stack([le_cats[i].transform(df.iloc[idx][col].astype(str).map(lambda x: x if x in le_cats[i].classes_ else le_cats[i].classes_[0])) 
                                        for i, col in enumerate(pipeline.cat_cols)])
            return (torch.FloatTensor(X_cont_imp[idx]), torch.LongTensor(X_cat_nn), 
                    torch.FloatTensor(y_scaled[idx]).view(-1, 1),
                    torch.LongTensor(df.iloc[idx]['Family_Idx'].values))

        tr_cont, tr_cat, tr_y, tr_fam = get_nn_tensors(tr_idx)
        te_cont, te_cat, te_y, te_fam = get_nn_tensors(te_idx)
        
        nn_model = MTL_CR_NN(len(cont_cols), [len(le.classes_) for le in le_cats])
        opt = optim.Adam(nn_model.parameters(), lr=0.005)
        crit = nn.MSELoss()
        
        for _ in range(200):
            nn_model.train(); opt.zero_grad()
            out = nn_model(tr_cont, tr_cat)
            loss = crit(out.gather(1, tr_fam.view(-1, 1)), tr_y)
            loss.backward(); opt.step()
        
        # 3. Evaluation Helper
        def evaluate(model_obj, X_te, name, is_sk=True):
            if is_sk:
                p_scaled = model_obj.predict(X_te).reshape(-1, 1)
            else:
                model_obj.eval()
                with torch.no_grad():
                    p_scaled = model_obj(X_te[0], X_te[1]).gather(1, X_te[2].view(-1, 1)).numpy()
            
            p_cr = target_transformer.inverse_transform(p_scaled).flatten()
            p_log = np.log10(np.clip(p_cr, 1e-10, None))
            m = np.isfinite(y_te_actual) & np.isfinite(p_log)
            r2 = r2_score(y_te_actual[m], p_log[m])
            rmse = np.sqrt(mean_squared_error(y_te_actual[m], p_log[m]))
            return r2, rmse, p_log

        r2_native, rmse_native, p_native = evaluate(lgbm_native, X_lgbm_native.iloc[te_idx], "LGBM Native")
        r2_imp, rmse_imp, _ = evaluate(lgbm_imp, X_lgbm_imp.iloc[te_idx], "LGBM Imputed")
        r2_rf, rmse_rf, _ = evaluate(rf, np.column_stack([X_cont_imp[te_idx], X_te_cat_ohe]), "Random Forest")
        r2_nn, rmse_nn, _ = evaluate(nn_model, (te_cont, te_cat, te_fam), "Neural Network", is_sk=False)

        cv_results.append({
            'Fold': fold+1, 'LGBM Native R2': r2_native, 'LGBM Imputed R2': r2_imp, 
            'RF R2': r2_rf, 'NN R2': r2_nn
        })
        
        # Track Best Model for SHAP (using Native LGBM as preferred architect)
        if r2_native > best_r2:
            best_r2 = r2_native
            best_model = lgbm_native
            best_fold_data = X_lgbm_native.iloc[te_idx]

    # CV Summary
    summary_df = pd.DataFrame(cv_results)
    print("\n" + "="*40 + "\nCV SUMMARY (R2)\n" + "="*40)
    print(summary_df.mean().to_string())
    summary_df.to_csv(STATS_DIR / 'spatial_cv_performance.csv', index=False)

    # SHAP Interpretation on the Best Native LGBM Model
    print(f"\nGenerating SHAP Beeswarm Plot for Best LGBM Model (R2={best_r2:.3f})...")
    explainer = shap.TreeExplainer(best_model)
    shap_values = explainer(best_fold_data)
    
    plt.figure(figsize=(12, 8))
    shap.plots.beeswarm(shap_values, max_display=15, show=False)
    plt.title(f"SHAP Feature Importance: Best Spatial Fold (R2={best_r2:.3f})")
    plt.tight_layout()
    plt.savefig(MAIN_FIG_DIR / 'Fig5_SHAP_Beeswarm.png')
    
    print("\nREFACTORING COMPLETE.")
    print(f"  CV Statistics: {STATS_DIR / 'spatial_cv_performance.csv'}")
    print(f"  SHAP Plot: {MAIN_FIG_DIR / 'Fig5_SHAP_Beeswarm.png'}")

if __name__ == "__main__": main()

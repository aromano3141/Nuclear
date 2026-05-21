"""
================================================================================
IAEA MODARIA II TROPICAL DATASET ANALYSIS - PRODUCTION GRADE
================================================================================

A production-grade refactoring of the Tropical Radioecological dataset analysis.
Implements iterative imputation, robust missing-data PCA, advanced feature 
engineering for botanical/climatic context, and a Multi-Task Learning (MTL) 
Neural Network.

Upgrades:
  1. MICE-based Iterative Imputation for soil parameters.
  2. Iterative SVD (EM-PCA) for robust multivariate analysis.
  3. Expanded PFT (Plant Functional Type) and Macro-Climate mapping.
  4. Multi-Task Learning (MTL) NN with 3 chemical family heads.

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
from sklearn.preprocessing import StandardScaler, RobustScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge
import warnings, json, copy, os
from datetime import datetime
from pathlib import Path

warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.optim as optim

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

# UPGRADE 3: Expanded PFT Mapping
PFT_MAP = {
    # Cereal Grains & Grasses
    'Rice': 'Cereal Grains', 'Maize': 'Cereal Grains', 'Corn': 'Cereal Grains', 'Millet': 'Cereal Grains',
    'Sorghum': 'Cereal Grains', 'Wheat': 'Cereal Grains', 'Paddy grass': 'Cereal Grains', 'Sedge grass': 'Cereal Grains',
    'Jasmine rice': 'Cereal Grains',
    # Root/Tuber Crops
    'Yam': 'Root/Tuber Crops', 'Cassava': 'Root/Tuber Crops', 'Potato': 'Root/Tuber Crops', 
    'Round yam': 'Root/Tuber Crops', 'Sweet potato': 'Root/Tuber Crops', 'Taro': 'Root/Tuber Crops',
    'Greater galangal': 'Root/Tuber Crops', 'Bush carrot': 'Root/Tuber Crops', 'Radish': 'Root/Tuber Crops',
    'Arum': 'Root/Tuber Crops', 'Cocoyam': 'Root/Tuber Crops', 'Long yam': 'Root/Tuber Crops',
    # Leafy Vegetables
    'Lettuce': 'Leafy Vegetables', 'Cabbage': 'Leafy Vegetables', 'Waterleaf': 'Leafy Vegetables', 
    'Spinach': 'Leafy Vegetables', 'Amaranth': 'Leafy Vegetables', 'Red spinach': 'Leafy Vegetables',
    'Red amaranth': 'Leafy Vegetables', 'Pui shak': 'Leafy Vegetables', 'Helencha': 'Leafy Vegetables',
    # Fruit Vegetables
    'Cucumber': 'Fruit Vegetables', 'Tomato': 'Fruit Vegetables', 'Eggplant': 'Fruit Vegetables',
    'Brinjal': 'Fruit Vegetables', "Ladies' fingers": 'Fruit Vegetables', 'Pumpkin': 'Fruit Vegetables',
    # Tree Fruits
    'Cacao': 'Tree Fruits', 'Cocoa': 'Tree Fruits', 'Passionfruit': 'Tree Fruits', 'Pawpaw': 'Tree Fruits', 
    'Papaya': 'Tree Fruits', 'Mango': 'Tree Fruits', 'Banana': 'Tree Fruits', 
    'Orange': 'Tree Fruits', 'Coconut': 'Tree Fruits', 'Kakadu plum': 'Tree Fruits',
    'Bush apple': 'Tree Fruits', 'Bush plum (green)': 'Tree Fruits', 'Bush plum (black)': 'Tree Fruits',
    'Noni': 'Tree Fruits', 'Cluster fig': 'Tree Fruits', 'Bush fig': 'Tree Fruits',
    'Sand palm': 'Tree Fruits', 'Gooseberry': 'Tree Fruits', 'White currant': 'Tree Fruits',
    'Black currant': 'Tree Fruits', 'Pineapple': 'Tree Fruits', 'Mandarin': 'Tree Fruits', 'Lemon': 'Tree Fruits',
    # Woody Perennials & Shrubs
    'Teak': 'Woody Perennials', 'Acacia': 'Woody Perennials', 'Rubber': 'Woody Perennials', 'Tea': 'Woody Perennials',
    # Legumes
    'Bean': 'Legumes', 'Lentil': 'Legumes'
}

# UPGRADE 3: Regional Macro-Climate Proxy
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

# UPGRADE 4: Chemical Families
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
        self.imputer = IterativeImputer(estimator=BayesianRidge(), max_iter=20, random_state=42)
        self.numeric_soil_cols = ['pH', 'OM', 'Clay', 'Sand', 'Silt', 'CEC']
        self.cat_cols = ['PFT', 'Climate', 'Compartment']
        self.le_dict = {}

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
        
        # UPGRADE 3: Feature Engineering
        df['PFT'] = df['Common name'].map(PFT_MAP).fillna('Other')
        df['Climate'] = df['Country'].map(CLIMATE_MAP).fillna('Other')
        
        self.df = df
        return df

    def run_imputation(self):
        print("  Running Iterative Imputation (MICE)...")
        soil_data = self.df[self.numeric_soil_cols].copy()
        imputed = self.imputer.fit_transform(soil_data)
        df_imputed = pd.DataFrame(imputed, columns=self.numeric_soil_cols, index=self.df.index)

        # Constraints
        df_imputed['pH'] = df_imputed['pH'].clip(0, 14)
        for col in ['OM', 'Clay', 'Sand', 'Silt', 'CEC']:
            df_imputed[col] = df_imputed[col].clip(lower=0)

        # Texture normalization
        texture_sum = df_imputed[['Sand', 'Silt', 'Clay']].sum(axis=1)
        mask = texture_sum > 0
        df_imputed.loc[mask, ['Sand', 'Silt', 'Clay']] = df_imputed.loc[mask, ['Sand', 'Silt', 'Clay']].div(texture_sum[mask], axis=0) * 100

        for col in self.numeric_soil_cols:
            self.df[f'{col}_imputed'] = df_imputed[col]
        
        return df_imputed

    def get_mtl_features(self):
        FAMILIES = ['Alkali', 'Metals', 'Actinides']
        FAMILY_TO_IDX = {f: i for i, f in enumerate(FAMILIES)}
        
        self.le_dict = {col: LabelEncoder().fit(self.df[col].astype(str)) for col in self.cat_cols}
        
        X_cont = self.df[[f'{col}_imputed' for col in self.numeric_soil_cols]].values
        X_cat = np.column_stack([self.le_dict[col].transform(self.df[col].astype(str)) for col in self.cat_cols])
        
        self.df['Family'] = self.df['Target'].map(TARGET_TO_FAMILY).fillna('Metals')
        self.df['Family_Idx'] = self.df['Family'].map(FAMILY_TO_IDX)
        
        self.scaler_y = StandardScaler()
        self.df['log_CR_scaled'] = self.scaler_y.fit_transform(self.df[['log_CR']])
        
        return X_cont, X_cat, FAMILIES

# ============================================================================
# ROBUST PCA CLASS
# ============================================================================

class RobustPCA:
    @staticmethod
    def iterative_svd(X, n_components=5, max_iter=100, tol=1e-4):
        X_filled = np.where(np.isnan(X), np.nanmean(X, axis=0), X)
        prev_X = X_filled.copy()
        
        for i in range(max_iter):
            pca = PCA(n_components=n_components)
            X_pca = pca.fit_transform(X_filled)
            X_reconstructed = pca.inverse_transform(X_pca)
            X_filled[np.isnan(X)] = X_reconstructed[np.isnan(X)]
            
            diff = np.linalg.norm(X_filled - prev_X) / np.linalg.norm(prev_X)
            if diff < tol: break
            prev_X = X_filled.copy()
        return X_filled, pca

# ============================================================================
# MTL NEURAL NETWORK CLASS
# ============================================================================

class MTL_CR_NN(nn.Module):
    def __init__(self, n_cont, cat_sizes, n_tasks=3, emb_dim=4):
        super().__init__()
        self.embs = nn.ModuleList([nn.Embedding(s, emb_dim) for s in cat_sizes])
        total_in = n_cont + len(cat_sizes) * emb_dim
        self.shared = nn.Sequential(
            nn.Linear(total_in, 128), nn.BatchNorm1d(128), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(128, 64), nn.GELU()
        )
        self.heads = nn.ModuleList([
            nn.Sequential(nn.Linear(64, 32), nn.GELU(), nn.Linear(32, 1))
            for _ in range(n_tasks)
        ])
        
    def forward(self, x_cont, x_cat):
        embeddings = [emb(x_cat[:, i]) for i, emb in enumerate(self.embs)]
        x = torch.cat([x_cont] + embeddings, dim=1)
        shared_out = self.shared(x)
        return torch.cat([head(shared_out) for head in self.heads], dim=1)

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    print("=" * 80)
    print("TROPICAL RADIONUCLIDE TRANSFER ANALYSIS - PRODUCTION GRADE")
    print("=" * 80)

    for d in [MAIN_FIG_DIR, SUPP_FIG_DIR, STATS_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    pipeline = DataPipeline('iaea-modaria-ii-tropical-dataset.csv')
    df = pipeline.load_and_preprocess()
    df_imputed = pipeline.run_imputation()

    # SECTION 2: ROBUST PCA
    print(f"\n{'='*80}\nSECTION 2: ROBUST PCA\n{'='*80}")
    pivot_df = df.pivot_table(index=['Country', 'Site', 'Common name', 'Year'], columns='Target', values='log_CR')
    active_targets = pivot_df.notna().sum()[pivot_df.notna().sum() > 5].index.tolist()
    X_sparse = pivot_df[active_targets].values
    
    if X_sparse.shape[0] > 10:
        _, pca_model = RobustPCA.iterative_svd(X_sparse, n_components=min(6, len(active_targets)))
        ld_df = pd.DataFrame(pca_model.components_.T, index=active_targets, columns=[f'PC{i+1}' for i in range(pca_model.n_components_)])
        ld_df.to_csv(STATS_DIR / 'S04_robust_pca_loadings.csv')
        
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(ld_df, annot=True, cmap='RdBu_r', center=0, ax=ax)
        ax.set_title('Robust PCA Loadings (Iterative SVD)', fontweight='bold')
        plt.tight_layout()
        fig.savefig(MAIN_FIG_DIR / 'Fig2_PCA.png')
        print("  → Fig2_PCA saved")

    # SECTION 3: SOIL DEPENDENCE (UPGRADE: Using Imputed Data)
    print(f"\n{'='*80}\nSECTION 3: SOIL CHEMISTRY DEPENDENCE\n{'='*80}")
    soil_params = ['pH_imputed', 'OM_imputed', 'Clay_imputed', 'CEC_imputed']
    targets_to_plot = ['Cs-137', 'Sr-90', 'Ra-226', 'K-40']
    fig, axes = plt.subplots(len(targets_to_plot), len(soil_params), figsize=(20, 16))
    for i, target in enumerate(targets_to_plot):
        for j, param in enumerate(soil_params):
            ax = axes[i, j]
            sub = df[(df['Target'] == target) & (df['log_CR'].notna())]
            if len(sub) > 10:
                sns.regplot(x=param, y='log_CR', data=sub, ax=ax, scatter_kws={'alpha':0.2}, line_kws={'color':'red'})
                r, p = pearsonr(sub[param], sub['log_CR'])
                ax.set_title(f"{target} vs {param}\nr={r:.2f}, p={p:.2g}", fontsize=10)
    plt.tight_layout()
    fig.savefig(MAIN_FIG_DIR / 'Fig3_Soil_Dependence.png')
    print("  → Fig3_Soil_Dependence saved")

    # SECTION 4: MTL NEURAL NETWORK
    print(f"\n{'='*80}\nSECTION 4: MULTI-TASK LEARNING\n{'='*80}")
    X_cont, X_cat, FAMILIES = pipeline.get_mtl_features()
    
    indices = np.arange(len(df))
    tr_idx, te_idx = train_test_split(indices, test_size=0.2, random_state=42)
    
    def to_t(idx):
        return (torch.FloatTensor(X_cont[idx]), torch.LongTensor(X_cat[idx]), 
                torch.FloatTensor(df.iloc[idx]['log_CR_scaled'].values).view(-1, 1),
                torch.LongTensor(df.iloc[idx]['Family_Idx'].values))

    tr_x_cont, tr_x_cat, tr_y, tr_fam = to_t(tr_idx)
    te_x_cont, te_x_cat, te_y, te_fam = to_t(te_idx)

    model = MTL_CR_NN(X_cont.shape[1], [len(pipeline.le_dict[c].classes_) for c in pipeline.cat_cols])
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-3)
    criterion = nn.MSELoss(reduction='none')

    train_losses, test_losses = [], []
    for epoch in range(600):
        model.train(); optimizer.zero_grad()
        out = model(tr_x_cont, tr_x_cat)
        loss = criterion(out.gather(1, tr_fam.view(-1, 1)), tr_y).mean()
        loss.backward(); optimizer.step()
        train_losses.append(loss.item())
        
        model.eval()
        with torch.no_grad():
            te_out = model(te_x_cont, te_x_cat)
            te_loss = criterion(te_out.gather(1, te_fam.view(-1, 1)), te_y).mean()
            test_losses.append(te_loss.item())
        if (epoch+1)%100 == 0: print(f"    Epoch {epoch+1:3d} | Loss: {loss.item():.4f} | Val: {te_loss.item():.4f}")

    # Evaluation
    model.eval()
    with torch.no_grad():
        preds_scaled = model(te_x_cont, te_x_cat).gather(1, te_fam.view(-1, 1)).numpy()
        preds = pipeline.scaler_y.inverse_transform(preds_scaled).flatten()
        actual = df.iloc[te_idx]['log_CR'].values
    
    g_r2, g_rmse = r2_score(actual, preds), np.sqrt(mean_squared_error(actual, preds))
    print(f"\n  MTL Performance: R2 = {g_r2:.3f}, RMSE = {g_rmse:.3f}")

    # Family-specific scores
    fam_scores = {}
    for i, fam in enumerate(FAMILIES):
        mask = (te_fam.numpy() == i)
        if mask.sum() > 5:
            r2_f = r2_score(actual[mask], preds[mask])
            rmse_f = np.sqrt(mean_squared_error(actual[mask], preds[mask]))
            fam_scores[fam] = {'R2': r2_f, 'RMSE': rmse_f, 'N': int(mask.sum())}
            print(f"    {fam:<10}: R2 = {r2_f:.3f}, RMSE = {rmse_f:.3f}")

    # Final Plots
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    axes[0].plot(train_losses, label='Train'); axes[0].plot(test_losses, label='Val'); axes[0].legend()
    
    sns.scatterplot(x=actual, y=preds, hue=[FAMILIES[i] for i in te_fam.numpy()], alpha=0.5, ax=axes[1])
    axes[1].plot([actual.min(), actual.max()], [actual.min(), actual.max()], 'r--')
    axes[1].set_title(f"MTL Prediction (Global R²={g_r2:.3f})")
    plt.tight_layout(); fig.savefig(MAIN_FIG_DIR / 'Fig4_NN_Results.png')

    # Statistics Export
    stats_data = {
        'n_samples': len(df), 
        'global_performance': {'R2': g_r2, 'RMSE': g_rmse},
        'family_performance': fam_scores,
        'pft_distribution': df['PFT'].value_counts().to_dict(),
        'climate_distribution': df['Climate'].value_counts().to_dict()
    }
    with open(STATS_DIR / 'S00_master_statistics.json', 'w') as f: json.dump(stats_data, f, indent=4)
    print("\nANALYSIS COMPLETE.")

if __name__ == "__main__": main()

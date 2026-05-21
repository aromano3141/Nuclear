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
from sklearn.preprocessing import StandardScaler, RobustScaler, LabelEncoder, PowerTransformer
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.linear_model import BayesianRidge
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
        # Safeguard: IterativeImputer with fallback
        self.imputer = IterativeImputer(estimator=BayesianRidge(), max_iter=20, random_state=42)
        self.fallback_imputer = SimpleImputer(strategy='median')
        self.feature_scaler = RobustScaler()
        self.target_transformer = PowerTransformer(method='yeo-johnson')
        self.numeric_soil_cols = ['pH', 'OM', 'Clay', 'Sand', 'Silt', 'CEC']
        self.cat_cols = ['PFT', 'Climate', 'Compartment']
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
        
        # Keep log_CR for PCA and basic stats, but not for NN training
        df['log_CR'] = np.log10(df['CR'].clip(lower=1e-10))
        
        # UPGRADE 3: Feature Engineering
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
        
        # 1. Imputation (Fit on Train, Transform Both)
        soil_train = self.df.iloc[tr_idx][self.numeric_soil_cols].copy()
        soil_test = self.df.iloc[te_idx][self.numeric_soil_cols].copy()
        
        # Fit primary imputer
        imputed_train = self.imputer.fit_transform(soil_train)
        imputed_test = self.imputer.transform(soil_test)
        
        # Fit fallback to catch any remaining NaNs
        self.fallback_imputer.fit(imputed_train)
        imputed_train = self.fallback_imputer.transform(imputed_train)
        imputed_test = self.fallback_imputer.transform(imputed_test)
        
        # Combine back to apply constraints and texture normalization
        full_imputed = np.zeros((len(self.df), len(self.numeric_soil_cols)))
        full_imputed[tr_idx] = imputed_train
        full_imputed[te_idx] = imputed_test
        df_imp = pd.DataFrame(full_imputed, columns=self.numeric_soil_cols, index=self.df.index)

        # Constraints & Texture Normalization
        df_imp['pH'] = df_imp['pH'].clip(0, 14)
        for col in ['OM', 'Clay', 'Sand', 'Silt', 'CEC']:
            df_imp[col] = df_imp[col].clip(lower=0)
        
        texture_sum = df_imp[['Sand', 'Silt', 'Clay']].sum(axis=1)
        mask = texture_sum > 0
        df_imp.loc[mask, ['Sand', 'Silt', 'Clay']] = df_imp.loc[mask, ['Sand', 'Silt', 'Clay']].div(texture_sum[mask], axis=0) * 100

        for col in self.numeric_soil_cols:
            self.df[f'{col}_imputed'] = df_imp[col]

        # 2. Domain-Specific Soil Chemical Interactions (UPGRADE 1)
        self.df['CEC_Clay_Ratio'] = self.df['CEC_imputed'] / (self.df['Clay_imputed'] + 1e-5)
        self.df['pH_OM_Interaction'] = self.df['pH_imputed'] * self.df['OM_imputed']
        
        # Final NaN check for interaction features
        for col in ['CEC_Clay_Ratio', 'pH_OM_Interaction']:
            self.df[col] = self.df[col].fillna(self.df[col].median())
        
        # 3. Categorical Encoding
        self.le_dict = {col: LabelEncoder().fit(self.df[col].astype(str)) for col in self.cat_cols}
        X_cat = np.column_stack([self.le_dict[col].transform(self.df[col].astype(str)) for col in self.cat_cols])
        
        # 4. Continuous Scaling (UPGRADE 2: RobustScaler fit on Train)
        cont_cols = [f'{col}_imputed' for col in self.numeric_soil_cols] + ['CEC_Clay_Ratio', 'pH_OM_Interaction']
        X_cont_raw = self.df[cont_cols].values
        
        self.feature_scaler.fit(X_cont_raw[tr_idx])
        X_cont = self.feature_scaler.transform(X_cont_raw)
        
        # 5. Target Transformation (UPGRADE 4: PowerTransformer fit on Train)
        y_raw = self.df[['CR']].values
        self.target_transformer.fit(y_raw[tr_idx])
        y_scaled = self.target_transformer.transform(y_raw)
        self.df['CR_scaled'] = y_scaled
        
        return X_cont, X_cat, y_scaled

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
        
        # UPGRADE 3: Replace BatchNorm with LayerNorm
        # Batch Normalization tracks global running mean and variance metrics that become highly unstable 
        # during masked multi-task passes, because missing data patterns fluctuate randomly across batches 
        # depending on which elements are recorded. Layer Normalization operates independently per sample row, 
        # neutralizing this statistical noise.
        self.shared = nn.Sequential(
            nn.Linear(total_in, 128), nn.LayerNorm(128), nn.GELU(), nn.Dropout(0.2),
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

    # Clean and create directories
    for d in [MAIN_FIG_DIR, SUPP_FIG_DIR, STATS_DIR]:
        if d.exists():
            for f in d.glob('*'):
                if f.is_file(): f.unlink()
        d.mkdir(parents=True, exist_ok=True)

    pipeline = DataPipeline('iaea-modaria-ii-tropical-dataset.csv')
    df = pipeline.load_and_preprocess()
    
    # UPGRADE: Split BEFORE fit
    tr_idx, te_idx = pipeline.split_data()
    X_cont, X_cat, y_scaled = pipeline.process_features(tr_idx, te_idx)

    # SECTION 1: CORRELATION ANALYSIS (S01)
    print(f"\n{'='*80}\nSECTION 1: CORRELATION ANALYSIS\n{'='*80}")
    corr_cols = [f'{c}_imputed' for c in pipeline.numeric_soil_cols] + ['log_CR']
    corr_df = df[corr_cols].dropna()
    
    if len(corr_df) > 10:
        pearson_mat = corr_df.corr(method='pearson')
        spearman_mat = corr_df.corr(method='spearman')
        
        pearson_mat.to_csv(STATS_DIR / 'S01_corr_pearson.csv')
        spearman_mat.to_csv(STATS_DIR / 'S01_corr_spearman.csv')
        
        # Sample sizes for correlation
        n_mat = pd.DataFrame(index=corr_cols, columns=corr_cols)
        for c1 in corr_cols:
            for c2 in corr_cols:
                n_mat.loc[c1, c2] = len(df[[c1, c2]].dropna())
        n_mat.to_csv(STATS_DIR / 'S01_corr_sample_sizes.csv')
        
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(spearman_mat, annot=True, cmap='coolwarm', center=0, ax=ax)
        ax.set_title("Spearman Correlation Matrix", fontweight='bold')
        fig.savefig(MAIN_FIG_DIR / 'Fig1_Correlations.png')
        print("  → Fig1_Correlations saved")

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

    # SECTION 3: SOIL DEPENDENCE (Using Imputed Data)
    print(f"\n{'='*80}\nSECTION 3: SOIL CHEMISTRY DEPENDENCE\n{'='*80}")
    soil_params = ['pH_imputed', 'OM_imputed', 'Clay_imputed', 'CEC_imputed']
    targets_to_plot = ['Cs-137', 'Sr-90', 'Ra-226', 'K-40']
    dependence_stats = []
    
    fig, axes = plt.subplots(len(targets_to_plot), len(soil_params), figsize=(20, 16))
    for i, target in enumerate(targets_to_plot):
        for j, param in enumerate(soil_params):
            ax = axes[i, j]
            sub = df[(df['Target'] == target) & (df['log_CR'].notna())]
            if len(sub) > 10:
                sns.regplot(x=param, y='log_CR', data=sub, ax=ax, scatter_kws={'alpha':0.2}, line_kws={'color':'red'})
                r, p = pearsonr(sub[param], sub['log_CR'])
                ax.set_title(f"{target} vs {param}\nr={r:.2f}, p={p:.2g}", fontsize=10)
                dependence_stats.append({'Target': target, 'Param': param, 'r': r, 'p': p, 'N': len(sub)})
    
    pd.DataFrame(dependence_stats).to_csv(STATS_DIR / 'S03_soil_dependence_stats.csv', index=False)
    plt.tight_layout()
    fig.savefig(MAIN_FIG_DIR / 'Fig3_Soil_Dependence.png')
    print("  → Fig3_Soil_Dependence saved")

    # SECTION 4: MTL NEURAL NETWORK
    print(f"\n{'='*80}\nSECTION 4: MULTI-TASK LEARNING\n{'='*80}")
    
    # Inverse-Frequency Loss Weighting (UPGRADE 5)
    tr_df = df.iloc[tr_idx]
    fam_counts = tr_df['Family_Idx'].value_counts().sort_index()
    weights = 1.0 / (fam_counts + 1e-5)
    weights = weights / weights.sum() * len(pipeline.fam_list)
    fam_weights = torch.FloatTensor(weights.values)
    print(f"  Family weights: {dict(zip(pipeline.fam_list, weights.values))}")

    def to_t(idx):
        return (torch.FloatTensor(X_cont[idx]), torch.LongTensor(X_cat[idx]), 
                torch.FloatTensor(y_scaled[idx]).view(-1, 1),
                torch.LongTensor(df.iloc[idx]['Family_Idx'].values))

    tr_x_cont, tr_x_cat, tr_y, tr_fam = to_t(tr_idx)
    te_x_cont, te_x_cat, te_y, te_fam = to_t(te_idx)

    model = MTL_CR_NN(X_cont.shape[1], [len(pipeline.le_dict[c].classes_) for c in pipeline.cat_cols])
    
    # UPGRADE 6: Adam with weight decay and Scheduler
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=15)
    criterion = nn.MSELoss(reduction='none')

    train_losses, test_losses = [], []
    for epoch in range(600):
        model.train(); optimizer.zero_grad()
        out = model(tr_x_cont, tr_x_cat)
        
        # Apply weighting
        raw_loss = criterion(out.gather(1, tr_fam.view(-1, 1)), tr_y)
        batch_weights = fam_weights[tr_fam].view(-1, 1)
        loss = (raw_loss * batch_weights).mean()
        
        loss.backward(); optimizer.step()
        train_losses.append(loss.item())
        
        model.eval()
        with torch.no_grad():
            te_out = model(te_x_cont, te_x_cat)
            te_loss = criterion(te_out.gather(1, te_fam.view(-1, 1)), te_y).mean()
            test_losses.append(te_loss.item())
        
        scheduler.step(te_loss)
        if (epoch+1)%100 == 0: 
            curr_lr = optimizer.param_groups[0]['lr']
            print(f"    Epoch {epoch+1:3d} | Loss: {loss.item():.4f} | Val: {te_loss.item():.4f} | LR: {curr_lr:.2e}")

    # Evaluation (UPGRADE 4: Inverse Transform)
    model.eval()
    with torch.no_grad():
        preds_scaled = model(te_x_cont, te_x_cat).gather(1, te_fam.view(-1, 1)).numpy()
        preds_cr = pipeline.target_transformer.inverse_transform(preds_scaled).flatten()
        preds = np.log10(np.clip(preds_cr, 1e-10, None))
        actual = df.iloc[te_idx]['log_CR'].values
    
    # UPGRADE: Final NaN/Inf check and filtering for robust evaluation
    mask = np.isfinite(actual) & np.isfinite(preds)
    if mask.sum() < len(actual):
        print(f"  Warning: Filtered {len(actual) - mask.sum()} non-finite samples during evaluation.")
    
    actual_f, preds_f = actual[mask], preds[mask]
    
    if len(actual_f) > 0:
        g_r2, g_rmse = r2_score(actual_f, preds_f), np.sqrt(mean_squared_error(actual_f, preds_f))
        print(f"\n  MTL Performance (on log-scale): R2 = {g_r2:.3f}, RMSE = {g_rmse:.3f}")
    else:
        print("\n  Error: No valid samples for evaluation.")
        g_r2, g_rmse = 0.0, 0.0

    # Family-specific scores
    fam_scores = {}
    for i, fam in enumerate(pipeline.fam_list):
        f_mask = (te_fam.numpy() == i) & mask
        if f_mask.sum() > 5:
            r2_f = r2_score(actual[f_mask], preds[f_mask])
            rmse_f = np.sqrt(mean_squared_error(actual[f_mask], preds[f_mask]))
            fam_scores[fam] = {'R2': r2_f, 'RMSE': rmse_f, 'N': int(f_mask.sum())}
            print(f"    {fam:<10}: R2 = {r2_f:.3f}, RMSE = {rmse_f:.3f}")

    # Final Plots
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    axes[0].plot(train_losses, label='Train'); axes[0].plot(test_losses, label='Val'); axes[0].legend()
    axes[0].set_title("Training History (Weighted Loss)")
    
    sns.scatterplot(x=actual, y=preds, hue=[pipeline.fam_list[i] for i in te_fam.numpy()], alpha=0.5, ax=axes[1])
    axes[1].plot([actual.min(), actual.max()], [actual.min(), actual.max()], 'r--')
    axes[1].set_title(f"MTL Prediction (Global R²={g_r2:.3f})")
    plt.tight_layout(); fig.savefig(MAIN_FIG_DIR / 'Fig4_NN_Results.png')

    # SECTION 5: SUPPLEMENTARY FIGURES
    print(f"\n{'='*80}\nSECTION 5: SUPPLEMENTARY FIGURES\n{'='*80}")
    
    # FigS1: Distributions
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.kdeplot(data=df, x='log_CR', hue='Family', fill=True, ax=ax)
    ax.set_title("Log-Transfer Factor Distributions by Chemical Family")
    fig.savefig(SUPP_FIG_DIR / 'FigS1_Distributions.png')
    
    # FigS2: Country Comparison
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.boxplot(data=df, x='Country', y='log_CR', ax=ax)
    plt.xticks(rotation=45)
    ax.set_title("Log-Transfer Factor by Country")
    plt.tight_layout()
    fig.savefig(SUPP_FIG_DIR / 'FigS2_Country_Comparison.png')
    print("  → Supplementary figures saved")

    # Statistics Export
    stats_data = {
        'n_samples': len(df), 
        'global_performance': {'R2': g_r2, 'RMSE': g_rmse},
        'family_performance': fam_scores,
        'pft_distribution': df['PFT'].value_counts().to_dict(),
        'climate_distribution': df['Climate'].value_counts().to_dict(),
        'family_weights': dict(zip(pipeline.fam_list, weights.values.tolist()))
    }
    with open(STATS_DIR / 'S00_master_statistics.json', 'w') as f: json.dump(stats_data, f, indent=4)
    # Also save as master_statistics.json for redundancy as seen in original tree
    with open(STATS_DIR / 'master_statistics.json', 'w') as f: json.dump(stats_data, f, indent=4)
    print("\nANALYSIS COMPLETE.")

if __name__ == "__main__": main()

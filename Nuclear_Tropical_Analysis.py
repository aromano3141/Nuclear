"""
================================================================================
IAEA MODARIA II TROPICAL DATASET ANALYSIS
================================================================================

Adapts methodology from Chernobyl/Nuclear scripts to analyze the Tropical 
dataset. Focuses on soil-to-plant transfer factors (CR) and their 
dependence on soil chemistry and plant characteristics.

Methods implemented:
  1. Multivariate Correlation Analysis (Pearson + Spearman)
  2. PCA of Concentration Ratios (CR)
  3. Soil-Chemistry vs CR Dependence
  4. Neural Network Surrogate for CR Prediction
  5. Statistical Summary and Visualization

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
from scipy.stats import pearsonr, spearmanr, shapiro, ks_2samp
from sklearn.preprocessing import StandardScaler, RobustScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.linear_model import LinearRegression
import warnings, json, copy, os
from datetime import datetime
from pathlib import Path

warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.optim as optim

np.random.seed(42)
torch.manual_seed(42)

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'figure.dpi': 150, 'savefig.dpi': 300, 'font.size': 10,
    'axes.labelsize': 11, 'axes.titlesize': 12, 'legend.fontsize': 9,
    'figure.max_open_warning': 50,
})

# ============================================================================
# PATHS
# ============================================================================

BASE_DIR = Path('.')
OUTPUT_DIR   = BASE_DIR / 'Results_Tropical'
MAIN_FIG_DIR = OUTPUT_DIR / 'Main_Figures'
SUPP_FIG_DIR = OUTPUT_DIR / 'Supplementary'
STATS_DIR    = OUTPUT_DIR / 'Statistics'
for d in [MAIN_FIG_DIR, SUPP_FIG_DIR, STATS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("TROPICAL RADIONUCLIDE TRANSFER ANALYSIS")
print("=" * 80)
print(f"  Base dir  : {BASE_DIR.absolute()}")
print(f"  Output    : {OUTPUT_DIR}")
print(f"  Timestamp : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# ============================================================================
# SECTION 1 — DATA LOADING & PREPROCESSING
# ============================================================================

print(f"\n{'='*80}")
print("SECTION 1: DATA LOADING & PREPROCESSING")
print(f"{'='*80}")

csv_path = BASE_DIR / 'iaea-modaria-ii-tropical-dataset.csv'
assert csv_path.exists(), f"Not found: {csv_path}"

df_raw = pd.read_csv(csv_path)
print(f"  Raw shape : {df_raw.shape}")

# ---- safe numeric conversion ------------------------------------------------

def safe_numeric(s):
    if s.dtype == 'object':
        s = s.astype(str).str.strip().str.replace(',', '.')
    return pd.to_numeric(s, errors='coerce')

# Clean columns
numeric_cols = ['CR', 'C_plant', 'C_soil', 'pH', 'OM', 'Sand', 'Silt', 'Clay', 'CEC', 'Exch. K', 'Exch. Ca', 'Exch. Mg']
for col in numeric_cols:
    if col in df_raw.columns:
        df_raw[col] = safe_numeric(df_raw[col])

# Combine Radionuclide and Element for identification
df_raw['Target'] = df_raw['Radionuclide'].fillna(df_raw['Element'])

# Filter for relevant targets (those with enough data)
counts = df_raw.groupby('Target')['CR'].count()
TARGETS = counts[counts > 20].index.tolist()
print(f"  Analysis Targets (N > 20): {TARGETS}")

# Radionuclide Config for plotting (colors)
palette = sns.color_palette("husl", len(TARGETS))
RN_CFG = {t: {'c': palette[i]} for i, t in enumerate(TARGETS)}

# Log-transform targets
df = df_raw.copy()
df['log_CR'] = np.log10(df['CR'].clip(lower=1e-10))

# ============================================================================
# SECTION 2 — MULTIVARIATE CORRELATIONS
# ============================================================================

print(f"\n{'='*80}")
print("SECTION 2: MULTIVARIATE CORRELATIONS")
print(f"{'='*80}")

# Pivot to get multivariate view per record context
# We use Site, Country, Common name, and Year as the 'station' equivalent
id_cols = ['Country', 'Site', 'Common name', 'Year', 'Compartment']
pivot_df = df.pivot_table(index=id_cols, columns='Target', values='log_CR')

# Select targets present in pivot
avail_targets = [t for t in TARGETS if t in pivot_df.columns]
pivot_df = pivot_df[avail_targets]

print(f"  Pivoted shape: {pivot_df.shape}")

corr_pearson = pivot_df.corr(method='pearson')
corr_spearman = pivot_df.corr(method='spearman')

# Sample size matrix
ns_mat = np.zeros((len(avail_targets), len(avail_targets)))
for i, t1 in enumerate(avail_targets):
    for j, t2 in enumerate(avail_targets):
        ns_mat[i, j] = pivot_df[[t1, t2]].dropna().shape[0]

# Save stats
corr_pearson.to_csv(STATS_DIR / 'S01_corr_pearson.csv')
corr_spearman.to_csv(STATS_DIR / 'S01_corr_spearman.csv')
pd.DataFrame(ns_mat, index=avail_targets, columns=avail_targets).to_csv(STATS_DIR / 'S01_corr_sample_sizes.csv')

# Plot correlations
fig, axes = plt.subplots(1, 2, figsize=(18, 8))
mask = np.triu(np.ones_like(corr_pearson, dtype=bool))

sns.heatmap(corr_pearson, mask=mask, cmap='RdBu_r', center=0, annot=True, fmt=".2f", ax=axes[0], vmin=-1, vmax=1)
axes[0].set_title('Pearson Correlation (log10 CR)', fontweight='bold')

sns.heatmap(corr_spearman, mask=mask, cmap='RdBu_r', center=0, annot=True, fmt=".2f", ax=axes[1], vmin=-1, vmax=1)
axes[1].set_title('Spearman Correlation (log10 CR)', fontweight='bold')

plt.tight_layout()
fig.savefig(MAIN_FIG_DIR / 'Fig1_Correlations.png')
print("  → Fig1_Correlations saved")

# ============================================================================
# SECTION 3 — PCA OF CONCENTRATION RATIOS
# ============================================================================

print(f"\n{'='*80}")
print("SECTION 3: PCA OF CONCENTRATION RATIOS")
print(f"{'='*80}")

# For PCA, we need sites with multiple targets.
# Let's try to find the largest subset of targets and sites that form a dense matrix
target_counts = pivot_df.notna().sum().sort_values(ascending=False)
for n_top in [8, 6, 4]:
    top_targets_pca = target_counts.head(n_top).index.tolist()
    pca_data = pivot_df[top_targets_pca].dropna()
    if pca_data.shape[0] > 15:
        print(f"  Performing PCA on {n_top} targets with {pca_data.shape[0]} samples.")
        break
else:
    pca_data = pd.DataFrame()

if not pca_data.empty:
    sc = StandardScaler()
    pca = PCA()
    scores = pca.fit_transform(sc.fit_transform(pca_data))
    
    ld_df = pd.DataFrame(pca.components_.T, index=pca_data.columns, columns=[f'PC{i+1}' for i in range(pca_data.shape[1])])
    ld_df.to_csv(STATS_DIR / 'S02_pca_loadings.csv')
    
    # Plot PCA
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # Scree Plot
    axes[0].bar(range(1, len(pca.explained_variance_ratio_) + 1), pca.explained_variance_ratio_ * 100, color='steelblue', alpha=0.7)
    axes[0].plot(range(1, len(pca.explained_variance_ratio_) + 1), np.cumsum(pca.explained_variance_ratio_) * 100, 'ro-', label='Cumulative')
    axes[0].set_title('Scree Plot', fontweight='bold')
    axes[0].set_xlabel('Principal Component')
    axes[0].set_ylabel('Variance Explained (%)')
    axes[0].legend()
    
    # Biplot (Loadings)
    sns.heatmap(ld_df.iloc[:, :min(4, ld_df.shape[1])], annot=True, cmap='RdBu_r', center=0, ax=axes[1])
    axes[1].set_title('PCA Loadings', fontweight='bold')
    
    plt.tight_layout()
    fig.savefig(MAIN_FIG_DIR / 'Fig2_PCA.png')
    print("  → Fig2_PCA saved")
else:
    print("  ⚠ Insufficient overlapping data for any PCA subset")

# ============================================================================
# SECTION 4 — SOIL CHEMISTRY DEPENDENCE
# ============================================================================

print(f"\n{'='*80}")
print("SECTION 4: SOIL CHEMISTRY DEPENDENCE")
print(f"{'='*80}")

soil_params = ['pH', 'OM', 'Clay', 'CEC']
targets_to_plot = ['Cs-137', 'Sr-90', 'Ra-226', 'K-40']

fig, axes = plt.subplots(len(targets_to_plot), len(soil_params), figsize=(20, 16))

soil_stats = []

for i, target in enumerate(targets_to_plot):
    for j, param in enumerate(soil_params):
        ax = axes[i, j]
        sub = df[(df['Target'] == target) & (df[param].notna()) & (df['log_CR'].notna())]
        if len(sub) > 10:
            sns.regplot(x=param, y='log_CR', data=sub, ax=ax, scatter_kws={'alpha':0.3, 's':10}, line_kws={'color':'red'})
            r, p = pearsonr(sub[param], sub['log_CR'])
            ax.set_title(f"{target} vs {param}\nr={r:.2f}, p={p:.2g}", fontsize=10)
            soil_stats.append({'Target': target, 'Param': param, 'r': r, 'p': p, 'N': len(sub)})
        else:
            ax.text(0.5, 0.5, 'Insufficient Data', ha='center', va='center')
            ax.set_title(f"{target} vs {param}")

plt.tight_layout()
fig.savefig(MAIN_FIG_DIR / 'Fig3_Soil_Dependence.png')
pd.DataFrame(soil_stats).to_csv(STATS_DIR / 'S03_soil_dependence_stats.csv', index=False)
print("  → Fig3_Soil_Dependence saved")

# ============================================================================
# SECTION 5 — NEURAL NETWORK SURROGATE
# ============================================================================

print(f"\n{'='*80}")
print("SECTION 5: NEURAL NETWORK SURROGATE")
print(f"{'='*80}")

# Predict log_CR using soil properties and plant compartment (categorical)
feat_cols = ['pH', 'OM', 'Clay', 'Sand', 'Silt', 'CEC']
# Drop rows where all features are NaN
nn_data = df.dropna(subset=['log_CR'])
# For features, we fill NaN with median to keep as much data as possible
for col in feat_cols:
    if col in nn_data.columns:
        nn_data[col] = nn_data[col].fillna(nn_data[col].median())

# Encoding categorical features
le_target = LabelEncoder()
nn_data['Target_Idx'] = le_target.fit_transform(nn_data['Target'])
le_comp = LabelEncoder()
nn_data['Comp_Idx'] = le_comp.fit_transform(nn_data['Compartment'].astype(str))

X = nn_data[feat_cols + ['Target_Idx', 'Comp_Idx']].values
y = nn_data['log_CR'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale
scaler_X = RobustScaler()
X_tr = scaler_X.fit_transform(X_train)
X_te = scaler_X.transform(X_test)

y_mean, y_std = y_train.mean(), y_train.std()
y_tr = (y_train - y_mean) / y_std
y_te = (y_test - y_mean) / y_std

# Tensors
X_tr_t = torch.FloatTensor(X_tr)
y_tr_t = torch.FloatTensor(y_tr).view(-1, 1)
X_te_t = torch.FloatTensor(X_te)
y_te_t = torch.FloatTensor(y_te).view(-1, 1)

class CR_NN(nn.Module):
    def __init__(self, n_in):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, 64),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Linear(32, 1)
        )
    def forward(self, x):
        return self.net(x)

model = CR_NN(X.shape[1])
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

train_losses = []
test_losses = []

print("  Training NN...")
for epoch in range(300):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_tr_t)
    loss = criterion(outputs, y_tr_t)
    loss.backward()
    optimizer.step()
    train_losses.append(loss.item())
    
    model.eval()
    with torch.no_grad():
        te_loss = criterion(model(X_te_t), y_te_t)
        test_losses.append(te_loss.item())

# Evaluate
model.eval()
with torch.no_grad():
    y_pred_te = model(X_te_t).numpy() * y_std + y_mean
    y_true_te = y_test

r2 = r2_score(y_true_te, y_pred_te)
rmse = np.sqrt(mean_squared_error(y_true_te, y_pred_te))
print(f"  NN Performance: R2 = {r2:.3f}, RMSE = {rmse:.3f}")

# Plot results
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
axes[0].plot(train_losses, label='Train')
axes[0].plot(test_losses, label='Test')
axes[0].set_title('Loss Convergence')
axes[0].legend()

axes[1].scatter(y_true_te, y_pred_te, alpha=0.3)
axes[1].plot([y_true_te.min(), y_true_te.max()], [y_true_te.min(), y_true_te.max()], 'r--')
axes[1].set_title(f'Prediction vs Actual (R2={r2:.2f})')
axes[1].set_xlabel('Actual log10(CR)')
axes[1].set_ylabel('Predicted log10(CR)')

plt.tight_layout()
fig.savefig(MAIN_FIG_DIR / 'Fig4_NN_Results.png')
print("  → Fig4_NN_Results saved")

# ============================================================================
# SECTION 6 — SUPPLEMENTARY: DISTRIBUTIONS AND COUNTRY ANALYSIS
# ============================================================================

print(f"\n{'='*80}")
print("SECTION 6: SUPPLEMENTARY ANALYSIS")
print(f"{'='*80}")

# Distribution of log_CR for top radionuclides
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
top_rn = ['Cs-137', 'Sr-90', 'Ra-226', 'K-40', 'Th-232', 'U-238']
for i, rn in enumerate(top_rn):
    ax = axes[i//3, i%3]
    data = df[df['Target'] == rn]['log_CR'].dropna()
    if len(data) > 0:
        sns.histplot(data, kde=True, ax=ax, color=RN_CFG.get(rn, {'c':'blue'})['c'])
        ax.set_title(f'{rn} log10(CR) Distribution (N={len(data)})', fontweight='bold')
    else:
        ax.text(0.5, 0.5, 'No Data', ha='center', va='center')

plt.tight_layout()
fig.savefig(SUPP_FIG_DIR / 'FigS1_Distributions.png')
print("  → FigS1_Distributions saved")

# Country-level Boxplot for Cs-137
fig, ax = plt.subplots(figsize=(14, 7))
cs137_data = df[df['Target'] == 'Cs-137']
if len(cs137_data) > 0:
    sns.boxplot(x='Country', y='log_CR', data=cs137_data, ax=ax, palette='Set3')
    ax.set_title('Cs-137 log10(CR) by Country', fontweight='bold')
    plt.xticks(rotation=45)
    plt.tight_layout()
    fig.savefig(SUPP_FIG_DIR / 'FigS2_Country_Comparison.png')
    print("  → FigS2_Country_Comparison saved")

# ============================================================================
# SECTION 7 — MASTER STATISTICS
# ============================================================================

print(f"\n{'='*80}")
print("SECTION 7: MASTER STATISTICS")
print(f"{'='*80}")

master_stats = {
    'total_records': len(df),
    'targets': TARGETS,
    'nn_performance': {'r2': r2, 'rmse': rmse},
    'data_availability': df.groupby('Target')['CR'].count().to_dict(),
    'pca_subset': top_targets_pca if not pca_data.empty else None
}

with open(STATS_DIR / 'master_statistics.json', 'w') as f:
    json.dump(master_stats, f, indent=4)

print(f"\n{'='*80}")
print("ANALYSIS COMPLETE")
print(f"  Results saved to {OUTPUT_DIR}")
print(f"{'='*80}")

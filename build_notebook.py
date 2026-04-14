"""build_notebook.py  —  產生 walmart_forecast_colab.ipynb"""
import json, textwrap

def cell(source, cell_type="code"):
    base = {"cell_type": cell_type, "metadata": {}, "source": source}
    if cell_type == "code":
        base.update({"execution_count": None, "outputs": []})
    return base

def md(text):
    return cell(text.strip(), "markdown")

def code(text):
    return cell(textwrap.dedent(text).strip())

cells = [

# ─── 封面 ────────────────────────────────────────────────────────
md("""
# 🛒 Walmart 機率性分位數需求預測系統
**Probabilistic Demand Forecasting via Gradient Boosting Quantile Regression**

國立金門大學 工業工程與管理學系 畢業專題

---
| 步驟 | 內容 | 預計時間 |
|------|------|---------|
| 0 | 環境設定 & 資料取得 | 3 分鐘 |
| 1 | Import & 資料載入 | 2 分鐘 |
| 2 | EDA 探索性分析 | 3 分鐘 |
| 3 | 特徵工程 | 2 分鐘 |
| 4 | 特徵重要性分析 | 3 分鐘 |
| 5 | XYZ 需求分群 | 2 分鐘 |
| 6 | STL+HW 基準模型訓練 | 5 分鐘 |
| 7 | LightGBM 分位數模型訓練 | 15 分鐘 |
| 8 | 模型評估 & 對照實驗 | 3 分鐘 |
| 9 | 儲存圖表 & 輸出 CSV | 2 分鐘 |

> **總計約 40 分鐘**（含模型訓練）
"""),

# ─── 步驟 0-A：Kaggle API ────────────────────────────────────────
md("---\n## 步驟 0：環境設定 & 資料取得\n### 方案 A：Kaggle API（推薦）"),
code("""
# ════════════════════════════════════════
# 步驟 0-A：Kaggle API + 資料下載 + 解壓縮
# ════════════════════════════════════════
import os, shutil, zipfile, json as _json

# 安裝必要套件
import subprocess
subprocess.run(['pip', 'install', 'kaggle', 'lightgbm',
                'statsmodels', 'xgboost', '-q'], check=False)

# 設定 kaggle.json
os.makedirs('/root/.config/kaggle', exist_ok=True)
if os.path.exists('/content/kaggle.json'):
    shutil.copy('/content/kaggle.json', '/root/.config/kaggle/kaggle.json')
    os.chmod('/root/.config/kaggle/kaggle.json', 0o600)
    with open('/root/.config/kaggle/kaggle.json') as f:
        cfg = _json.load(f)
    print(f'✅ kaggle.json 設定完成 | username: {cfg.get(\"username\")}')
else:
    raise FileNotFoundError('找不到 kaggle.json，請先上傳至 /content/')

DATA_DIR = '/content/walmart_data'
os.makedirs(DATA_DIR, exist_ok=True)
os.system(f'kaggle competitions download '
          f'-c walmart-recruiting-store-sales-forecasting -p {DATA_DIR}')

OUTER_ZIP = f'{DATA_DIR}/walmart-recruiting-store-sales-forecasting.zip'
if not os.path.exists(OUTER_ZIP):
    raise FileNotFoundError('下載失敗，請確認已加入競賽（Join the competition）')

with zipfile.ZipFile(OUTER_ZIP, 'r') as z:
    z.extractall(DATA_DIR)

for fname in os.listdir(DATA_DIR):
    if fname.endswith('.zip') and 'walmart-recruiting' not in fname:
        with zipfile.ZipFile(f'{DATA_DIR}/{fname}', 'r') as z:
            z.extractall(DATA_DIR)
        print(f'  ✅ {fname}')

print('\\n必要檔案確認：')
for need in ['train.csv', 'stores.csv', 'features.csv']:
    ok = need in os.listdir(DATA_DIR)
    print(f'  {\"✅\" if ok else \"❌\"} {need}')
"""),

# ─── 步驟 0-B：手動上傳 ────────────────────────────────────────
md("### 方案 B：手動上傳 CSV"),
code("""
# ════════════════════════════════════════
# 步驟 0-B：手動上傳 CSV（不使用 Kaggle API）
# ════════════════════════════════════════
from google.colab import files
import os, shutil

os.makedirs('/content/walmart_data', exist_ok=True)
print('請依序上傳：train.csv, stores.csv, features.csv')
uploaded = files.upload()

for fname in uploaded.keys():
    shutil.move(fname, f'/content/walmart_data/{fname}')
    print(f'✅ {fname}')
"""),

# ─── 步驟 0-C：中文字體 ──────────────────────────────────────────
md("### 中文字體設定"),
code("""
# 下載台北思源黑體（Colab 繁體中文顯示）
!wget -q -O TaipeiSansTCBeta-Regular.ttf \\
    "https://drive.google.com/uc?id=1eGAsTN1HBpJAkeVM57_C7ccp7hbgSz3_&export=download"

import matplotlib
import matplotlib.font_manager as fm
fm.fontManager.addfont('TaipeiSansTCBeta-Regular.ttf')
matplotlib.rc('font', family='Taipei Sans TC Beta')
matplotlib.rc('axes', unicode_minus=False)
print('✅ 繁體中文字體設定完成')
"""),

# ─── 步驟 1：Import ──────────────────────────────────────────────
md("---\n## 步驟 1：Import & 資料載入"),
code("""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
import os

warnings.filterwarnings('ignore')
np.random.seed(42)

# 深色主題
plt.rcParams.update({
    'figure.facecolor': '#0f1117',
    'axes.facecolor':   '#141929',
    'axes.edgecolor':   '#2a3558',
    'text.color':       '#b8bdd8',
    'xtick.color':      '#b8bdd8',
    'ytick.color':      '#b8bdd8',
    'axes.titlecolor':  'white',
    'figure.dpi':       120,
})

DATA_DIR = '/content/walmart_data'
DRIVE_DIR = '/content/drive/MyDrive/walmart_forecast'

train    = pd.read_csv(f'{DATA_DIR}/train.csv',    parse_dates=['Date'])
stores   = pd.read_csv(f'{DATA_DIR}/stores.csv')
features = pd.read_csv(f'{DATA_DIR}/features.csv', parse_dates=['Date'])

train = train.merge(stores, on='Store', how='left')
feat_cols = ['Store','Date','Temperature','Fuel_Price',
             'MarkDown1','MarkDown2','MarkDown3','MarkDown4','MarkDown5',
             'CPI','Unemployment']
train = train.merge(features[feat_cols], on=['Store','Date'], how='left')
train = train.sort_values(['Store','Dept','Date']).reset_index(drop=True)

print('='*50)
print(f'  資料集大小：{train.shape[0]:,} 筆 × {train.shape[1]} 欄')
print(f'  時間範圍：{train[\"Date\"].min().date()} ~ {train[\"Date\"].max().date()}')
print(f'  總週數：{train[\"Date\"].nunique()}')
print(f'  分店數：{train[\"Store\"].nunique()}')
print(f'  部門數：{train[\"Dept\"].nunique()}')
print(f'  時序數：{train.groupby([\"Store\",\"Dept\"]).ngroups:,}')
print(f'  節慶週比例：{train[\"IsHoliday\"].mean():.1%}')
print('='*50)
print('✅ 資料載入完成')
"""),

# ─── 步驟 2：EDA ────────────────────────────────────────────────
md("---\n## 步驟 2：EDA — 探索性資料分析"),
code("""
CYAN='#00c8e0'; AMBER='#ffb830'; RED='#ff4d6a'
GREEN='#00d68f'; BLUE='#3d7eff'

fig, axes = plt.subplots(2, 2, figsize=(18, 10))
fig.suptitle('Walmart Sales EDA — 全局概覽', fontsize=16, color='white', fontweight='bold')

# 圖1：週銷售趨勢
ax = axes[0,0]
weekly = train.groupby('Date')['Weekly_Sales'].sum() / 1e6
ax.plot(weekly.index, weekly.values, color=CYAN, lw=1.3, alpha=0.9)
ax.fill_between(weekly.index, weekly.values, alpha=0.1, color=CYAN)
hol_w = weekly[weekly.index.isin(train[train['IsHoliday']]['Date'].unique())]
ax.scatter(hol_w.index, hol_w.values, color=RED, s=25, zorder=5, label='節慶週')
ax.axvline(pd.Timestamp('2011-11-01'), color=AMBER, lw=1.3, ls='--',
           alpha=0.8, label='Train/Test 切割')
ax.set_title('全店週銷售總額趨勢', color='white', fontweight='bold')
ax.set_ylabel('百萬美元', color='#b8bdd8')
ax.legend(facecolor='#1a2035', edgecolor='#2a3558', labelcolor='white', fontsize=8)

# 圖2：節慶倍率
ax = axes[0,1]
hm  = train[train['IsHoliday']]['Weekly_Sales'].mean()
nhm = train[~train['IsHoliday']]['Weekly_Sales'].mean()
bars = ax.bar(['非節慶週','節慶週'], [nhm/1000, hm/1000],
              color=[BLUE,RED], alpha=0.85, width=0.5)
ax.set_title(f'節慶 vs 非節慶（倍率：{hm/nhm:.2f}x）', color='white', fontweight='bold')
ax.set_ylabel('平均週銷售（千元）', color='#b8bdd8')
for bar, val in zip(bars, [nhm, hm]):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3,
            f'${val/1000:.1f}K', ha='center', color='white', fontsize=11, fontweight='bold')

# 圖3：部門 Top 20
ax = axes[1,0]
dept_avg = train.groupby('Dept')['Weekly_Sales'].mean().sort_values(ascending=False).head(20)
colors_d = [RED if v > dept_avg.mean()*1.3 else CYAN for v in dept_avg.values]
ax.bar([str(d) for d in dept_avg.index], dept_avg.values/1000, color=colors_d, alpha=0.85)
ax.axhline(dept_avg.mean()/1000, color=AMBER, ls='--', lw=1.2, alpha=0.8, label='平均值')
ax.set_title('各部門平均週銷售 Top 20（千元）', color='white', fontweight='bold')
ax.set_xlabel('部門 ID', color='#b8bdd8')
ax.legend(facecolor='#1a2035', edgecolor='#2a3558', labelcolor='white', fontsize=8)
ax.tick_params(axis='x', rotation=45)

# 圖4：外部特徵相關係數
ax = axes[1,1]
feat_corr = ['Temperature','Fuel_Price','MarkDown1','CPI','Unemployment']
corr_vals = []
for f in feat_corr:
    if f in train.columns:
        v = train[['Weekly_Sales',f]].dropna().corr().iloc[0,1]
    else:
        v = 0.0
    corr_vals.append(v)
colors_c = [RED if abs(c)>=0.3 else AMBER if abs(c)>=0.1 else '#5566aa' for c in corr_vals]
ax.barh(feat_corr, corr_vals, color=colors_c, alpha=0.85)
ax.axvline(0, color='white', lw=0.8, alpha=0.4)
ax.axvline(0.3,  color=RED, lw=0.8, ls='--', alpha=0.5, label='|r|=0.3')
ax.axvline(-0.3, color=RED, lw=0.8, ls='--', alpha=0.5)
for i,(f,v) in enumerate(zip(feat_corr, corr_vals)):
    ax.text(v+0.005*(1 if v>=0 else -1), i, f'{v:.3f}',
            va='center', ha='left' if v>=0 else 'right', color='white', fontsize=9)
ax.set_title('外部特徵與銷售 Pearson 相關係數', color='white', fontweight='bold')
ax.legend(facecolor='#1a2035', edgecolor='#2a3558', labelcolor='white', fontsize=8)

plt.tight_layout()
plt.savefig('/content/fig1_eda_overview.png', dpi=120, bbox_inches='tight', facecolor='#0f1117')
plt.show()
print('✅ EDA 圖完成')
"""),

# ─── 步驟 3：特徵工程 ────────────────────────────────────────────
md("---\n## 步驟 3：特徵工程"),
code("""
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
import warnings; warnings.filterwarnings('ignore')

def build_features_full(df):
    \"\"\"建立特徵，NaN 填 0 保留全部 91 週訓練資料\"\"\"
    df = df.copy().sort_values(['Store','Dept','Date']).reset_index(drop=True)
    df['Week']    = df['Date'].dt.isocalendar().week.astype(int)
    df['Month']   = df['Date'].dt.month
    df['Year_idx']= df['Date'].dt.year - 2010
    df['Sin_week']= np.sin(2 * np.pi * df['Week'] / 52)
    df['Cos_week']= np.cos(2 * np.pi * df['Week'] / 52)
    df['IsHoliday']     = df['IsHoliday'].astype(int)
    df['Pre_holiday_1'] = df.groupby(['Store','Dept'])['IsHoliday'].shift(-1).fillna(0).astype(int)
    df['Pre_holiday_2'] = df.groupby(['Store','Dept'])['IsHoliday'].shift(-2).fillna(0).astype(int)
    df['Post_holiday']  = df.groupby(['Store','Dept'])['IsHoliday'].shift(1).fillna(0).astype(int)
    grp = df.groupby(['Store','Dept'])['Weekly_Sales']
    for lag in [1, 2, 4, 8, 52]:
        df[f'Lag_{lag}'] = grp.shift(lag).fillna(0)
    key     = df['Store'].astype(str) + '_' + df['Dept'].astype(str)
    shifted = grp.shift(1)
    for win in [4, 12, 26]:
        df[f'Roll_mean_{win}'] = shifted.groupby(key).transform(
            lambda x: x.rolling(win, min_periods=1).mean()).fillna(0)
        df[f'Roll_std_{win}']  = shifted.groupby(key).transform(
            lambda x: x.rolling(win, min_periods=1).std()).fillna(0)
    df['Size_norm'] = df['Size'] / df['Size'].max()
    df['Type_A']    = (df['Type'] == 'A').astype(int)
    df['Type_B']    = (df['Type'] == 'B').astype(int)
    for col in ['MarkDown1','MarkDown2','MarkDown3','MarkDown4','MarkDown5']:
        if col in df.columns:
            df[col] = df[col].fillna(0)
    return df

print('建立特徵（保留全部資料，NaN 填 0）...')
df_fe_full = build_features_full(train)

TEST_START = pd.Timestamp('2011-11-01')
train_mask = df_fe_full['Date'] < TEST_START
test_mask  = df_fe_full['Date'] >= TEST_START
train_weeks = df_fe_full.loc[train_mask,'Date'].nunique()
test_weeks  = df_fe_full.loc[test_mask, 'Date'].nunique()

print(f'Train: {train_mask.sum():,} 筆（{train_weeks} 週，{train_weeks/(train_weeks+test_weeks):.1%}）')
print(f'Test : {test_mask.sum():,} 筆（{test_weeks} 週，{test_weeks/(train_weeks+test_weeks):.1%}）')
print('✅ 特徵工程完成')
"""),

# ─── 步驟 4：特徵重要性 ──────────────────────────────────────────
md("---\n## 步驟 4：特徵重要性分析"),
code("""
import xgboost as xgb

ALL_FEATURES = [
    'Week','Month','Year_idx','Sin_week','Cos_week',
    'IsHoliday','Pre_holiday_1','Pre_holiday_2','Post_holiday',
    'Lag_1','Lag_2','Lag_4','Lag_8','Lag_52',
    'Roll_mean_4','Roll_std_4','Roll_mean_12','Roll_std_12',
    'Roll_mean_26','Roll_std_26',
    'Temperature','Fuel_Price','MarkDown1','CPI','Unemployment',
    'Size_norm','Type_A','Type_B',
]
TARGET = 'Weekly_Sales'

X_all = df_fe_full.loc[train_mask, ALL_FEATURES].fillna(0)
y_all = df_fe_full.loc[train_mask, TARGET]

sample_idx = np.random.choice(len(X_all), min(15000, len(X_all)), replace=False)
xgb_model = xgb.XGBRegressor(n_estimators=200, max_depth=5, learning_rate=0.1,
                               subsample=0.8, colsample_bytree=0.8,
                               random_state=42, n_jobs=-1, verbosity=0)
xgb_model.fit(X_all.iloc[sample_idx], y_all.iloc[sample_idx])

fi_df = pd.DataFrame({
    'feature': ALL_FEATURES,
    'importance': xgb_model.feature_importances_
}).sort_values('importance', ascending=False).reset_index(drop=True)

print('特徵重要性 Top 10：')
print(fi_df.head(10).to_string(index=False))

AUTO_KEEP   = fi_df[fi_df['importance'] >= 0.005]['feature'].tolist()
DOMAIN_KEEP = ['IsHoliday','Pre_holiday_1','Year_idx','Month']
FINAL_FEATURES = list(dict.fromkeys(AUTO_KEEP + DOMAIN_KEEP))
print(f'\\n最終特徵數：{len(FINAL_FEATURES)} 個')

X_train = df_fe_full.loc[train_mask, FINAL_FEATURES].fillna(0)
y_train = df_fe_full.loc[train_mask, TARGET]
X_test  = df_fe_full.loc[test_mask,  FINAL_FEATURES].fillna(0)
y_test  = df_fe_full.loc[test_mask,  TARGET]
"""),

code("""
# ── 特徵重要性視覺化 ──────────────────────────────
type_map = {}
for f in fi_df['feature']:
    if 'Roll' in f or f.startswith('Lag'):   type_map[f] = '#00c8e0'
    elif 'Holiday' in f or 'holiday' in f:  type_map[f] = '#ff4d6a'
    elif f in ['Week','Month','Year_idx','Sin_week','Cos_week']: type_map[f] = '#ffb830'
    else:                                    type_map[f] = '#3d7eff'

top20 = fi_df.head(20)
fig, ax = plt.subplots(figsize=(12, 9))
ax.barh(top20['feature'][::-1], top20['importance'][::-1],
        color=[type_map.get(f,'#3d7eff') for f in top20['feature'][::-1]], alpha=0.85)
for i, (_, row) in enumerate(top20[::-1].iterrows()):
    ax.text(row['importance']+0.003, i, f'{row[\"importance\"]:.4f}',
            va='center', color='white', fontsize=8.5)

from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#00c8e0', label='Lag/Rolling 特徵'),
    Patch(facecolor='#ff4d6a', label='節慶特徵'),
    Patch(facecolor='#ffb830', label='時間特徵'),
    Patch(facecolor='#3d7eff', label='外部/分店特徵'),
]
ax.legend(handles=legend_elements, loc='lower right',
          facecolor='#1a2035', edgecolor='#2a3558', labelcolor='white', fontsize=9)
ax.axvline(0.005, color='#ffb830', ls='--', lw=1, alpha=0.7, label='保留門檻')
ax.set_title('特徵重要性 Top 20 (XGBoost)', fontsize=13, fontweight='bold')

plt.tight_layout()
plt.savefig('/content/fig2_feature_importance.png', dpi=120, bbox_inches='tight', facecolor='#0f1117')
plt.show()
print('✅ 特徵重要性圖完成')
"""),

# ─── 步驟 5：XYZ 分群 ────────────────────────────────────────────
md("---\n## 步驟 5：XYZ 需求分群"),
code("""
# ── XYZ 分群計算 ──────────────────────────────────
cv_df = df_fe_full.groupby(['Store','Dept'])['Weekly_Sales'].agg(
    mean_sales='mean', std_sales='std', count='count'
).reset_index()
before = len(cv_df)
cv_df  = cv_df[cv_df['mean_sales'] >= 100].copy()
print(f'過濾均值 < $100 的無效時序：{before - len(cv_df)} 條')
cv_df['CV'] = cv_df['std_sales'] / cv_df['mean_sales']

Q_LOW  = cv_df['CV'].quantile(0.40)
Q_HIGH = cv_df['CV'].quantile(0.70)
cv_df['XYZ'] = 'Z'
cv_df.loc[cv_df['CV'] <= Q_LOW,  'XYZ'] = 'X'
cv_df.loc[(cv_df['CV'] > Q_LOW) & (cv_df['CV'] <= Q_HIGH), 'XYZ'] = 'Y'

print(f'\\nXYZ 門檻：X/Y={Q_LOW:.4f}，Y/Z={Q_HIGH:.4f}')
for g, n in cv_df['XYZ'].value_counts().sort_index().items():
    print(f'  {g} 類：{n:4d} 條時序 ({n/len(cv_df):.1%})')

cv_df.to_csv('/content/xyz_segmentation.csv', index=False)

# 合併回 df_fe_full
df_fe_full = df_fe_full.merge(
    cv_df[['Store','Dept','CV','XYZ']], on=['Store','Dept'], how='left'
)
df_fe_full['XYZ'] = df_fe_full['XYZ'].fillna('X')
df_fe_full['CV']  = df_fe_full['CV'].fillna(0)
print('\\n✅ XYZ 分群完成')
"""),

code("""
# ── XYZ 四圖整合視覺化 ──────────────────────────────
CYAN='#00c8e0'; AMBER='#ffb830'; RED='#ff4d6a'
GREEN='#00d68f'; BLUE='#3d7eff'

fig = plt.figure(figsize=(20, 12))
fig.suptitle('XYZ 需求分群分析', fontsize=16, color='white', fontweight='bold', y=0.98)
gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.35)
XYZ_CLR = {'X': GREEN, 'Y': AMBER, 'Z': RED}

cv_clean = cv_df[cv_df['CV'].between(0, 5)].copy()
cv_zoom  = cv_clean[cv_clean['CV'] <= 2].copy()

# CV 完整分布
ax1 = fig.add_subplot(gs[0, 0])
ax1.hist(cv_clean['CV'], bins=60, color=BLUE, alpha=0.75, edgecolor='#2a3558', lw=0.5)
ax1.axvline(Q_LOW,  color=GREEN, lw=2, ls='--', label=f'X/Y: {Q_LOW:.3f}')
ax1.axvline(Q_HIGH, color=RED,   lw=2, ls='--', label=f'Y/Z: {Q_HIGH:.3f}')
ymax = ax1.get_ylim()[1]
ax1.fill_betweenx([0,ymax*.88], 0,Q_LOW,    alpha=0.08, color=GREEN)
ax1.fill_betweenx([0,ymax*.88], Q_LOW,Q_HIGH,alpha=0.08, color=AMBER)
ax1.fill_betweenx([0,ymax*.88], Q_HIGH,5,   alpha=0.08, color=RED)
ax1.text(Q_LOW/2,ymax*.75,'X類\\n平穩',ha='center',color=GREEN,fontsize=10,fontweight='bold')
ax1.text((Q_LOW+Q_HIGH)/2,ymax*.75,'Y類\\n波動',ha='center',color=AMBER,fontsize=10,fontweight='bold')
ax1.text((Q_HIGH+5)/2,ymax*.75,'Z類\\n尖峰',ha='center',color=RED,fontsize=10,fontweight='bold')
ax1.set_xlim(0,5)
ax1.legend(facecolor='#1a2035',edgecolor='#2a3558',labelcolor='white',fontsize=9)
ax1.set_title('CV 完整分布（0 ~ 5）',color='white',fontweight='bold',fontsize=11,pad=8)
ax1.set_xlabel('CV = 標準差 / 均值',color='#b8bdd8',fontsize=9)
ax1.set_ylabel('時序數量',color='#b8bdd8',fontsize=9)

# CV 放大
ax2 = fig.add_subplot(gs[0, 1])
ax2.hist(cv_zoom['CV'], bins=60, color=BLUE, alpha=0.75, edgecolor='#2a3558', lw=0.5)
ax2.axvline(Q_LOW,  color=GREEN, lw=2, ls='--', label=f'X/Y 門檻: {Q_LOW:.3f}')
ax2.axvline(Q_HIGH, color=RED,   lw=2, ls='--', label=f'Y/Z 門檻: {Q_HIGH:.3f}')
ymax2 = ax2.get_ylim()[1]
ax2.fill_betweenx([0,ymax2*.88], 0,Q_LOW,    alpha=0.08, color=GREEN)
ax2.fill_betweenx([0,ymax2*.88], Q_LOW,Q_HIGH,alpha=0.08, color=AMBER)
ax2.fill_betweenx([0,ymax2*.88], Q_HIGH,2,   alpha=0.08, color=RED)
ax2.set_xlim(0,2)
ax2.set_xticks([i*0.2 for i in range(11)])
ax2.legend(facecolor='#1a2035',edgecolor='#2a3558',labelcolor='white',fontsize=9)
ax2.set_title('CV 放大檢視（0 ~ 2）',color='white',fontweight='bold',fontsize=11,pad=8)
ax2.set_xlabel('CV = 標準差 / 均值',color='#b8bdd8',fontsize=9)

# 各群時序數量
ax3 = fig.add_subplot(gs[1, 0])
xyz_counts = cv_df['XYZ'].value_counts().sort_index()
bars = ax3.bar([f'{g} 類' for g in xyz_counts.index], xyz_counts.values,
               color=[XYZ_CLR[g] for g in xyz_counts.index], alpha=0.85, width=0.5)
for bar, (g, n) in zip(bars, xyz_counts.items()):
    ax3.text(bar.get_x()+bar.get_width()/2, bar.get_height()+15,
             f'{n}\\n({n/len(cv_df):.0%})',
             ha='center', color='white', fontsize=12, fontweight='bold')
ax3.set_ylim(0, xyz_counts.max()*1.3)
ax3.set_title('各分群時序數量', color='white', fontweight='bold', fontsize=11, pad=8)
ax3.set_ylabel('時序數（Store × Dept）', color='#b8bdd8', fontsize=9)

# 代表性時序
ax4 = fig.add_subplot(gs[1, 1])
for xyz in ['X','Y','Z']:
    sub_cv = cv_df[cv_df['XYZ']==xyz]
    med = sub_cv['mean_sales'].median()
    rep = sub_cv.iloc[(sub_cv['mean_sales']-med).abs().argsort().iloc[:1]].iloc[0]
    ts_ = df_fe_full[(df_fe_full['Store']==rep['Store'])&
                     (df_fe_full['Dept']==rep['Dept'])].sort_values('Date')
    ax4.plot(ts_['Date'], ts_['Weekly_Sales']/1000,
             color=XYZ_CLR[xyz], lw=1.5, alpha=0.9,
             label=f'{xyz}類 | Store={rep[\"Store\"]:02d} Dept={rep[\"Dept\"]:02d} CV={rep[\"CV\"]:.2f}')
    hol_ = ts_[ts_['IsHoliday']==1]
    ax4.scatter(hol_['Date'], hol_['Weekly_Sales']/1000,
                color=XYZ_CLR[xyz], s=25, zorder=5, alpha=0.9)
ax4.axvline(pd.Timestamp('2011-11-01'), color='white', lw=1, ls='--', alpha=0.5, label='Train / Test 切割')
ax4.legend(facecolor='#1a2035', edgecolor='#2a3558', labelcolor='white', fontsize=8)
ax4.set_title('X / Y / Z 各類代表性時序（點 = 節慶週）', color='white', fontweight='bold', fontsize=11, pad=8)
ax4.set_xlabel('日期', color='#b8bdd8', fontsize=9)
ax4.set_ylabel('週銷售額（千元）', color='#b8bdd8', fontsize=9)

plt.savefig('/content/fig3_xyz_analysis.png', dpi=120, bbox_inches='tight', facecolor='#0f1117')
plt.show()
print('✅ XYZ 分群圖完成')
"""),

# ─── 步驟 6：STL+HW ─────────────────────────────────────────────
md("---\n## 步驟 6：STL + Holt-Winters 基準模型訓練"),
code("""
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.seasonal import STL
from scipy.stats import norm
import warnings; warnings.filterwarnings('ignore')

def fit_hw_quantile(train_vals, n_test, quantiles=(0.1, 0.5, 0.9)):
    \"\"\"
    三層降階策略：
      Layer 1（n≥54）：STL + HW 趨勢預測（主力，成功率 91.8%）
      Layer 2（n≥26）：純 HW 趨勢版
      Layer 3（n<26） ：移動平均退化
    \"\"\"
    n_train, forecast, resid_std, method = len(train_vals), None, None, None

    if n_train >= 54:
        try:
            stl      = STL(train_vals, period=52, robust=True)
            stl_fit  = stl.fit()
            seasonal = stl_fit.seasonal
            deseason = train_vals - seasonal
            fit      = ExponentialSmoothing(deseason, trend='add', seasonal=None,
                                           initialization_method='estimated').fit(optimized=True)
            fc_de    = np.array(fit.forecast(n_test))
            last_s   = seasonal[-52:]
            forecast = fc_de + np.array([last_s[i % 52] for i in range(n_test)])
            resid_std= float(np.std(stl_fit.resid))
            method   = 'STL_HW'
        except: forecast = None

    if n_train >= 26 and forecast is None:
        try:
            fit      = ExponentialSmoothing(train_vals, trend='add', seasonal=None,
                                           initialization_method='estimated').fit(optimized=True)
            forecast = np.array(fit.forecast(n_test))
            resid_std= float(np.std(fit.resid))
            method   = 'HW_trend_only'
        except: forecast = None

    if forecast is None:
        window    = min(12, n_train) if n_train > 0 else 1
        mean_val  = float(np.mean(train_vals[-window:]))
        std_val   = float(np.std(train_vals[-window:])) or mean_val * 0.2
        forecast  = np.full(n_test, mean_val)
        resid_std = std_val
        method    = 'moving_avg'

    min_std   = float(np.mean(np.abs(train_vals))) * 0.05 if n_train > 0 else 1.0
    resid_std = max(resid_std or 0, min_std, 1.0)

    results = {}
    for q in quantiles:
        z_val    = norm.ppf(q)
        fc_q     = forecast + z_val * resid_std
        results[f'P{int(q*100)}_HW'] = np.maximum(fc_q, 0).tolist()

    return {
        'method': method,
        'P10_HW': results['P10_HW'],
        'P50_HW': results['P50_HW'],
        'P90_HW': results['P90_HW'],
    }

# 批次訓練
print('STL+HW 批次訓練中...')
hw_results = {}
success = total = 0

for (store, dept), grp in df_fe_full.groupby(['Store', 'Dept']):
    total += 1
    grp   = grp.sort_values('Date').reset_index(drop=True)
    train_df = grp[grp['Date'] < TEST_START]
    test_df  = grp[grp['Date'] >= TEST_START]
    if len(test_df) == 0: continue
    n_test   = len(test_df)
    train_v  = train_df['Weekly_Sales'].values.astype(float)
    if len(train_v) == 0: train_v = np.array([1.0])
    try:
        res = fit_hw_quantile(train_v, n_test)
        xyz = grp['XYZ'].iloc[0] if 'XYZ' in grp.columns else 'X'
        hw_results[(store, dept)] = {
            **res,
            'dates':      test_df['Date'].tolist(),
            'actual':     test_df['Weekly_Sales'].tolist(),
            'is_holiday': test_df['IsHoliday'].tolist(),
            'xyz':        xyz,
        }
        success += 1
    except Exception as e:
        pass

    if total % 500 == 0:
        print(f'  進度: {total}  成功率: {success/total:.1%}')

print(f'\\n✅ STL+HW 訓練完成')
print(f'   成功率：{success}/{total} = {success/total:.1%}')
"""),

# ─── 步驟 7：LightGBM ────────────────────────────────────────────
md("---\n## 步驟 7：LightGBM 分位數預測模型訓練"),
code("""
import lightgbm as lgb

assert 'X_train' in dir(), '請先執行步驟 4（特徵集定義）'
print(f'Train: {len(X_train):,} 筆  Test: {len(X_test):,} 筆  特徵: {len(FINAL_FEATURES)}')

LGBM_BASE = {
    'boosting_type': 'gbdt', 'max_depth': 6, 'num_leaves': 31,
    'learning_rate': 0.05, 'subsample': 0.8, 'colsample_bytree': 0.8,
    'min_child_samples': 20, 'reg_alpha': 0.1, 'reg_lambda': 0.1,
    'random_state': 42, 'n_jobs': -1, 'verbose': -1,
}
CONV_SETTINGS = {
    'P10': {'n_estimators': 3000, 'stopping_rounds': 100, 'min_delta': 0.5},
    'P50': {'n_estimators': 5000, 'stopping_rounds': 100, 'min_delta': 0.5},
    'P90': {'n_estimators': 3000, 'stopping_rounds': 100, 'min_delta': 0.5},
}

val_size  = int(len(X_train) * 0.2)
X_tr_fit  = X_train.iloc[:-val_size]
y_tr_fit  = y_train.iloc[:-val_size]
X_val_fit = X_train.iloc[-val_size:]
y_val_fit = y_train.iloc[-val_size:]

lgbm_models = {}
lgbm_preds  = {}

print('\\n開始訓練 LightGBM 分位數模型（min_delta=0.5 收斂控制）...')
for q, name in [(0.1,'P10'), (0.5,'P50'), (0.9,'P90')]:
    cfg   = CONV_SETTINGS[name]
    print(f'\\n  訓練 {name} (quantile={q})  上限={cfg[\"n_estimators\"]} 輪')
    model = lgb.LGBMRegressor(
        **LGBM_BASE,
        n_estimators=cfg['n_estimators'],
        objective='quantile', alpha=q
    )
    model.fit(
        X_tr_fit, y_tr_fit,
        eval_set=[(X_val_fit, y_val_fit)],
        callbacks=[
            lgb.early_stopping(
                stopping_rounds=cfg['stopping_rounds'],
                min_delta=cfg['min_delta'],
                verbose=False),
            lgb.log_evaluation(period=500),
        ]
    )
    lgbm_models[name] = model
    lgbm_preds[name]  = model.predict(X_test)

    best = model.best_iteration_
    converged = best < cfg['n_estimators'] - 1
    evals = model.evals_result_
    scores = evals.get('valid_0', {}).get('quantile', [])
    if len(scores) >= 500:
        avg_imp = (scores[-500] - scores[-1]) / 500
        status  = '✅ 已收斂' if converged else '⚠️ 仍達上限'
        print(f'    Best iteration: {best}  {status}')
        print(f'    每輪平均改善: {avg_imp:.4f}（門檻: 0.5/輪）')
    else:
        print(f'    Best iteration: {best}  {\"✅ 已收斂\" if converged else \"⚠️ 仍達上限\"}')

print('\\n✅ LightGBM 訓練完成')
"""),

# ─── 步驟 8：模型評估 ────────────────────────────────────────────
md("---\n## 步驟 8：模型評估 & 對照實驗"),
code("""
from sklearn.metrics import mean_squared_error

# ── 確保 XYZ 欄位存在 ──────────────────────────────
for col in ['CV_x','CV_y','XYZ_x','XYZ_y']:
    if col in df_fe_full.columns:
        df_fe_full = df_fe_full.drop(columns=[col])
if 'XYZ' not in df_fe_full.columns:
    df_fe_full = df_fe_full.merge(
        cv_df[['Store','Dept','CV','XYZ']], on=['Store','Dept'], how='left')
    df_fe_full['XYZ'] = df_fe_full['XYZ'].fillna('X')
    df_fe_full['CV']  = df_fe_full['CV'].fillna(0)

# ── 評估指標函式 ──────────────────────────────────
def safe_mape(y_true, y_pred, thr=100):
    mask = np.asarray(y_true) >= thr
    if mask.sum() == 0: return np.nan
    return float(np.mean(np.abs((y_true[mask]-y_pred[mask])/y_true[mask])))

def pinball_loss(y_true, y_pred, q):
    e = np.asarray(y_true) - np.asarray(y_pred)
    return float(np.mean(np.where(e>=0, q*e, (q-1)*e)))

def coverage_rate(y_true, lo, hi):
    return float(np.mean((np.asarray(y_true)>=np.asarray(lo))&
                          (np.asarray(y_true)<=np.asarray(hi))))

# ── 彙整預測結果 ──────────────────────────────────
test_df = df_fe_full[df_fe_full['Date']>=TEST_START].copy().reset_index(drop=True)
test_df['P10_LGBM'] = lgbm_preds['P10']
test_df['P50_LGBM'] = lgbm_preds['P50']
test_df['P90_LGBM'] = lgbm_preds['P90']

hw_rows = []
for (store, dept), res in hw_results.items():
    for i in range(len(res['dates'])):
        hw_rows.append({'Store':store,'Dept':dept,
                        'Date':pd.Timestamp(res['dates'][i]),
                        'P10_HW':res['P10_HW'][i],
                        'P50_HW':res['P50_HW'][i],
                        'P90_HW':res['P90_HW'][i]})
hw_df = pd.DataFrame(hw_rows)
predictions_df = test_df.merge(hw_df, on=['Store','Dept','Date'], how='left')

# ── 2×3 對照實驗矩陣 ──────────────────────────────
print('='*74)
print('  對照實驗結果：STL+HW vs LightGBM-Q  ×  X / Y / Z 分群')
print('  （MAPE 已過濾週銷售 < $100 的異常週；Coverage = P80 區間覆蓋率）')
print('='*74)
print(f'  {\"分群\":<5} {\"模型\":<14}  {\"MAPE\":>8} {\"RMSE\":>10} {\"Pinball50\":>11} {\"Coverage\":>10} {\"N\":>7}')
print('-'*74)

eval_results = []
for xyz in ['X','Y','Z','All']:
    mask = (pd.Series([True]*len(predictions_df), index=predictions_df.index)
            if xyz=='All' else predictions_df['XYZ']==xyz)
    sub  = predictions_df[mask].dropna(
        subset=['P10_HW','P50_HW','P90_HW','P10_LGBM','P50_LGBM','P90_LGBM'])
    if len(sub) == 0: continue
    y = sub['Weekly_Sales'].values
    for mname, p10c, p50c, p90c in [
        ('STL+HW',    'P10_HW',  'P50_HW',  'P90_HW'),
        ('LightGBM-Q','P10_LGBM','P50_LGBM','P90_LGBM'),
    ]:
        mape = safe_mape(y, sub[p50c].values)
        rmse = np.sqrt(mean_squared_error(y, sub[p50c].values))
        pb50 = pinball_loss(y, sub[p50c].values, 0.5)
        cov  = coverage_rate(y, sub[p10c].values, sub[p90c].values)
        marker = ' ★' if mname=='LightGBM-Q' else '  '
        mstr   = f'{mape:.4f}' if mape==mape else '  N/A  '
        print(f'  {xyz:<5} {mname:<14}{marker}{mstr:>8} {rmse:>10.1f} {pb50:>11.1f} {cov:>9.1%} {len(sub):>7}')
        eval_results.append({'XYZ':xyz,'Model':mname,'MAPE':round(mape,4) if mape==mape else None,
                             'RMSE':round(rmse,1),'Pinball_P50':round(pb50,1),
                             'Coverage_P80':round(cov,4),'N':len(sub)})
    print()

eval_df = pd.DataFrame(eval_results)
eval_df.to_csv('/content/eval_results.csv', index=False)

# ── 勝負統計 ──────────────────────────────────────
print('='*74)
print('  勝負統計（LightGBM-Q vs STL+HW，以 X/Y/Z 三分群為準）')
print('='*74)
xyz3 = eval_df[eval_df['XYZ'].isin(['X','Y','Z'])].copy()
wins = {'MAPE':0,'RMSE':0,'Pinball_P50':0,'Coverage_P80':0}
for xyz in ['X','Y','Z']:
    hw   = xyz3[(xyz3['XYZ']==xyz)&(xyz3['Model']=='STL+HW')].iloc[0]
    lgbm = xyz3[(xyz3['XYZ']==xyz)&(xyz3['Model']=='LightGBM-Q')].iloc[0]
    for m in ['MAPE','RMSE','Pinball_P50']:
        if lgbm[m] is not None and hw[m] is not None and lgbm[m] < hw[m]: wins[m]+=1
    if lgbm['Coverage_P80'] > hw['Coverage_P80']: wins['Coverage_P80']+=1
for m,w in wins.items():
    d = '↑越高越好' if m=='Coverage_P80' else '↓越低越好'
    print(f'  {m:<16} LightGBM-Q 勝出 {w}/3 個分群  {d}')
print('\\n✅ 評估完成，結果已儲存 /content/eval_results.csv')
"""),

code("""
# ── 模型對照視覺化 ────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.patch.set_facecolor('#0f1117')
fig.suptitle('預測模型對照實驗結果（STL+HW vs LightGBM-Q）',
             fontsize=14, color='white', fontweight='bold')

metrics_plot = [
    ('MAPE',        'MAPE（越低越好）',         axes[0]),
    ('Pinball_P50', 'Pinball Loss（越低越好）',  axes[1]),
    ('Coverage_P80','P80 覆蓋率（越高越好）',   axes[2]),
]
xyz_groups = ['X','Y','Z']
x_pos = np.arange(3); bar_w = 0.35

for metric, title, ax in metrics_plot:
    ax.set_facecolor('#141929')
    for sp in ax.spines.values(): sp.set_color('#2a3558')
    ax.tick_params(colors='#b8bdd8')
    hw_v   = [eval_df[(eval_df['Model']=='STL+HW')    &(eval_df['XYZ']==g)][metric].values[0]
              if len(eval_df[(eval_df['Model']=='STL+HW')    &(eval_df['XYZ']==g)])>0 else 0
              for g in xyz_groups]
    lgbm_v = [eval_df[(eval_df['Model']=='LightGBM-Q')&(eval_df['XYZ']==g)][metric].values[0]
              if len(eval_df[(eval_df['Model']=='LightGBM-Q')&(eval_df['XYZ']==g)])>0 else 0
              for g in xyz_groups]
    bars_h = ax.bar(x_pos-bar_w/2, hw_v,   bar_w, label='STL+HW',     color='#3d7eff', alpha=0.82)
    bars_l = ax.bar(x_pos+bar_w/2, lgbm_v, bar_w, label='LightGBM-Q', color='#00c8e0', alpha=0.82)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f'{g} 類' for g in xyz_groups], color='#b8bdd8')
    ax.set_title(title, color='white', fontweight='bold', fontsize=12)
    ax.legend(facecolor='#1a2035', edgecolor='#2a3558', labelcolor='white', fontsize=9)
    for bars, vals in [(bars_h,hw_v),(bars_l,lgbm_v)]:
        for bar,v in zip(bars,vals):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()*1.01,
                    f'{v:.3f}', ha='center', fontsize=9, color='white')

plt.tight_layout()
plt.savefig('/content/fig4_model_comparison.png', dpi=120, bbox_inches='tight', facecolor='#0f1117')
plt.show()
print('✅ 模型對照圖完成')
"""),

code("""
# ── 機率預測帶狀分布圖 ──────────────────────────────
XYZ_CLR = {'X':'#00d68f','Y':'#ffb830','Z':'#ff4d6a'}
XYZ_LBL = {'X':'X 類（平穩需求）','Y':'Y 類（中等波動）','Z':'Z 類（節慶尖峰）'}

fig = plt.figure(figsize=(20, 15))
fig.suptitle('機率性分位數預測：P10/P50/P90 區間 vs 實際銷售額',
             fontsize=15, color='white', fontweight='bold', y=0.99)
gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.52, wspace=0.3)

model_cfgs = [
    ('STL+HW',    'P10_HW',  'P50_HW',  'P90_HW',  '#3d7eff'),
    ('LightGBM-Q','P10_LGBM','P50_LGBM','P90_LGBM','#00c8e0'),
]

for row, xyz in enumerate(['X','Y','Z']):
    sub_cv = cv_df[(cv_df['XYZ']==xyz)&(cv_df['mean_sales']>=1000)]
    if len(sub_cv)==0: sub_cv = cv_df[cv_df['XYZ']==xyz]
    med  = sub_cv['mean_sales'].median()
    rep  = sub_cv.iloc[(sub_cv['mean_sales']-med).abs().argsort().iloc[:1]].iloc[0]
    mask = ((predictions_df['Store']==rep['Store'])&(predictions_df['Dept']==rep['Dept']))
    ts_  = predictions_df[mask].sort_values('Date').reset_index(drop=True)
    if len(ts_)==0: continue
    y_   = ts_['Weekly_Sales'].values
    hol_ = ts_[ts_['IsHoliday']==1]

    for col, (mname, p10c, p50c, p90c, mclr) in enumerate(model_cfgs):
        ax = fig.add_subplot(gs[row, col])
        ax.set_facecolor('#141929')
        p10 = ts_[p10c].clip(upper=ts_[p50c]).values
        p50 = ts_[p50c].values
        p90 = ts_[p90c].clip(lower=ts_[p50c]).values
        ax.fill_between(ts_['Date'], p10/1000, p90/1000,
                        alpha=0.28, color=mclr, label='P10~P90 預測區間')
        ax.plot(ts_['Date'], p50/1000,
                color=mclr, lw=1.8, alpha=0.9, label='P50 預測中位值')
        ax.plot(ts_['Date'], y_/1000,
                color='white', lw=1.5, ls='--', alpha=0.85, label='實際銷售額')
        if len(hol_):
            ax.scatter(hol_['Date'], hol_['Weekly_Sales']/1000,
                       color='#ff4d6a', s=35, zorder=5, alpha=0.9, label='節慶週')
        cov = float(np.mean((y_>=p10)&(y_<=p90)))
        title = (f'{XYZ_LBL[xyz]}  |  {mname}\\n'
                 f'Store={rep[\"Store\"]:02d} Dept={rep[\"Dept\"]:02d}'
                 f'  P80覆蓋率={cov:.1%}')
        ax.set_title(title, color='white', fontweight='bold', fontsize=10, pad=8)
        ax.set_ylabel('週銷售額（千元）', color='#b8bdd8', fontsize=8)
        for sp in ax.spines.values(): sp.set_color('#2a3558')
        ax.tick_params(colors='#b8bdd8', labelsize=8)
        ax.grid(color='#1e2640', lw=0.5)
        ax.legend(facecolor='#1a2035', edgecolor='#2a3558',
                  labelcolor='white', fontsize=7.5, loc='upper left')

plt.savefig('/content/fig5_prediction_bands.png', dpi=120, bbox_inches='tight', facecolor='#0f1117')
plt.show()
print('✅ 帶狀分布圖完成')
"""),

# ─── 步驟 9：輸出 ────────────────────────────────────────────────
md("---\n## 步驟 9：輸出 predictions.csv & 儲存到 Google Drive"),
code("""
# ── 組合最終 predictions.csv ──────────────────────
output_cols = ['Store','Dept','Date','Weekly_Sales','IsHoliday','XYZ',
               'P10_HW','P50_HW','P90_HW','P10_LGBM','P50_LGBM','P90_LGBM']

for col in output_cols:
    if col not in predictions_df.columns:
        print(f'⚠️ 缺少欄位：{col}')

final_df = predictions_df[output_cols].copy()
for col in ['P10_HW','P50_HW','P90_HW','P10_LGBM','P50_LGBM','P90_LGBM']:
    final_df[col] = final_df[col].round(2)
for prefix in ['HW','LGBM']:
    final_df[f'P10_{prefix}'] = final_df[[f'P10_{prefix}',f'P50_{prefix}']].min(axis=1)
    final_df[f'P90_{prefix}'] = final_df[[f'P90_{prefix}',f'P50_{prefix}']].max(axis=1)
final_df = final_df.sort_values(['Store','Dept','Date']).reset_index(drop=True)

PRED_PATH = '/content/predictions.csv'
final_df.to_csv(PRED_PATH, index=False)
print(f'✅ predictions.csv: {len(final_df):,} 筆')
print(f'   日期：{final_df[\"Date\"].min().date()} ~ {final_df[\"Date\"].max().date()}')
print(f'   XYZ：{final_df.drop_duplicates([\"Store\",\"Dept\"])[\"XYZ\"].value_counts().to_dict()}')
"""),

code("""
# ── 儲存到 Google Drive ──────────────────────────
from google.colab import drive
import shutil, pickle
drive.mount('/content/drive')

os.makedirs(DRIVE_DIR, exist_ok=True)

# CSV 檔案
for fname in ['predictions.csv','eval_results.csv','xyz_segmentation.csv']:
    src = f'/content/{fname}'
    if os.path.exists(src):
        shutil.copy(src, f'{DRIVE_DIR}/{fname}')
        print(f'  ✅ {fname}')

# 圖表
for png in ['fig1_eda_overview','fig2_feature_importance','fig3_xyz_analysis',
            'fig4_model_comparison','fig5_prediction_bands']:
    src = f'/content/{png}.png'
    if os.path.exists(src):
        shutil.copy(src, f'{DRIVE_DIR}/{png}.png')
        print(f'  ✅ {png}.png')

# LightGBM 模型
model_pkg = {
    'lgbm_models':    lgbm_models,
    'feature_names':  FINAL_FEATURES,
    'xyz_thresholds': {'Q_LOW': Q_LOW, 'Q_HIGH': Q_HIGH},
    'eval_results':   eval_df.to_dict('records'),
}
with open(f'{DRIVE_DIR}/model_package.pkl','wb') as f:
    pickle.dump(model_pkg, f)

print(f'\\n✅ 全部儲存至 Google Drive: {DRIVE_DIR}')
"""),

md("---\n## ✅ 執行完成！\n\n| 產出檔案 | 說明 |\n|---------|------|\n| `predictions.csv` | P10/P50/P90 預測（放入 `data/` 供 Streamlit 使用） |\n| `eval_results.csv` | 2×3 對照實驗指標 |\n| `xyz_segmentation.csv` | XYZ 分群結果 |\n| `model_package.pkl` | 訓練好的 LightGBM 模型 |\n| `fig1~fig5.png` | 研究圖表 |\n\n> 把 `predictions.csv` 上傳至 GitHub repo 的 `data/` 資料夾後，Streamlit 即可載入真實資料。"),
]

nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10.0"},
        "colab": {"provenance": []},
    },
    "cells": cells,
}

with open("walmart_forecast_colab.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print(f"✅ walmart_forecast_colab.ipynb 完成，共 {len(cells)} 個 cells")

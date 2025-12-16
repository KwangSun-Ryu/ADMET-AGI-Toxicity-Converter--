# ============================================
# 안정화 버전: LightGBM / GBDT(sklearn) / CatBoost
# - 모델별 SelectFromModel(importance median)로 서로 다른 피처셋
# - 100회 반복 AUC 전부 기록(누락 방지)
# - 반복별 Feature Importance 저장
# - (옵션) Top-10 피처 빈도(%) 히트맵
# ============================================
import os, glob, numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import GradientBoostingClassifier
import lightgbm as lgb
from catboost import CatBoostClassifier, Pool

# (옵션) 시각화
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------
# 0) 경로 & 출력 폴더
# -----------------------------
DATA_PATH = r"C:\Users\nicep\project\Canonicalized_Tox21_with_rdkit_descriptors.csv"
OUT_MODELS, OUT_RESULTS = "models", "results"
os.makedirs(OUT_MODELS, exist_ok=True); os.makedirs(OUT_RESULTS, exist_ok=True)

# -----------------------------
# 1) 데이터 로드 & 피처/타깃 분리
# -----------------------------
df = pd.read_csv(DATA_PATH)

drop_cols = ["ASSAY_NAME","LABEL","SMILES","Can_SMILES"]
X = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")
X = X.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
X = X.dropna(axis=1, how="all")
y = pd.to_numeric(df["LABEL"], errors="coerce")

# LABEL 결측 제거
if y.isna().any():
    valid_idx = ~y.isna()
    X = X.loc[valid_idx]
    y = y.loc[valid_idx]

# (안정화 A) 극단값/inf 처리 + winsorize(1~99%) → 폭주/오버플로 방지
X = X.mask(np.abs(X) > 1e10)  # 너무 큰 절대값은 NaN
low = X.quantile(0.01, numeric_only=True)
high = X.quantile(0.99, numeric_only=True)
X = X.clip(lower=low, upper=high, axis=1)

# -----------------------------
# 2) 모델 팩토리
# -----------------------------
def get_model(name, seed):
    if name == "LightGBM":
        return lgb.LGBMClassifier(
            n_estimators=300, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            max_depth=-1, random_state=seed, n_jobs=-1
        )
    if name == "GBDT":  # sklearn GradientBoosting
        return GradientBoostingClassifier(
            n_estimators=300, learning_rate=0.05,
            max_depth=3, subsample=0.8,  # stochastic GBDT로 일반화/안정성↑
            random_state=seed
        )
    if name == "CatBoost":
        return CatBoostClassifier(
            iterations=300, depth=6, learning_rate=0.05,
            loss_function="Logloss", verbose=False, random_state=seed
        )
    raise ValueError(name)

models = ["LightGBM", "GBDT", "CatBoost"]

# -----------------------------
# 3) 100회 반복 - 모델별 SelectFromModel + 학습/평가 + 저장
# -----------------------------
auc_records = []
imputer_strategy = "median"

for run in range(100):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=0.3, random_state=run
    )

    imputer = SimpleImputer(strategy=imputer_strategy)
    X_train = pd.DataFrame(imputer.fit_transform(X_train), columns=X.columns, index=X_train.index)
    X_test  = pd.DataFrame(imputer.transform(X_test), columns=X.columns, index=X_test.index)

    X_train = X_train.replace([np.inf, -np.inf], np.nan)
    X_test  = X_test.replace([np.inf, -np.inf], np.nan)
    if X_train.isna().any().any():
        X_train = pd.DataFrame(SimpleImputer(strategy="median").fit_transform(X_train),
                               columns=X_train.columns, index=X_train.index)
    if X_test.isna().any().any():
        X_test = pd.DataFrame(SimpleImputer(strategy="median").fit_transform(X_test),
                              columns=X_test.columns, index=X_test.index)

    for name in models:
        try:
            selector_est = get_model(name, seed=run)
            train_est = get_model(name, seed=run)

            selector_est.fit(X_train, y_train)

            if hasattr(selector_est, "feature_importances_"):
                base_importance = selector_est.feature_importances_
            elif hasattr(selector_est, "coef_"):
                base_importance = np.abs(selector_est.coef_[0])
            else:
                base_importance = np.ones(X_train.shape[1])

            selector = SelectFromModel(selector_est, threshold="median", prefit=True)
            X_train_sel = selector.transform(X_train)
            X_test_sel  = selector.transform(X_test)
            mask = selector.get_support()
            selected_features = X.columns[mask]

            X_train_sel = pd.DataFrame(X_train_sel, columns=selected_features, index=X_train.index)
            X_test_sel  = pd.DataFrame(X_test_sel,  columns=selected_features, index=X_test.index)

            if X_train_sel.shape[1] == 0:
                bi = np.asarray(base_importance, dtype=float)
                if np.all((~np.isfinite(bi)) | (bi == 0)):
                    # 모두 0/NaN이면 표준편차 최대 컬럼 선택
                    stds = X_train.to_numpy().std(axis=0)
                    top_idx = int(np.nanargmax(stds))
                else:
                    top_idx = int(np.nanargmax(bi))
                selected_features = X.columns[[top_idx]]
                X_train_sel = pd.DataFrame(X_train.iloc[:, [top_idx]].to_numpy(),
                                           columns=selected_features, index=X_train.index)
                X_test_sel  = pd.DataFrame(X_test.iloc[:, [top_idx]].to_numpy(),
                                           columns=selected_features, index=X_test.index)

     
            train_est.fit(X_train_sel, y_train)

  
            if hasattr(train_est, "predict_proba"):
                proba = train_est.predict_proba(X_test_sel)
                if proba.ndim == 2 and proba.shape[1] >= 2:
                    y_prob = proba[:, 1]
                else:
                    y_prob = np.zeros(len(y_test), dtype=float)
            elif hasattr(train_est, "decision_function"):
                scores = train_est.decision_function(X_test_sel)
                y_prob = (scores - scores.min()) / (scores.max() - scores.min() + 1e-12)
            else:
                y_prob = train_est.predict(X_test_sel).astype(float)

            auc = np.nan if len(np.unique(y_test)) < 2 else roc_auc_score(y_test, y_prob)

            auc_records.append({
                "run": run, "model": name, "auc": auc, "n_features": len(selected_features)
            })

            if name == "CatBoost":
                pool = Pool(X_train_sel, y_train, feature_names=list(selected_features))
                fi_vals = train_est.get_feature_importance(pool)
            elif hasattr(train_est, "feature_importances_"):
                fi_vals = train_est.feature_importances_
            elif hasattr(train_est, "coef_"):
                fi_vals = np.abs(train_est.coef_[0])
            else:
                fi_vals = np.ones(len(selected_features))

            fi_df = pd.DataFrame({
                "run": run, "model": name,
                "feature": selected_features, "importance": fi_vals
            }).sort_values("importance", ascending=False)
            fi_df.to_csv(os.path.join(OUT_RESULTS, f"feature_importance_{name}_run{run}.csv"),
                         index=False)

            print(f"[run {run:03d}] {name}: AUC={auc if pd.notna(auc) else np.nan:.4f} | "
                  f"n_feat={len(selected_features)} | FI saved.")

        except Exception as e:
            auc_records.append({"run": run, "model": name, "auc": np.nan, "n_features": 0})
            print(f"[run {run:03d}] {name}: ERROR -> {e}")

# -----------------------------
# 4) AUC 저장 + 요약 통계
# -----------------------------
auc_df = pd.DataFrame(auc_records)
auc_df.to_csv(os.path.join(OUT_RESULTS, "auc_by_run.csv"), index=False)

summary = (auc_df.groupby("model")["auc"]
           .agg(auc_mean=lambda s: np.nanmean(s),
                auc_std=lambda s: np.nanstd(s),
                auc_min=lambda s: np.nanmin(s) if (~s.isna()).any() else np.nan,
                auc_median=lambda s: np.nanmedian(s),
                auc_max=lambda s: np.nanmax(s) if (~s.isna()).any() else np.nan)
           .reset_index())
summary.to_csv(os.path.join(OUT_RESULTS, "auc_summary.csv"), index=False)
print("\n=== AUC Summary (100 runs) ===")
print(summary.round(4))

# -----------------------------
# 5) (옵션) 모델별 Top-10 피처 '빈도(%)' 히트맵
# -----------------------------
fi_files = []
for name in models:
    fi_files += glob.glob(os.path.join(OUT_RESULTS, f"feature_importance_{name}_run*.csv"))

if fi_files:
    dfs = []
    for fp in fi_files:
        df_fi = pd.read_csv(fp)  # run, model, feature, importance
        df_fi["rank"] = df_fi.groupby(["run","model"])["importance"]\
                             .rank(ascending=False, method="first")
        dfs.append(df_fi)
    fi_all = pd.concat(dfs, ignore_index=True)

    topk = fi_all[fi_all["rank"] <= 10].copy()
    freq = topk.groupby(["feature","model"]).size().unstack(fill_value=0)
    freq = freq.reindex(columns=models, fill_value=0)

    iters = auc_df.groupby("model")["run"].nunique().reindex(models).fillna(0)
    pct = freq.div(iters.replace(0, np.nan), axis=1) * 100
    pct = pct.fillna(0).round(1)
    pct = pct.loc[pct.mean(1).sort_values(ascending=False).index]

    plt.figure(figsize=(12, max(8, 0.35*len(pct))))
    ax = sns.heatmap(pct, cmap='Blues', vmin=0, vmax=100,
                     linewidths=.4, linecolor='black', cbar=True)
    cbar = ax.collections[0].colorbar
    cbar.set_ticks([0, 20, 40, 60, 80, 100])

    ax.set_title("Top-10 Feature Frequency (%) by Model (100 runs)", pad=12)
    ax.set_xlabel("Model"); ax.set_ylabel("Feature")
    ax.set_xticklabels(models, rotation=0)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    plt.tight_layout(); plt.show()
else:
    print("no FI")

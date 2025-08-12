import pandas as pd

df = pd.read_csv("eclsk.csv")

cols = [
    "CHILDID", 
    "avg_MIRT", 
    "P1HSEVER",
    "GENDER", "WKWHITE", "WKSESL", "S2KPUPRI", "apprchT1",
    "P1FSTAMP", "ONEPARENT", "WKCAREPK", "P1HSCALE", "P1SADLON"
]

df_final = df[cols].copy().dropna()

df_final.to_csv("final.csv", index=False)
print(f"保存成功，共 {df_final.shape[0]} 行 × {df_final.shape[1]} 列，文件名：final.csv")


import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

df = pd.read_csv("final.csv")

y = df["avg_MIRT"].values
T = df["P1HSEVER"].values
X = df[["GENDER", "WKWHITE", "WKSESL", "S2KPUPRI", "apprchT1",
        "P1FSTAMP", "ONEPARENT", "WKCAREPK", "P1HSCALE", "P1SADLON"]].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

ps_model = LogisticRegression(solver='liblinear')
ps_model.fit(X_scaled, T)
propensity_scores = ps_model.predict_proba(X_scaled)[:, 1]

treated_idx = np.where(T == 1)[0]
control_idx = np.where(T == 0)[0]

nn = NearestNeighbors(n_neighbors=1)
nn.fit(propensity_scores[control_idx].reshape(-1, 1))
distances, matched_control_idx = nn.kneighbors(propensity_scores[treated_idx].reshape(-1, 1))

treated_y = y[treated_idx]
matched_y = y[control_idx][matched_control_idx.flatten()]

ate_psm = np.mean(treated_y - matched_y)
print(f"PSM 匹配后的 ATE：{ate_psm:.4f}")

X_matched = X_scaled[treated_idx]
reg = LinearRegression().fit(X_matched, treated_y - matched_y)
adj_effect = reg.intercept_
print(f"PSM + 回归调整后的 ATE：{adj_effect:.4f}")


import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from bartpy.sklearnmodel import SklearnModel
import matplotlib.pyplot as plt

df = pd.read_csv("final.csv")

y = df["avg_MIRT"].values                    
T = df["P1HSEVER"].values             
X = df.drop(columns=["CHILDID", "avg_MIRT", "P1HSEVER"]).values 
child_ids = df["CHILDID"].values
ses = df["WKSESL"].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model_treated = SklearnModel()
model_control = SklearnModel()

model_treated.fit(X_scaled[T == 1], y[T == 1])
model_control.fit(X_scaled[T == 0], y[T == 0])

mu1 = model_treated.predict(X_scaled)
mu0 = model_control.predict(X_scaled)
ite = mu1 - mu0
ate = np.mean(ite)
print(f"BART 估计的 ATE：{ate:.4f}")


df["ITE"] = ite
df["SES"] = ses
df["SES_group"] = pd.qcut(df["SES"], q=3, labels=["Low", "Medium", "High"])
grouped = df.groupby("SES_group")["ITE"].mean()
print("\n按 SES 分组的 CATE（BART）：")
print(grouped)

ite_df = df[["CHILDID", "avg_MIRT", "P1HSEVER", "ITE"]]
ite_df.to_csv("ite_results.csv", index=False)
print("\nITE 结果已保存为 ite_results.csv ")


import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from econml.grf import CausalForest

df = pd.read_csv("final.csv")

y = df["avg_MIRT"].values
T = df["P1HSEVER"].values
X = df[["GENDER", "WKWHITE", "WKSESL", "S2KPUPRI", "apprchT1",
        "P1FSTAMP", "ONEPARENT", "WKCAREPK", "P1HSCALE", "P1SADLON"]]
child_ids = df["CHILDID"].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = CausalForest(
    n_estimators=500,
    min_samples_leaf=50,
    min_samples_split=100,
    max_depth=5,
    min_weight_fraction_leaf=0.1,
    random_state=42,
    honest=False
)

model.fit(X=X_scaled, y=y, T=T)

ite = model.predict(X_scaled)

ate = np.mean(ite)
print(f"Causal Forest 估计的 ATE：{ate:.4f}")

df["ITE_CF"] = ite

df["SES"] = df["WKSESL"]
df["SES_group"] = pd.qcut(df["SES"], q=3, labels=["Low", "Medium", "High"])
cate_grouped = df.groupby("SES_group")["ITE_CF"].mean()
print("\n按 SES 分组的 CATE（Causal Forest）：")
print(cate_grouped)

importances = model.feature_importances_
print("\n特征重要性（feature importances）：")
for name, val in zip(X.columns, importances):
    print(f"{name}: {val:.4f}")

df.to_csv("causal_forest_results.csv", index=False)
print("\n结果保存为 causal_forest_results.csv ")


import pandas as pd


bart_df = pd.read_csv("ite_results.csv")
bart_df = bart_df[["CHILDID", "ITE"]].rename(columns={"ITE": "ITE_BART"})

cf_df = pd.read_csv("causal_forest_results.csv") 
cf_df = cf_df[["CHILDID", "ITE_CF"]].rename(columns={"ITE_CF": "ITE_CF"})

merged_df = pd.merge(bart_df, cf_df, on="CHILDID", how="inner")

merged_df.to_csv("ite_comparison_bart_vs_cf.csv", index=False)

print("成功生成 ITE 对比文件：ite_comparison_bart_vs_cf.csv")


import pandas as pd
import numpy as np
import random
from sklearn.preprocessing import StandardScaler
from bartpy.sklearnmodel import SklearnModel
import matplotlib.pyplot as plt
import seaborn as sns

random.seed(42)
np.random.seed(42)

df = pd.read_csv("final.csv")
y = df["avg_MIRT"].values
T = df["P1HSEVER"].values
X = df.drop(columns=["CHILDID", "avg_MIRT", "P1HSEVER"])
child_ids = df["CHILDID"].values
ses = df["WKSESL"].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model_treated = SklearnModel()
model_control = SklearnModel()

model_treated.fit(X_scaled[T == 1], y[T == 1])
model_control.fit(X_scaled[T == 0], y[T == 0])

mu1 = model_treated.predict(X_scaled)
mu0 = model_control.predict(X_scaled)
ite = mu1 - mu0
ate = np.mean(ite)

ses_group = pd.qcut(ses, q=3, labels=["Low", "Medium", "High"])
results_df = pd.DataFrame({
    "CHILDID": child_ids,
    "avg_MIRT": y,
    "P1HSEVER": T,
    "ITE": ite,
    "WKSESL": ses,
    "SES_group": ses_group
})

cate_by_ses = results_df.groupby("SES_group")["ITE"].mean()
cate_low = cate_by_ses["Low"]
cate_med = cate_by_ses["Medium"]
cate_high = cate_by_ses["High"]

print("\n📊 贝叶斯加法回归树（BART）结果汇总：")
print(f"BART ATE：{ate:.4f}")
print("SES 分组下的 CATE：")
print(f"  • Low SES：{cate_low:.4f}")
print(f"  • Medium SES：{cate_med:.4f}")
print(f"  • High SES：{cate_high:.4f}")

summary_df = pd.DataFrame({
    "Metric": ["ATE", "CATE_Low", "CATE_Medium", "CATE_High"],
    "Value": [ate, cate_low, cate_med, cate_high]
})

results_df.to_csv("bart_ite_individuals.csv", index=False)
summary_df.to_csv("bart_summary_ate_cate.csv", index=False)

plt.figure(figsize=(8, 5))
plt.hist(ite, bins=30, edgecolor='black')
plt.axvline(x=ate, color='red', linestyle='--', label=f'ATE = {ate:.2f}')
plt.title("Figure 1. Distribution of Individual Treatment Effects (BART)")
plt.xlabel("ITE Value")
plt.ylabel("Frequency")
plt.legend()
plt.tight_layout()
plt.savefig("fig1_bart_ite_histogram.png")
plt.close()

plt.figure(figsize=(6, 5))
cate_by_ses.plot(kind='bar', color='skyblue', edgecolor='black')
plt.ylabel("CATE")
plt.title("Figure 2. CATE by Socioeconomic Status Group (BART)")
plt.tight_layout()
plt.savefig("fig2_bart_cate_by_ses.png")
plt.close()

plt.figure(figsize=(8, 5))
sns.regplot(x=results_df["WKSESL"], y=results_df["ITE"], scatter_kws={'alpha':0.3})
plt.xlabel("Socioeconomic Status (WKSESL)")
plt.ylabel("Individual Treatment Effect (ITE)")
plt.title("Figure 3. ITE vs. SES (BART)")
plt.tight_layout()
plt.savefig("fig3_bart_ite_vs_ses.png")
plt.close()

print("\n✅ 所有结果与图像已成功生成：")
print(" - bart_ite_individuals.csv")
print(" - bart_summary_ate_cate.csv")
print(" - fig1_bart_ite_histogram.png")
print(" - fig2_bart_cate_by_ses.png")
print(" - fig3_bart_ite_vs_ses.png")





import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from econml.grf import CausalForest
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("final.csv")
y = df["avg_MIRT"].values
T = df["P1HSEVER"].values
X = df[["GENDER", "WKWHITE", "WKSESL", "S2KPUPRI", "apprchT1",
        "P1FSTAMP", "ONEPARENT", "WKCAREPK", "P1HSCALE", "P1SADLON"]]
child_ids = df["CHILDID"].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = CausalForest(
    n_estimators=500,
    min_samples_leaf=50,
    min_samples_split=100,
    max_depth=5,
    min_weight_fraction_leaf=0.1,
    random_state=42,
    honest=False
)
model.fit(X=X_scaled, y=y, T=T)

ite = model.predict(X_scaled)
ate = np.mean(ite)

df["ITE_CF"] = ite
df["SES"] = df["WKSESL"]
df["SES_group"] = pd.qcut(df["SES"], q=3, labels=["Low", "Medium", "High"])
cate_grouped = df.groupby("SES_group")["ITE_CF"].mean()

importances = model.feature_importances_
feature_names = X.columns
feature_df = pd.DataFrame({
    "Variable": feature_names,
    "Importance": importances
}).sort_values(by="Importance", ascending=False)

print("\n📊 因果森林（Causal Forest）结果汇总：")
print(f"CF ATE：{ate:.4f}")
print("SES 分组下的 CATE：")
for grp, val in cate_grouped.items():
    print(f"  • {grp} SES：{val:.4f}")

df.to_csv("causal_forest_results_small_sample.csv", index=False)

plt.figure(figsize=(8, 5))
plt.hist(df["ITE_CF"], bins=30, edgecolor='black')
plt.axvline(x=ate, color='red', linestyle='--', label=f'ATE = {ate:.2f}')
plt.title("Figure 4. Distribution of Individual Treatment Effects (Causal Forest)")
plt.xlabel("ITE Value")
plt.ylabel("Frequency")
plt.legend()
plt.tight_layout()
plt.savefig("fig4_cf_ite_histogram.png")
plt.close()

plt.figure(figsize=(6, 5))
cate_grouped.plot(kind='bar', color='lightgreen', edgecolor='black')
plt.ylabel("CATE")
plt.title("Figure 5. CATE by Socioeconomic Status Group (Causal Forest)")
plt.tight_layout()
plt.savefig("fig5_cf_cate_by_ses.png")
plt.close()

plt.figure(figsize=(8, 5))
sns.barplot(x="Importance", y="Variable", data=feature_df, palette="viridis")
plt.title("Figure 6. Feature Importances (Causal Forest)")
plt.tight_layout()
plt.savefig("fig6_cf_feature_importance.png")
plt.close()

print("\n 所有 Causal Forest 结果与图表已保存：")
print(" - causal_forest_results_small_sample.csv")
print(" - fig4_cf_ite_histogram.png")
print(" - fig5_cf_cate_by_ses.png")
print(" - fig6_cf_feature_importance.png")



import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

df = pd.read_csv("final.csv")
y = df["avg_MIRT"].values
T = df["P1HSEVER"].values
X = df[["GENDER", "WKWHITE", "WKSESL", "S2KPUPRI", "apprchT1",
        "P1FSTAMP", "ONEPARENT", "WKCAREPK", "P1HSCALE", "P1SADLON"]]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

ps_model = LogisticRegression()
ps_model.fit(X_scaled, T)
ps = ps_model.predict_proba(X_scaled)[:, 1]

treated_idx = np.where(T == 1)[0]
control_idx = np.where(T == 0)[0]

nn = NearestNeighbors(n_neighbors=1)
nn.fit(ps[control_idx].reshape(-1, 1))
distances, matched_control_idx = nn.kneighbors(ps[treated_idx].reshape(-1, 1))

treated_y = y[treated_idx]
matched_control_y = y[control_idx][matched_control_idx.flatten()]
ate_matched = np.mean(treated_y - matched_control_y)

matched_treated_X = X_scaled[treated_idx]
matched_control_X = X_scaled[control_idx][matched_control_idx.flatten()]
matched_X = np.vstack([matched_treated_X, matched_control_X])
matched_T = np.hstack([np.ones(len(treated_idx)), np.zeros(len(matched_control_idx))])
matched_Y = np.hstack([treated_y, matched_control_y])

reg = LinearRegression()
reg.fit(np.column_stack([matched_T, matched_X]), matched_Y)
ate_regression = reg.coef_[0]

print("\n📊 PSM 估计结果：")
print(f"  • 匹配后 ATE：{ate_matched:.4f}")
print(f"  • 匹配 + 回归调整 ATE：{ate_regression:.4f}")

matched_df = pd.DataFrame(matched_X, columns=X.columns)
matched_df["T"] = matched_T
matched_df["Y"] = matched_Y
matched_df.to_csv("psm_matched_results.csv", index=False)

plt.figure(figsize=(6, 5))
bars = plt.bar(["Matched", "Matched + Regression"],
               [ate_matched, ate_regression],
               color=["lightblue", "steelblue"])

plt.ylim(-1.4, 0)
plt.ylabel("Average Treatment Effect (ATE)")
plt.title("Figure 7. ATE Estimates from PSM")
plt.axhline(0, color='black', linestyle='--')

for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, height + 0.02,
             f"{height:.2f}", ha='center', va='bottom')

plt.tight_layout()
plt.savefig("fig7_psm_ate_comparison.png")
plt.close()

print("\n 已保存：psm_matched_results.csv")
print(" 图像：fig7_psm_ate_comparison.png")




"""Golub weighted voting applied to WDBC: exploration, CV, stability, comparison.

Run with:  uv run python scripts/pipeline.py
Outputs go to results/ (tables as CSV, numbers in results.json, figures as PNG).

Reference for the method: Golub et al. (1999), Science 286:531-537.
No code was reused from other sources; scikit-learn is used for CV, scaling,
logistic regression and metrics.
"""
import hashlib
import itertools
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
RES.mkdir(exist_ok=True)

SEED = 32
N_SELECTED = 10

# Integrity check: the data file must match the SHA256 recorded in data/MANIFEST.json
manifest = json.loads((ROOT / "data" / "MANIFEST.json").read_text())
sha = hashlib.sha256((ROOT / "data" / "wdbc.csv").read_bytes()).hexdigest()
if sha != manifest["sha256"]:
    raise SystemExit(f"wdbc.csv does not match MANIFEST.json (got {sha}); do not replace the supplied file.")


# Load data. malignant = 1 (positive class), benign = 0
df = pd.read_csv(ROOT / "data" / "wdbc.csv")
feature_names = [c for c in df.columns if c not in ("sample_id", "diagnosis")]
X = df[feature_names].to_numpy(dtype=float)
y = (df["diagnosis"] == "malignant").astype(int).to_numpy()
results = {"seed": SEED, "n_samples": int(len(df)), "n_features": len(feature_names),
           "n_malignant": int(y.sum()), "n_benign": int((1 - y).sum())}


# 1. Data exploration. Descriptive only, nothing here is used for selection
expl = pd.DataFrame({
    "mean_benign": df[y == 0][feature_names].mean(),
    "mean_malignant": df[y == 1][feature_names].mean(),
    "sd_all": df[feature_names].std(),
})
# Univariate AUC of each raw feature (0.5 = no link, 1 = malignant always higher)
expl["univariate_auc"] = [roc_auc_score(y, df[f]) for f in feature_names]
expl.index.name = "feature"
expl.round(4).to_csv(RES / "table_exploration.csv")

fig, ax = plt.subplots(figsize=(7, 7))
order = expl["univariate_auc"].sort_values().index
ax.barh(order, expl.loc[order, "univariate_auc"], color="#4477aa")
ax.axvline(0.5, color="k", lw=0.8)
ax.set_xlabel("Univariate AUC (malignant vs benign)")
ax.set_title("Link between each feature and the diagnosis")
plt.tight_layout(); plt.savefig(RES / "fig1_univariate_auc.png", dpi=150); plt.close()

fig, ax = plt.subplots(figsize=(7, 6))
im = ax.imshow(df[feature_names].corr().to_numpy(), cmap="coolwarm", vmin=-1, vmax=1)
ax.set_xticks(range(30)); ax.set_yticks(range(30))
ax.set_xticklabels(feature_names, rotation=90, fontsize=6)
ax.set_yticklabels(feature_names, fontsize=6)
plt.colorbar(im, label="Pearson correlation")
ax.set_title("Correlation between the 30 features")
plt.tight_layout(); plt.savefig(RES / "fig2_correlation.png", dpi=150); plt.close()


# 2. Weighted voting (Golub et al. 1999), everything estimated on the train fold
def wv_fit(X_tr, y_tr, k=N_SELECTED):
    X1, X0 = X_tr[y_tr == 1], X_tr[y_tr == 0]
    mu1, mu0 = X1.mean(0), X0.mean(0)
    # sample standard deviation (ddof=1); the assignment does not specify the convention
    s1, s0 = X1.std(0, ddof=1), X0.std(0, ddof=1)
    denom = s1 + s0
    denom[denom == 0] = np.finfo(float).eps  # guard that shouldn't be used but just in case
    w = (mu1 - mu0) / denom                    # signal-to-noise weight
    sel = np.argsort(-np.abs(w))[:k]           # ten largest |w|
    mid = (mu1 + mu0) / 2                      # decision midpoint per feature
    return sel, w[sel], mid[sel]


def wv_score(X_te, sel, w, mid):
    # V(x) = sum_g w_g (x_g - midpoint_g); malignant if V >= 0
    return ((X_te[:, sel] - mid) * w).sum(1)


# 3-5. Repeated stratified 5-fold CV (3 repeats), same splits for both models
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=SEED)
rows, selected_sets, cms = [], [], {}
for i, (tr, te) in enumerate(cv.split(X, y)):
    rep, fold = i // 5 + 1, i % 5 + 1
    # weighted voting
    sel, w, mid = wv_fit(X[tr], y[tr])
    V = wv_score(X[te], sel, w, mid)
    pred = (V >= 0).astype(int)
    selected_sets.append([feature_names[j] for j in sel])
    cm = confusion_matrix(y[te], pred, labels=[0, 1])
    cms[("WV", i)] = cm
    rows.append(dict(model="WV", repeat=rep, fold=fold,
                     auc=roc_auc_score(y[te], V),
                     balanced_accuracy=balanced_accuracy_score(y[te], pred),
                     accuracy=(pred == y[te]).mean(),
                     sensitivity=cm[1, 1] / cm[1].sum(),
                     specificity=cm[0, 0] / cm[0].sum()))
    # logistic regression, scaler fitted on the training fold only
    sc = StandardScaler().fit(X[tr])
    lr = LogisticRegression(solver="liblinear", C=1.0, max_iter=1000).fit(sc.transform(X[tr]), y[tr])
    p = lr.predict_proba(sc.transform(X[te]))[:, 1]
    pred = (p >= 0.5).astype(int)
    cm = confusion_matrix(y[te], pred, labels=[0, 1])
    cms[("LR", i)] = cm
    rows.append(dict(model="LR", repeat=rep, fold=fold,
                     auc=roc_auc_score(y[te], p),
                     balanced_accuracy=balanced_accuracy_score(y[te], pred),
                     accuracy=(pred == y[te]).mean(),
                     sensitivity=cm[1, 1] / cm[1].sum(),
                     specificity=cm[0, 0] / cm[0].sum()))

folds = pd.DataFrame(rows)
folds.round(4).to_csv(RES / "table_metrics_per_fold.csv", index=False)

metrics = ["auc", "balanced_accuracy", "accuracy", "sensitivity", "specificity"]
summary = folds.groupby("model")[metrics].agg(["mean", "std"])
summary.columns = [f"{m}_{s}" for m, s in summary.columns]
summary.round(4).to_csv(RES / "table_metrics_summary.csv")

# Example model = first split (repeat 1, fold 1): confusion matrix of both methods
example = {}
for m in ("WV", "LR"):
    cm = cms[(m, 0)]
    pd.DataFrame(cm, index=["true benign", "true malignant"],
                 columns=["pred benign", "pred malignant"]).to_csv(RES / f"table_confusion_example_{m}.csv")
    r = folds[(folds.model == m) & (folds.repeat == 1) & (folds.fold == 1)].iloc[0]
    example[m] = dict(cm=cm.tolist(), auc=float(r.auc), balanced_accuracy=float(r.balanced_accuracy))

# Paired difference WV - LR over the 15 identical splits
wv, lr_ = folds[folds.model == "WV"].reset_index(), folds[folds.model == "LR"].reset_index()
diff = {m: dict(mean=float((wv[m] - lr_[m]).mean()), sd=float((wv[m] - lr_[m]).std()),
                wv_better_in=int((wv[m] > lr_[m]).sum()), tie=int((wv[m] == lr_[m]).sum()))
        for m in ("auc", "balanced_accuracy")}


# 4. Feature selection stability
freq = pd.Series(0, index=feature_names, name="selection_count")
for s in selected_sets:
    freq[s] += 1
freq_df = pd.DataFrame({"selection_count": freq, "selection_frequency": freq / len(selected_sets)})
freq_df.sort_values("selection_count", ascending=False).to_csv(RES / "table_selection_frequency.csv")
pd.DataFrame({"split": range(1, 16), "selected_features": [", ".join(s) for s in selected_sets]}
             ).to_csv(RES / "table_selected_sets.csv", index=False)

jacc = [len(set(a) & set(b)) / len(set(a) | set(b)) for a, b in itertools.combinations(selected_sets, 2)]

fig, ax = plt.subplots(figsize=(7, 7))
o = freq_df["selection_frequency"].sort_values().index
ax.barh(o, freq_df.loc[o, "selection_frequency"], color="#cc6677")
ax.set_xlabel("Selection frequency over the 15 training folds")
ax.set_title("Weighted voting: feature selection frequency")
plt.tight_layout(); plt.savefig(RES / "fig3_selection_frequency.png", dpi=150); plt.close()

fig, axes = plt.subplots(1, 2, figsize=(7, 3.5))
for ax, m, t in zip(axes, ["auc", "balanced_accuracy"], ["AUC", "Balanced accuracy"]):
    ax.boxplot([wv[m], lr_[m]], tick_labels=["Weighted voting", "Logistic reg."])
    ax.set_title(t)
plt.tight_layout(); plt.savefig(RES / "fig4_model_comparison.png", dpi=150); plt.close()

results.update(
    example=example, mean_jaccard=float(np.mean(jacc)), min_jaccard=float(np.min(jacc)),
    max_jaccard=float(np.max(jacc)), n_always_selected=int((freq == 15).sum()),
    n_ever_selected=int((freq > 0).sum()), paired_difference=diff,
    summary=json.loads(summary.to_json()),
)
(RES / "results.json").write_text(json.dumps(results, indent=2))


# LaTeX inputs: numbers and tables used by report/report.tex (tables and main numbers)
def tx(s):
    return str(s).replace("_", r"\_")


def pm(m, k):
    return f"{summary.loc[m, f'{k}_mean']:.3f} $\\pm$ {summary.loc[m, f'{k}_std']:.3f}"


order_auc = expl["univariate_auc"].sort_values(ascending=False)
near_half = expl["univariate_auc"].sub(0.5).abs().sort_values().index[:4]
sel_names = list(freq_df.sort_values("selection_count", ascending=False).index[:int((freq == 15).sum())])
mac = {
    "seed": SEED, "nsamples": len(df), "nfeatures": len(feature_names), "nmal": int(y.sum()), "nben": int((1 - y).sum()),
    "pctmal": f"{100 * y.mean():.1f}",
    "topA": tx(order_auc.index[0]), "topAauc": f"{order_auc.iloc[0]:.3f}",
    "topB": tx(order_auc.index[1]), "topBauc": f"{order_auc.iloc[1]:.3f}",
    "topC": tx(order_auc.index[2]), "topCauc": f"{order_auc.iloc[2]:.3f}",
    "nearhalf": ", ".join(tx(f) for f in near_half),
    "exwvauc": f"{example['WV']['auc']:.3f}", "exwvba": f"{example['WV']['balanced_accuracy']:.3f}",
    "exwvfn": example["WV"]["cm"][1][0], "exwvmal": sum(example["WV"]["cm"][1]), "exwvfp": example["WV"]["cm"][0][1],
    "nalways": int((freq == 15).sum()), "nnever": len(feature_names) - int((freq > 0).sum()),
    "jaccard": f"{np.mean(jacc):.3f}", "selfeatures": ", ".join(tx(f) for f in sel_names),
    "wvbasd": f"{summary.loc['WV', 'balanced_accuracy_std']:.3f}",
    "diffauc": f"{diff['auc']['mean']:.3f}", "diffba": f"{diff['balanced_accuracy']['mean']:.3f}",
    "nwvbetter": diff["auc"]["wv_better_in"] + diff["balanced_accuracy"]["wv_better_in"],
}
for m in ("WV", "LR"):
    for k, short in [("auc", "auc"), ("balanced_accuracy", "ba"), ("accuracy", "acc"),
                     ("sensitivity", "sens"), ("specificity", "spec")]:
        mac[f"{m.lower()}{short}"] = pm(m, k)
RES_TEX = ROOT / "report" / "generated"
RES_TEX.mkdir(parents=True, exist_ok=True)
(RES_TEX / "numbers.tex").write_text("".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in mac.items()))

cmx = example
(RES_TEX / "tab_confusion.tex").write_text(
    "\\begin{tabular}{lcccc}\\toprule\n & WV pred. benign & WV pred. malignant & LR pred. benign & LR pred. malignant \\\\\\midrule\n"
    f"True benign & {cmx['WV']['cm'][0][0]} & {cmx['WV']['cm'][0][1]} & {cmx['LR']['cm'][0][0]} & {cmx['LR']['cm'][0][1]} \\\\\n"
    f"True malignant & {cmx['WV']['cm'][1][0]} & {cmx['WV']['cm'][1][1]} & {cmx['LR']['cm'][1][0]} & {cmx['LR']['cm'][1][1]} \\\\\n"
    "\\bottomrule\\end{tabular}\n")
(RES_TEX / "tab_summary.tex").write_text(
    "\\begin{tabular}{lcc}\\toprule\nMetric & Weighted voting & Logistic regression \\\\\\midrule\n"
    + "".join(f"{lab} & {mac['wv' + s]} & {mac['lr' + s]} \\\\\n" for lab, s in
              [("AUC", "auc"), ("Balanced accuracy", "ba"), ("Accuracy", "acc"),
               ("Sensitivity (malignant)", "sens"), ("Specificity (benign)", "spec")])
    + "\\bottomrule\\end{tabular}\n")
(RES_TEX / "tab_features.tex").write_text(
    "\\begin{tabular}{lrrrr}\\toprule\nFeature & Mean benign & Mean malignant & Univariate AUC & Selection freq. \\\\\\midrule\n"
    + "".join(f"\\texttt{{{tx(f)}}} & {expl.loc[f, 'mean_benign']:.4g} & {expl.loc[f, 'mean_malignant']:.4g} & "
              f"{expl.loc[f, 'univariate_auc']:.3f} & {freq_df.loc[f, 'selection_frequency']:.2f} \\\\\n" for f in feature_names)
    + "\\bottomrule\\end{tabular}\n")

print(json.dumps(results, indent=2))
print(freq_df.sort_values("selection_count", ascending=False).head(20))
print(summary.round(4).T)

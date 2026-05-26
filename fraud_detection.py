"""
=============================================================================
 Credit Card Fraud Detection — Hybrid Autoencoder + XGBoost
=============================================================================
 An end-to-end fraud detection pipeline that combines:

   1.  An Autoencoder trained ONLY on legitimate transactions to learn the
       manifold of "normal" behaviour. Reconstruction error on unseen data
       acts as an unsupervised anomaly score.

   2.  An XGBoost classifier trained on the labelled data with
       `scale_pos_weight` to handle the ~5% positive class.

   3.  A HYBRID model in which the Autoencoder's reconstruction error is
       injected as an extra engineered feature into XGBoost. Intuition:
       XGBoost can carve up the labelled feature space, but it can struggle
       with subtle out-of-distribution patterns. The AE error gives it an
       unsupervised "weirdness" signal it would not otherwise see.

 Outputs:
   - Comparison plots (PR curves, confusion matrices) for the three models.
   - SHAP summary plot showing which features (including AE error) drive
     the hybrid model's predictions.
   - A printed report of AUPRC, AUROC, precision, recall, F1.
=============================================================================
"""

# -----------------------------------------------------------------------------
# 0. Imports & reproducibility
# -----------------------------------------------------------------------------
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    precision_recall_curve, average_precision_score, roc_auc_score,
    confusion_matrix, classification_report, f1_score
)

import xgboost as xgb
import shap

import tensorflow as tf
from tensorflow.keras import layers, Model, callbacks, optimizers

warnings.filterwarnings("ignore")

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# -----------------------------------------------------------------------------
# 1. Synthetic dataset
# -----------------------------------------------------------------------------
# The brief specifies a 10,000-row dataset with the listed columns but does
# not provide one, so we generate a realistic synthetic dataset where fraud
# patterns are correlated with the features in plausible ways (e.g. fraud is
# more likely at odd hours, with foreign transactions, low device trust, and
# unusual amounts). Replace this block with `pd.read_csv(...)` if you have
# the real data.
# -----------------------------------------------------------------------------
def generate_synthetic_data(n=10_000, fraud_rate=0.05, seed=SEED):
    """
    Generate a deliberately challenging synthetic dataset. Fraud rows have
    distributions that OVERLAP heavily with legitimate ones — only a subset
    of fraud is "obvious" (clear odd-hours / foreign / low-trust signature).
    The rest is camouflaged: it looks normal on every individual feature
    but is anomalous in the joint distribution. This is the regime where
    the AE's unsupervised manifold-distance signal earns its keep.
    """
    rng = np.random.default_rng(seed)
    n_fraud = int(n * fraud_rate)
    n_legit = n - n_fraud

    # ---- Legitimate transactions ----------------------------------------
    legit = pd.DataFrame({
        "amount":              rng.lognormal(mean=3.5, sigma=0.9, size=n_legit),
        "transaction_hour":    rng.integers(0, 24, size=n_legit),
        "merchant_category":   rng.choice(
            ["grocery", "restaurant", "gas", "retail", "online", "travel"],
            size=n_legit, p=[0.25, 0.20, 0.15, 0.20, 0.15, 0.05]),
        "foreign_transaction": rng.choice([0, 1], size=n_legit, p=[0.93, 0.07]),
        "location_mismatch":   rng.choice([0, 1], size=n_legit, p=[0.95, 0.05]),
        "device_trust_score":  np.clip(rng.normal(0.75, 0.15, n_legit), 0, 1),
        "velocity_last_24h":   rng.poisson(3.0, n_legit),
        "cardholder_age":      np.clip(rng.normal(45, 14, n_legit), 18, 90).astype(int),
        "is_fraud":            0,
    })
    # Make legit hour distribution skew toward daytime by re-sampling 40%
    mask = rng.random(n_legit) < 0.4
    legit.loc[mask, "transaction_hour"] = rng.integers(8, 22, mask.sum())

    # ---- Fraud: split into "obvious" and "camouflaged" buckets ----------
    n_obvious = int(n_fraud * 0.55)
    n_camo    = n_fraud - n_obvious

    obvious = pd.DataFrame({
        "amount":              rng.lognormal(4.4, 1.2, n_obvious),
        "transaction_hour":    rng.choice(list(range(0, 5)) + list(range(22, 24)), n_obvious),
        "merchant_category":   rng.choice(
            ["grocery", "restaurant", "gas", "retail", "online", "travel"],
            size=n_obvious, p=[0.05, 0.05, 0.10, 0.15, 0.50, 0.15]),
        "foreign_transaction": rng.choice([0, 1], n_obvious, p=[0.50, 0.50]),
        "location_mismatch":   rng.choice([0, 1], n_obvious, p=[0.40, 0.60]),
        "device_trust_score":  np.clip(rng.normal(0.40, 0.20, n_obvious), 0, 1),
        "velocity_last_24h":   rng.poisson(6.5, n_obvious),
        "cardholder_age":      np.clip(rng.normal(45, 14, n_obvious), 18, 90).astype(int),
        "is_fraud":            1,
    })

    # Camouflaged fraud: looks normal on most features, only weird in
    # joint-feature space (e.g. medium amount + correct hour + trusted
    # device, but a rare 3-way combination of merchant+foreign+velocity).
    camo = pd.DataFrame({
        "amount":              rng.lognormal(3.6, 1.0, n_camo),
        "transaction_hour":    rng.integers(8, 22, n_camo),
        "merchant_category":   rng.choice(
            ["grocery", "restaurant", "gas", "retail", "online", "travel"],
            size=n_camo, p=[0.20, 0.15, 0.10, 0.20, 0.25, 0.10]),
        "foreign_transaction": rng.choice([0, 1], n_camo, p=[0.80, 0.20]),
        "location_mismatch":   rng.choice([0, 1], n_camo, p=[0.85, 0.15]),
        "device_trust_score":  np.clip(rng.normal(0.65, 0.15, n_camo), 0, 1),
        "velocity_last_24h":   rng.poisson(4.5, n_camo),
        "cardholder_age":      np.clip(rng.normal(45, 14, n_camo), 18, 90).astype(int),
        "is_fraud":            1,
    })

    df = pd.concat([legit, obvious, camo], ignore_index=True)
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    return df


# -----------------------------------------------------------------------------
# 2. Preprocessing
# -----------------------------------------------------------------------------
def preprocess(df):
    """One-hot encode categoricals, scale numerics, return X, y, and the
    fitted scaler/column list for downstream reuse."""
    y = df["is_fraud"].astype(int).values
    X = df.drop(columns=["is_fraud"])

    # One-hot encode the single categorical column
    X = pd.get_dummies(X, columns=["merchant_category"], prefix="mcat",
                       drop_first=False, dtype=float)

    feature_names = X.columns.tolist()
    return X, y, feature_names


def make_splits(X, y, feature_names):
    """
    Stratified 70/15/15 split. Returns:
      - (X_train, X_val, X_test, y_train, y_val, y_test) scaled to [0,1]-ish
      - X_train_normal: the subset of TRAINING data with y==0, used to train
        the Autoencoder unsupervised. We deliberately exclude validation and
        test data so the AE never sees them during training.
    Numeric columns are standardised using stats fit on the TRAINING set only.
    """
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=0.15, stratify=y, random_state=SEED)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=0.1765,   # 0.1765 * 0.85 ≈ 0.15
        stratify=y_trainval, random_state=SEED)

    # Scale all features. One-hot columns are already in {0,1}, so scaling
    # them is harmless and keeps the AE input on one consistent scale.
    scaler = StandardScaler()
    X_train_s = pd.DataFrame(scaler.fit_transform(X_train),
                             columns=feature_names, index=X_train.index)
    X_val_s   = pd.DataFrame(scaler.transform(X_val),
                             columns=feature_names, index=X_val.index)
    X_test_s  = pd.DataFrame(scaler.transform(X_test),
                             columns=feature_names, index=X_test.index)

    X_train_normal = X_train_s[y_train == 0].copy()

    return (X_train_s, X_val_s, X_test_s,
            y_train, y_val, y_test,
            X_train_normal, scaler)


# -----------------------------------------------------------------------------
# 3. Phase A — Autoencoder
# -----------------------------------------------------------------------------
# Architecture rationale:
#   - Symmetric encoder/decoder funnelling down to a small bottleneck. With
#     ~14 input features after one-hot encoding, a 14 → 8 → 4 → 8 → 14 shape
#     is a good default: small enough to force the model to compress, large
#     enough to capture interactions.
#   - ReLU on hidden layers, linear on the output (we are reconstructing
#     standardised continuous values, not probabilities).
#   - MSE loss directly corresponds to the per-sample reconstruction error
#     we will use as the anomaly score.
#   - EarlyStopping on validation loss prevents overfitting to the normal
#     manifold; if the AE memorises perfectly, fraud points are no longer
#     "harder" to reconstruct.
# -----------------------------------------------------------------------------
def build_autoencoder(input_dim, bottleneck=4):
    inp = layers.Input(shape=(input_dim,), name="input")
    x = layers.Dense(8, activation="relu")(inp)
    z = layers.Dense(bottleneck, activation="relu", name="bottleneck")(x)
    x = layers.Dense(8, activation="relu")(z)
    out = layers.Dense(input_dim, activation="linear", name="reconstruction")(x)

    ae = Model(inp, out, name="autoencoder")
    ae.compile(optimizer=optimizers.Adam(1e-3), loss="mse")
    return ae


def train_autoencoder(X_train_normal, X_val_s, y_val):
    """
    Train AE only on legitimate samples. We monitor val_loss on the FULL
    validation set; this is slightly unusual (val contains some fraud) but
    it matches how the AE will be used at inference: fraud should produce
    HIGH loss, so we don't want to optimise val_loss to zero — we just want
    to stop when reconstruction of normals has plateaued.
    For a stricter setup, monitor on a held-out subset of normals instead.
    """
    # Split a held-out chunk of normals for early stopping (cleaner signal
    # than mixing fraud into the AE's validation).
    val_normal_idx = (y_val == 0)
    X_val_normal = X_val_s[val_normal_idx]

    ae = build_autoencoder(X_train_normal.shape[1], bottleneck=4)
    early = callbacks.EarlyStopping(monitor="val_loss", patience=10,
                                    restore_best_weights=True)
    history = ae.fit(
        X_train_normal, X_train_normal,
        validation_data=(X_val_normal, X_val_normal),
        epochs=200, batch_size=64, shuffle=True,
        callbacks=[early], verbose=0,
    )
    print(f"  AE trained for {len(history.history['loss'])} epochs "
          f"(best val_loss={min(history.history['val_loss']):.4f})")
    return ae


def reconstruction_error(ae, X):
    """Per-sample MSE between input and reconstruction."""
    recon = ae.predict(X, verbose=0)
    return np.mean(np.square(X.values - recon), axis=1)


# -----------------------------------------------------------------------------
# 4. Phase B — XGBoost
# -----------------------------------------------------------------------------
def train_xgboost(X_train, y_train, X_val, y_val, extra_label=""):
    pos = (y_train == 1).sum()
    neg = (y_train == 0).sum()
    scale = neg / max(pos, 1)   # standard recipe for imbalance
    print(f"  XGBoost {extra_label}: scale_pos_weight = {scale:.2f}")

    model = xgb.XGBClassifier(
        n_estimators=400,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        scale_pos_weight=scale,
        eval_metric="aucpr",
        early_stopping_rounds=30,
        tree_method="hist",
        random_state=SEED,
        n_jobs=-1,
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    return model


# -----------------------------------------------------------------------------
# 5. Phase C — Hybrid integration
# -----------------------------------------------------------------------------
# The hybrid model simply concatenates the AE reconstruction error as one
# additional column ("ae_error") to the existing feature matrix and retrains
# XGBoost on the augmented matrix. Two important details:
#
#   - The AE was trained on X_train_normal (legitimate rows from the train
#     split). The error feature for the train/val/test sets is computed by
#     scoring those rows through the SAME frozen AE, so there is no leakage:
#     no test labels touched the AE, and no test rows touched XGBoost's fit.
#   - We standardise / clip the error to keep XGBoost's split-finding stable
#     on outliers (XGBoost handles them fine, but log1p smooths the tail).
# -----------------------------------------------------------------------------
def add_ae_feature(X, ae):
    err = reconstruction_error(ae, X)
    X_out = X.copy()
    X_out["ae_error"] = np.log1p(err)   # smooth the long tail
    return X_out


# -----------------------------------------------------------------------------
# 6. Evaluation helpers
# -----------------------------------------------------------------------------
def evaluate(name, y_true, scores):
    """Return a dict of metrics and the threshold that maximises F1."""
    auprc = average_precision_score(y_true, scores)
    auroc = roc_auc_score(y_true, scores)

    prec, rec, thr = precision_recall_curve(y_true, scores)
    # F1 at every threshold; pick the best one
    f1s = 2 * prec * rec / (prec + rec + 1e-12)
    best = np.nanargmax(f1s[:-1]) if len(thr) > 0 else 0
    best_thr = thr[best] if len(thr) > 0 else 0.5
    y_pred = (scores >= best_thr).astype(int)

    cm = confusion_matrix(y_true, y_pred)
    return {
        "name": name,
        "auprc": auprc, "auroc": auroc,
        "best_thr": best_thr,
        "precision": prec[best], "recall": rec[best], "f1": f1s[best],
        "cm": cm, "scores": scores, "y_pred": y_pred,
        "pr_curve": (prec, rec),
    }


def print_report(metrics):
    print(f"\n  {metrics['name']}")
    print(f"    AUPRC: {metrics['auprc']:.4f} | AUROC: {metrics['auroc']:.4f}")
    print(f"    @ best F1 threshold ({metrics['best_thr']:.4f}):")
    print(f"      precision = {metrics['precision']:.4f}")
    print(f"      recall    = {metrics['recall']:.4f}")
    print(f"      F1        = {metrics['f1']:.4f}")
    tn, fp, fn, tp = metrics['cm'].ravel()
    print(f"      TN={tn}  FP={fp}  FN={fn}  TP={tp}")


def plot_pr_curves(all_metrics, path):
    plt.figure(figsize=(7, 5.5))
    for m in all_metrics:
        p, r = m["pr_curve"]
        plt.plot(r, p, label=f"{m['name']} (AUPRC={m['auprc']:.3f})", linewidth=2)
    plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.title("Precision–Recall Curves")
    plt.legend(loc="lower left"); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(path, dpi=120); plt.close()


def plot_confusion_matrices(all_metrics, path):
    fig, axes = plt.subplots(1, len(all_metrics), figsize=(5 * len(all_metrics), 4.5))
    if len(all_metrics) == 1:
        axes = [axes]
    for ax, m in zip(axes, all_metrics):
        sns.heatmap(m["cm"], annot=True, fmt="d", cmap="Blues", cbar=False,
                    xticklabels=["Legit", "Fraud"],
                    yticklabels=["Legit", "Fraud"], ax=ax)
        ax.set_title(f"{m['name']}\nF1={m['f1']:.3f}")
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    plt.tight_layout(); plt.savefig(path, dpi=120); plt.close()


# -----------------------------------------------------------------------------
# 7. SHAP interpretability
# -----------------------------------------------------------------------------
def shap_explain(model, X_sample, path):
    """TreeSHAP on the hybrid XGBoost model. Saves a summary plot showing
    which features push predictions toward fraud."""
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    plt.figure()
    shap.summary_plot(shap_values, X_sample, show=False, max_display=15)
    plt.tight_layout()
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()


# -----------------------------------------------------------------------------
# 8. Orchestration
# -----------------------------------------------------------------------------
def main():
    print("=" * 70)
    print(" Hybrid Autoencoder + XGBoost Fraud Detection")
    print("=" * 70)

    # ---- 8.1 Data --------------------------------------------------------
    print("\n[1/6] Generating synthetic dataset (10,000 rows, ~5% fraud)...")
    df = generate_synthetic_data(n=10_000, fraud_rate=0.05)
    print(f"  Shape: {df.shape}")
    print(f"  Fraud rate: {df['is_fraud'].mean():.3%}")

    # ---- 8.2 Preprocess --------------------------------------------------
    print("\n[2/6] Preprocessing (one-hot + standardisation)...")
    X, y, feat_names = preprocess(df)
    (X_tr, X_va, X_te,
     y_tr, y_va, y_te,
     X_tr_norm, scaler) = make_splits(X, y, feat_names)
    print(f"  Train: {X_tr.shape}  ({y_tr.sum()} fraud)")
    print(f"  Val:   {X_va.shape}  ({y_va.sum()} fraud)")
    print(f"  Test:  {X_te.shape}  ({y_te.sum()} fraud)")
    print(f"  AE training pool (normals only): {X_tr_norm.shape}")

    # ---- 8.3 Phase A: Autoencoder ---------------------------------------
    print("\n[3/6] Phase A — Training Autoencoder on legitimate transactions...")
    ae = train_autoencoder(X_tr_norm, X_va, y_va)

    err_test = reconstruction_error(ae, X_te)
    ae_metrics = evaluate("Pure Autoencoder", y_te, err_test)

    # ---- 8.4 Phase B: XGBoost -------------------------------------------
    print("\n[4/6] Phase B — Training pure XGBoost on full labelled data...")
    xgb_pure = train_xgboost(X_tr, y_tr, X_va, y_va, extra_label="(pure)")
    xgb_scores = xgb_pure.predict_proba(X_te)[:, 1]
    xgb_metrics = evaluate("Pure XGBoost", y_te, xgb_scores)

    # ---- 8.5 Phase C: Hybrid --------------------------------------------
    print("\n[5/6] Phase C — Building Hybrid (XGBoost + AE error feature)...")
    X_tr_h = add_ae_feature(X_tr, ae)
    X_va_h = add_ae_feature(X_va, ae)
    X_te_h = add_ae_feature(X_te, ae)
    xgb_hybrid = train_xgboost(X_tr_h, y_tr, X_va_h, y_va, extra_label="(hybrid)")
    hybrid_scores = xgb_hybrid.predict_proba(X_te_h)[:, 1]
    hybrid_metrics = evaluate("Hybrid (XGB + AE error)", y_te, hybrid_scores)

    # ---- 8.6 Reports & plots --------------------------------------------
    print("\n[6/6] Evaluation summary")
    print("=" * 70)
    all_m = [ae_metrics, xgb_metrics, hybrid_metrics]
    for m in all_m:
        print_report(m)

    print("\n  Saving plots...")
    plot_pr_curves(all_m, f"{OUTPUT_DIR}/pr_curves.png")
    plot_confusion_matrices(all_m, f"{OUTPUT_DIR}/confusion_matrices.png")

    print("  Generating SHAP summary for hybrid model...")
    # Sample for speed; TreeSHAP is exact but plotting 1500 points is plenty.
    sample = X_te_h.sample(min(1500, len(X_te_h)), random_state=SEED)
    shap_explain(xgb_hybrid, sample, f"{OUTPUT_DIR}/shap_summary.png")

    # Feature importance (gain) as a quick text summary alongside SHAP
    booster = xgb_hybrid.get_booster()
    imp = booster.get_score(importance_type="gain")
    imp_df = pd.DataFrame(
        sorted(imp.items(), key=lambda kv: -kv[1]),
        columns=["feature", "gain"]
    )
    print("\n  Top features by XGBoost gain (hybrid model):")
    print(imp_df.head(10).to_string(index=False))

    # Comparison table
    print("\n" + "=" * 70)
    print(" Model comparison (test set)")
    print("=" * 70)
    summary = pd.DataFrame([
        {"Model": m["name"], "AUPRC": m["auprc"], "AUROC": m["auroc"],
         "Precision": m["precision"], "Recall": m["recall"], "F1": m["f1"]}
        for m in all_m
    ])
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    summary.to_csv(f"{OUTPUT_DIR}/model_comparison.csv", index=False)

    # Quick verdict
    best = summary.loc[summary["AUPRC"].idxmax(), "Model"]
    print(f"\n  Best model by AUPRC: {best}")
    lift = (hybrid_metrics["auprc"] - xgb_metrics["auprc"]) / xgb_metrics["auprc"] * 100
    print(f"  Hybrid vs. pure XGBoost AUPRC lift: {lift:+.2f}%")

    print(f"\n  All artefacts saved to ./{OUTPUT_DIR}/")
    print("=" * 70)


if __name__ == "__main__":
    main()

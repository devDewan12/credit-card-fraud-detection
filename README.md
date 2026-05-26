# Credit Card Fraud Detection — Hybrid Autoencoder + XGBoost

End-to-end fraud detection pipeline implementing the three-phase
unsupervised + supervised hybrid approach.

---

## 🔗 Run on Google Colab

Try the full pipeline directly in the browser — no local setup required:

👉 **[Open in Google Colab](https://colab.research.google.com/drive/1hc59lrx2HraiOfNbO06eoYz5K6JKrGt6?usp=sharing)**

> Tip: The script runs end-to-end in ~30–90 seconds, so a CPU runtime is
> fine. Switch to GPU only if you scale the dataset up significantly.

---

## Architecture

```
                                ┌───────────────────────┐
                  ┌────────────►│  Phase A: Autoencoder │──┐
                  │             │  (trained on legit    │  │  reconstruction
   Preprocessed   │             │   transactions only)  │  │  error
   features ──────┤             └───────────────────────┘  │
                  │                                        │
                  │             ┌───────────────────────┐  │
                  ├────────────►│  Phase B: XGBoost     │  │
                  │             │  (pure classifier)    │  │
                  │             └───────────────────────┘  │
                  │                                        ▼
                  │             ┌─────────────────────────────────┐
                  └────────────►│  Phase C: Hybrid XGBoost        │
                                │  (XGBoost + AE error feature)   │
                                └─────────────────────────────────┘
```

## Project Steps

1. **Generates 10,000 synthetic transactions** with the 8 features from the
   brief and a ~5% fraud rate. Fraud comes in two flavours: ~55% "obvious"
   (odd hours, low device trust, foreign, high velocity) and ~45%
   "camouflaged" (looks normal on every individual feature; only anomalous
   in the joint distribution). Replace `generate_synthetic_data()` with
   `pd.read_csv()` to use real data.
2. **Preprocesses**: one-hot encodes `merchant_category`, standard-scales
   everything with stats fit on the training split only.
3. **Splits**: stratified 70/15/15 train/val/test. The Autoencoder gets a
   "normal-only" view of the training set; the validation set used for AE
   early stopping is also restricted to legitimate transactions.
4. **Phase A**: 13 → 8 → 4 → 8 → 13 symmetric autoencoder, MSE loss, Adam,
   EarlyStopping with `restore_best_weights=True`. Per-sample MSE on test
   data is the unsupervised anomaly score.
5. **Phase B**: XGBoost with `scale_pos_weight = neg/pos`, AUCPR eval
   metric, early stopping at 30 rounds.
6. **Phase C**: Re-trains XGBoost on the same data with the AE's
   `log1p(reconstruction_error)` appended as a 14th feature.
7. **Evaluation**: AUPRC, AUROC, F1-optimal precision/recall/threshold,
   and confusion matrices for all three models.
8. **Interpretability**: TreeSHAP summary plot for the hybrid model plus a
   gain-based feature importance table.

## How to run

```bash
pip install pandas numpy scikit-learn xgboost tensorflow shap matplotlib seaborn
python fraud_detection.py
```

Runs end-to-end in roughly 30–90 seconds on CPU.

> Prefer a zero-install option? Use the
> [Google Colab notebook](https://colab.research.google.com/drive/1hc59lrx2HraiOfNbO06eoYz5K6JKrGt6?usp=sharing)
> instead — everything runs in the browser.

## Outputs (saved to `./outputs/`)

| File                     | Contents                                                |
| ------------------------ | ------------------------------------------------------- |
| `pr_curves.png`          | Precision–recall curves for AE, pure XGBoost, hybrid    |
| `confusion_matrices.png` | Confusion matrices at each model's F1-optimal threshold |
| `shap_summary.png`       | TreeSHAP beeswarm for the hybrid model                  |
| `model_comparison.csv`   | AUPRC / AUROC / precision / recall / F1 per model       |

## Expected results

On the synthetic data the script produces, you should see roughly:

| Model            | AUPRC     | AUROC     |
| ---------------- | --------- | --------- |
| Pure Autoencoder | ~0.54     | ~0.82     |
| Pure XGBoost     | ~0.62     | ~0.87     |
| **Hybrid**       | **~0.67** | **~0.88** |

The ~7% AUPRC lift of the hybrid over pure XGBoost is the payoff of the
"twist": the AE's reconstruction error captures camouflaged fraud that
looks normal on every individual feature but lives outside the manifold
of legitimate behaviour XGBoost is fitting from labels alone.

## Why this hybrid works (and when it doesn't)

- **AE strengths**: detects out-of-distribution patterns even for fraud
  types never seen during training; needs no labels.
- **AE weaknesses**: high false-positive rate; legitimate-but-unusual
  transactions (large legitimate purchases, foreign travel) score high.
- **XGBoost strengths**: precise where it has label signal.
- **XGBoost weaknesses**: misses novel patterns; can overfit to the
  specific fraud distribution in the training labels.
- **Hybrid**: XGBoost gets to use the AE's "weirdness" signal as one
  feature among many. It can learn to trust AE error in some regions of
  feature space and ignore it in others. If your data is dominated by one
  fraud archetype that XGBoost already separates cleanly, the hybrid lift
  will be small. The harder and more heterogeneous the fraud, the more
  the AE earns its keep.

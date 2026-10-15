# MamaDey AI

**Title:** An AI-Enabled, Community-Integrated Framework for Early Maternal Risk Prediction in Ogun State, Nigeria

**MSc Dissertation** Data Science, University of East London (2026)  
**Author:** Kanyisola Fagbayi
**Submission:** 24 May 2026

---

## Overview

Nigeria accounts for nearly one-fifth of global maternal deaths. In Ogun State, the primary drivers of maternal mortality are not the absence of services; they are **data invisibility**, disrupted continuity of care, and the structural inability of existing health information systems to act on risk before it becomes fatal.

MamaDey AI proposes a three-layer, AI-enabled framework that:

1. **Generates born-digital community-level data** through Community Health Worker (CHW) mobile tools and women's group engagement; bypassing the paper-based transcription bottleneck that currently degrades PHC data quality
2. **Applies supervised machine learning** to classify pregnant women as low, medium, or high risk; achieving AUC-ROC of 0.8444 (Random Forest) and 0.8586 (XGBoost).
3. **Delivers structured decision support** to CHWs, facility staff, and Ministry dashboards; integrated with existing systems including DHIS2, BHCPF, and the MAMII initiative

The framework is designed to complement, not replace existing Nigerian government interventions, and is aligned with the national goal of eliminating preventable maternal deaths by 2030.

---

## Project Structure

```
MamaDey_ai/
├── data/
│   ├── raw/          <- Downloaded NDHS 2018 files (NGIR7BDT.DTA etc.) 
│   ├── processed/    <- Cleaned datasets output by notebooks
│   └── synthetic/    <- Synthetically generated community-level records
├── notebooks/
│   ├── week_1_3_ndhs_eda.ipynb                 <- Data loading, cleaning, EDA
│   ├── week_4_5_ndhs_feature_engineering.ipynb <- Feature engineering + synthetic data
│   ├── week_6_7_baseline_models.ipynb          <- Logistic regression + Random Forest
│   └── week_8_10_xgboost_evaluation.ipynb      <- XGBoost, SMOTE, final evaluation
├── models/           <- Saved model objects (.pkl)
├── outputs/          <- Figures, reports, CSVs for dissertation
├── scripts/          <- Any reusable utility functions
└── README.md
```

## Research Status

| Phase                                         | Status      |
|-----------------------------------------------|-------------|
| Proposal and literature review                | Complete    |
| Data acquisition (NDHS 2018)                  | Complete    |
| Data cleaning and EDA                         | Complete    |
| Feature engineering and synthetic data        | Complete    |
| Model development (Random Forest, XGBoost, LR)| Complete    |
| SHAP interpretability analysis                | Complete    |
| Dissertation writing                          | Complete    | 
| Submission                                    | Complete (May 2026; award pending conferment) |

---

## Framework Architecture

```
┌─────────────────────────────────────────────────────────┐
│  Layer 1: Community Data Input (MamaCare)               │
│  MamaCare app · CHW visits · Women's groups (NFWP)      │
└────────────────────────┬────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────┐
│  Layer 2: AI Engine (MamaDey AI)                        │
│  Data integration · Risk prediction · RAG triage        │
└────────────────────────┬────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────┐
│  Layer 3: Decision Support Outputs                      │
│  CHW alerts · Facility dashboard · Ministry reporting   │
└────────────────────────┬────────────────────────────────┘
                         ↓
        DHIS2 · BHCPF · MAMII · PHC network
```

---

## Key Findings (EDA)

From the Southwest Nigeria analytical sample (NDHS 2018):

| Metric                              | Result        |
|-------------------------------------|---------------|
| ANC completion (4+ visits)          | 88.5%         |
| Home delivery rate                  | 17.7% (n=434) |
| Completed ANC but delivered at home | 10.6% (n=272) |
| ANC dropout (1-3 visits)            | 5.0%  (n=122) |
| No formal care at all               | 6.5% (n=159)  |
| Any TBA involvement                 | 7.2%          |
| Skilled birth attendance            | 79.8%         |

**Headline finding:** 88.5% of women completed the full WHO-recommended ANC schedule, yet 17.7% still delivered at home, evidencing that ANC attendance data alone is insufficient as a predictor of delivery risk, and that the continuity gap between antenatal engagement and facility delivery is the primary data visibility failure the framework addresses.

---
## Notebooks

All notebooks follow an object-oriented pipeline pattern.

| Notebook                                  | Description                                               |
|-------------------------------------------|-----------------------------------------------------------|
| `week_1_3_ndhs_eda.ipynb`                 | Data loading, cleaning, EDA, gap report                   |
| `week_4_5_ndhs_feature_engineering.ipynb` | Feature engineering, synthetic data, train/test split     |
| `week_6_7_baseline_models.ipynb`          | Logistic Regression baseline, Random Forest, GridSearchCV |
| `week_8_10_xgboost_evaluation.ipynb`      | XGBoost, SMOTE, final evaluation metrics, SHAP            |

---
### Pipeline architecture


Weeks 1-3                   Weeks 4-5                 Weeks 6-10
─────────────               ──────────────            ─────────────────
ColumnAuditor               FeatureEngineer           SMOTEPreprocessor
NDHSLoader          ──>     SyntheticBuilder  ──>     ModelTrainer
NDHSCleaner                 DatasetAssembler          FinalEvaluator
MaternalEDA                                           SHAPAnalyser

---

### Model Results

All three models evaluated on held-out test set (n=673, 6.4% adverse outcome rate):

| Model               | AUC-ROC  | Sensitivity | Specificity | F1-Score | Brier Score | CV AUC Mean |
|---------------------|----------|-------------|-------------|----------|-------------|-------------|
| Logistic Regression | 0.8120   | 0.6512      | 0.8397      | 0.3256   | 0.1443      | 0.8999      |
| **Random Forest**   |**0.8444**| **0.6977**  | **0.7952**  |**0.2970**| **0.0521**  | **0.9782**  |
| XGBoost             | 0.8586   | 0.6047      | 0.8762      | 0.3537   | 0.0533      | 0.9726      |

**Recommended model: Random Forest** highest sensitivity (0.6977), correctly identifying 30 of 43 adverse cases. In clinical deployment, sensitivity is the priority metric: a missed high-risk woman is a more costly error than an unnecessary CHW alert.

**All three models exceed the dissertation target AUC-ROC >= 0.80.**

### Known limitations

- **Cross-validation AUC is inflated.** CV AUC (e.g. Random Forest 0.978) was computed on the SMOTE-resampled training set, so synthetic minority samples appear on both sides of each fold. All headline results above come from the untouched held-out test set, which was never resampled.
- **Decision thresholds were tuned on the test set.** Sensitivity, specificity and F1 are therefore optimistic. AUC-ROC and Brier Score do not depend on a threshold and are unaffected.
- **Synthetic records.** 800 synthetic community records were generated from published and NDHS aggregate parameters, so results indicate framework feasibility, not deployment performance. Validation on real programme data is required before any clinical use.

A v2 pipeline will apply SMOTE inside an `imblearn` Pipeline during cross-validation and choose thresholds from out-of-fold training predictions.

---

## Top SHAP Features

Features driving predictions (mean absolute SHAP value, Random Forest):

| Rank | Feature | Mean \|SHAP\| | Clinical meaning |
|---|---|---|---|
| 1 | anc_urine_sample | 0.0666 | Absence of basic pre-eclampsia screening |
| 2 | living_children | 0.0470 | Parity risk |
| 3 | postnatal_check | 0.0433 | Post-delivery care engagement |
| 4 | skilled_birth | 0.0365 | Trained clinical attendance at delivery |
| 5 | wealth_index | 0.0291 | Socioeconomic vulnerability |
| 6 | anc_provider_tba | 0.0275 | TBA involvement at ANC |
| 7 | anc_visits | 0.0267 | ANC attendance frequency |

---


## Dataset

**Primary dataset:** Nigeria Demographic and Health Survey (NDHS) 2018 Individual Recode  
**Source:** [DHS Program](https://dhsprogram.com) - publicly available, requires free registration  
**Geographic filter:** South-West zone -> analytical sample of ~2,563 women with recent births

**Supplementary:** Synthetically generated community records (n=800) modelling the ANC dropout population not captured in the NDHS; women who deliver at home after disengaging from formal care.

**Combined modelling dataset:** 3,363 records · 30 features · 80/20 stratified split

> **Data files are not included in this repository.** The NDHS data is governed by DHS Program terms of use which prohibit redistribution. To replicate this analysis, register at [dhsprogram.com](https://dhsprogram.com) and request the Nigeria 2018 Individual Recode (NGIR7BDT). Place the downloaded .DTA file in `data/raw/` before running the notebooks.

---

## Policy Alignment

This framework is designed to complement existing Nigerian government maternal health initiatives:

- **MAMII** (Maternal and Neonatal Mortality Reduction Innovation Initiative) pilot LGAs: Ado-Odo/Ota, Ijebu-Ode, Odeda
- **BHCPF** (Basic Health Care Provision Fund)
- **DHIS2** national health information system
- **Nigeria Health Sector Renewal Investment Initiative**

---

## Future Work

Following dissertation submission (May 2026), a subsequent phase of development is planned to translate the MamaDey AI framework into a production-ready application, subject to regulatory approval by relevant Nigerian health authorities and structured stakeholder engagement with the Ogun State Ministry of Health and MAMII programme partners.

---

## Citation

```bibtex
@mastersthesis{fagbayi2026MamaDey,
  author  = {Fagbayi, Kanyisola},
  title   = {MamaDey AI: An Artificial Intelligence-Enabled, Community-Integrated
             Framework for Early Maternal Risk Prediction in Ogun State, Nigeria},
  school  = {University of East London},
  year    = {2026},
  type    = {MSc Dissertation}
}
```

---

## Acknowledgements

Primary dataset: Nigeria Demographic and Health Survey 2018, provided by the DHS Program (ICF International) and the National Population Commission of Nigeria.

---

## License

Code: [MIT License](LICENSE)  
Dissertation text: All rights reserved - Kanyisola Fagbayi, 2026

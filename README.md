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
2. **Applies supervised machine learning** to classify pregnant women as low, medium, or high risk; targeting AUC-ROC ≥ 0.80
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
│   ├── ndhs_eda.ipynb          <- Data loading, cleaning, EDA
│   ├── features_synthetic.ipynb  <- Feature engineering + synthetic data
│   ├── baseline_models.ipynb     <- Logistic regression + Random Forest
│   └── xgboost_evaluation.ipynb  <- XGBoost, SMOTE, final evaluation
├── models/           <- Saved model objects (.pkl)
├── outputs/          <- Figures, reports, CSVs for dissertation
├── scripts/          <- Any reusable utility functions
└── README.md
```

## Research Status

| Phase | Status |
|---|---|
| Proposal and literature review | Complete |
| Data acquisition (NDHS 2018) | Complete |
| Data cleaning and EDA | Complete |
| Feature engineering and synthetic data | Complete |
| Model development (Random Forest, XGBoost, LR) | In progress |
| Framework evaluation | Pending |
| Dissertation writing and submission | Pending - 24 May 2026 |

---

## Framework Architecture

```
┌─────────────────────────────────────────────────────────┐
│  Layer 1: Community Data Input (MamaCare)              │
│  MamaCare app · CHW visits · Women's groups (NFWP)      │
└────────────────────────┬────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────┐
│  Layer 2: AI Engine (MamaDey AI)                       │
│  Data integration · Risk prediction · RAG triage        │
└────────────────────────┬────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────┐
│  Layer 3: Decision Support Outputs                     │
│  CHW alerts · Facility dashboard · Ministry reporting   │
└────────────────────────┬────────────────────────────────┘
                         ↓
        DHIS2 · BHCPF · MAMII · PHC network
```

---

## Dataset

**Primary dataset:** Nigeria Demographic and Health Survey (NDHS) 2018 Individual Recode  
**Source:** [DHS Program](https://dhsprogram.com) - publicly available, requires free registration  
**Geographic filter:** South-West zone -> analytical sample of ~2,563 women with recent births

**Supplementary:** Synthetically generated community records (n=800) modelling the ANC dropout population not captured in the NDHS; women who deliver at home after disengaging from formal care.

> **Data files are not included in this repository.** The NDHS data is governed by DHS Program terms of use which prohibit redistribution. To replicate this analysis, register at [dhsprogram.com](https://dhsprogram.com) and request the Nigeria 2018 Individual Recode (NGIR7BDT).

---

## Key Findings (EDA)

From the Southwest Nigeria analytical sample (NDHS 2018):

| Metric | Result |
|---|---|
| ANC completion (4+ visits) | 88.5% |
| Home delivery rate | 17.7% (n=434) |
| ANC dropout (1-3 visits) | 5.0% |
| No formal care at all | 6.5% (n=159) |
| Any TBA involvement | 7.2% |
| Skilled birth attendance | 79.8% |

**Headline finding:** 88.5% of women completed the full WHO-recommended ANC schedule, yet 17.7% still delivered at home, evidencing that ANC attendance data alone is insufficient as a predictor of delivery risk, and that the continuity gap between antenatal engagement and facility delivery is the primary data visibility failure the framework addresses.

---
## Notebooks

All notebooks follow an object-oriented pipeline pattern.

| Notebook | Description |
|---|---|
| `ndhs_eda_oop.ipynb` | Data loading, cleaning, EDA, gap report |
| `features_synthetic.ipynb` | Feature engineering, synthetic data, train/test split |
| `baseline_models.ipynb` | Logistic Regression baseline, Random Forest |
| `xgboost_evaluation.ipynb` | XGBoost, SMOTE, final evaluation metrics |

### Pipeline architecture

```
Week 1-2                    Week 3                    Week 4-5
─────────────               ──────────────            ─────────────────
ColumnAuditor               FeatureEngineer           ModelTrainer
NDHSLoader          ──>     SyntheticBuilder  ──>     Evaluator
NDHSCleaner                 DatasetAssembler          FeatureImportance
MaternalEDA
```

## Model Targets

| Metric | Target |
|---|---|
| AUC-ROC | ≥ 0.80 |
| Sensitivity (Recall) | Maximised clinical priority |
| F1-Score | Reported |
| Brier Score | Reported (calibration) |
| Cross-validation | Stratified 5-fold |

**Algorithms:** Random Forest (primary) · XGBoost (comparator) · Logistic Regression (interpretable baseline)  
**Class imbalance:** SMOTE applied to training set only

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

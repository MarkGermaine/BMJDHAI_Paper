# BMJ DHAI_Paper

# Early Prediction of Gestational Diabetes Mellitus using Machine Learning

This repository contains the code and resources for the research paper titled: **"Evaluation of Machine Learning Models for Early Prediction of Gestational Diabetes Using Retrospective Electronic Health Records from Current and Previous Pregnancies"**. 

This work focuses on the development and evaluation of machine learning models to predict Gestational Diabetes Mellitus (GDM) early in pregnancy using electronic health record (EHR) data. We explore the predictive power of data from the first antenatal visit and assess the improvement in model performance when incorporating data from previous pregnancies.

## Abstract (abridged)

**Objective** Assess whether a compact set of routinely collected variables can predict gestational diabetes mellitus (GDM) at the first antenatal visit, and quantify the added value of previous-pregnancy information.

**Methods** In 27 561 pregnancies (GDM 11.6 %) we trained logistic-regression (LR), random-forest (RF), XGBoost (XGB) and explainable-boosting (EBM) models in four cohorts:  
*First-trimester* (9 variables), *Nulliparous* (7 variables), *Multiparous* (8 variables, first-trimester + previous pregnancy) and *Past-pregnancy* (6 variables, pre-conception).  
Performance was estimated with 5-fold nested cross-validation; discrimination (AUROC ± 95 % CI), calibration and decision-curve net benefit were reported.

**Results** First-trimester models showed AUROC ~0.82 with good calibration.  Multiparous models achieved AUROC 0.88–0.89 after adding previous-pregnancy features.  Past-pregnancy models alone still reached AUROC ~0.86.  All parsimonious models out-performed a dummy baseline and provided positive net benefit across clinically relevant thresholds.

**Conclusion** A handful of non-invasive predictors can give robust early-pregnancy GDM risk estimates; previous-pregnancy history further improves accuracy in multiparous women.  External validation and prospective trials remain necessary before clinical deployment.

---

## 🚀 Quickstart (< 5 minutes)

**Get up and running with the pipeline immediately:**

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Generate synthetic test data
python scripts/generate_dummy_data.py --both --output data

# 3. Run the full pipeline
python run_pipeline.py --data data/dummy_data_large.csv --output output

# 4. Check results
ls output/  # View ROC curves, PR curves, and metrics comparison plots
```

**What you get:**
- ✅ Trained models for all 4 cohorts (First-Trimester, Nulliparous, Multiparous, Past-Pregnancy)
- ✅ 4 algorithms per cohort (Random Forest, Logistic Regression, XGBoost, Explainable Boosting Classifier)
- ✅ Performance metrics: AUROC, sensitivity, specificity, F1, calibration curves, O:E ratios
- ✅ ROC and Precision-Recall visualizations
- ✅ Saved model results (`.joblib` files in `output/`)

---

## 📦 Installation & Dependencies

**Requirements:**
- Python 3.8+
- See `requirements.txt` for full list

**Install dependencies:**

```bash
pip install -r requirements.txt
```

**Key packages:**
- `pandas`, `numpy` — data handling
- `scikit-learn` — preprocessing, model training, evaluation
- `xgboost` — gradient boosting
- `interpret-ml` — explainable boosting classifier
- `matplotlib` — visualizations
- `joblib` — model serialization
- `pytest` — testing

---

## 💻 Usage Examples

### Run the full pipeline with default settings (small test data)
```bash
python run_pipeline.py
```

### Run with custom data
```bash
python run_pipeline.py --data /path/to/your/data.csv --output results/
```

### Run specific cohorts only
```bash
python run_pipeline.py --cohorts first_trimester nulliparous --data data/dummy_data_large.csv
```

### Generate synthetic data (for testing/development)
```bash
# Generate both small and large datasets
python scripts/generate_dummy_data.py --both --output data

# Or generate only large dataset (1500 samples)
python scripts/generate_dummy_data.py --large --output data --seed 42
```

### Run test suite
```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_preprocessing.py -v

# Run with coverage
pytest tests/ --cov=src
```

### Load and inspect saved results
```python
import joblib

# Load results
results = joblib.load('output/model_results_20260410_093312.joblib')

# Access results by cohort and algorithm
first_trimester_rf_metrics = results['first_trimester']['Random Forest']['metrics']
print(f"AUROC: {first_trimester_rf_metrics['auroc']:.4f}")
print(f"Sensitivity: {first_trimester_rf_metrics['sensitivity']:.4f}")
print(f"Specificity: {first_trimester_rf_metrics['specificity']:.4f}")
```

---

## 🏗️ Architecture & Code Overview

**Directory Structure:**

```
BMJDHAI_Paper/
├── src/                              # Core Python modules
│   ├── __init__.py
│   ├── utils.py                      # Feature definitions, constants per cohort
│   ├── preprocessing.py              # Data loading, feature engineering, train/val/test split
│   ├── models.py                     # Training, hyperparameter tuning, evaluation metrics
│   ├── visualization.py              # ROC/PR curves, metrics comparison plots
│   └── pipelines/                    # Cohort-specific orchestrators
│       ├── __init__.py
│       ├── first_trimester.py        # First-Trimester & Nulliparous pipeline
│       └── multiparous.py            # Multiparous & Past-Pregnancy pipeline
│
├── scripts/                          # Utility scripts
│   └── generate_dummy_data.py        # Create synthetic test datasets
│
├── tests/                            # Pytest test suite
│   ├── conftest.py                   # Pytest fixtures (dummy data)
│   ├── test_preprocessing.py         # Validate data preprocessing  
│   ├── test_models.py                # Validate model training & evaluation
│   ├── test_pipelines.py             # End-to-end pipeline tests
│   └── test_output_validation.py     # Validate output metrics
│
├── data/                             # Synthetic datasets (generated)
│   ├── dummy_data_small.csv          # 200 samples (for quick tests)
│   └── dummy_data_large.csv          # 1500 samples (for realistic runs)
│
├── output/                           # Results and visualizations (auto-generated)
│   ├── roc_curves_*.png
│   ├── pr_curves_*.png
│   ├── metrics_comparison_*.png
│   └── model_results_*.joblib
│
├── Models/                           # Pre-trained models (from original research)
│   ├── First_Trimester.joblib
│   ├── Nulliparous_Models.joblib
│   ├── Multiparous_Models.joblib
│   └── Past_Pregnancy.joblib
│
├── General Code.ipynb                # Original notebook (reference only)
├── requirements.txt                  # Python dependencies
├── run_pipeline.py                   # Main entry point
└── README.md                         # This file
```

**Module Responsibilities:**

| Module | Purpose |
|--------|---------|
| `utils.py` | Cohort feature definitions, hyperparameter ranges, validation rules |
| `preprocessing.py` | Load CSV → drop duplicates → one-hot encode categoricals → standardize numericals → GroupShuffleSplit |
| `models.py` | RandomizedSearchCV with 5-fold StratifiedKFold CV → train 4 algorithms → evaluate with bootstrap AUROC ± 95% CI |
| `visualization.py` | Plot ROC/PR curves with confidence intervals, metrics comparison bar charts |
| `pipelines/first_trimester.py` | Orchestrate FTP-9 (9 features) & Nulliparous (7 features) cohorts |
| `pipelines/multiparous.py` | Orchestrate Multiparous (8 features) & Past-Pregnancy (6 features) cohorts |
| `run_pipeline.py` | CLI interface: load data → route to appropriate pipeline → save results |

**Data Flow:**

```
CSV Input
  ↓
preprocessing.prepare_first_trimester_data()  OR  preprocessing.prepare_multiparous_data()
  ↓
Feature engineering (OneHotEncoder + StandardScaler)
  ↓
Train/Val/Test split (GroupShuffleSplit to keep patient IDs together)
  ↓
models.train_model_with_hyperparameter_tuning()  [×4 algorithms]
  ↓
models.evaluate_model()  [AUROC, sensitivity, specificity, F1, calibration]
  ↓
visualization.plot_*() + joblib.dump()
  ↓
Results in output/ directory
```

---

## 📊 Data Specifications

### Input CSV Requirements

Your CSV file must include **all** of the following columns (or a subset appropriate for your cohort):

**Common to all cohorts:**
- `ID` — unique patient identifier
- `GDM` — target variable (0=no GDM, 1=GDM diagnosed)
- `Date of Birth` — for sorting sequential pregnancies (format: YYYY-MM-DD)
- `Ethnic Origin of Patient` — categorical (e.g., "White", "Black", "Asian")
- `Age at booking` — age at first antenatal visit (numeric)
- `BMI` — body mass index (numeric)
- `Systolic BP` — systolic blood pressure (numeric)
- `Diastolic BP` — diastolic blood pressure (numeric)

**First-Trimester & Nulliparous cohorts:**
- `Hx_GDM` — history of GDM (0/1) [Nulliparous uses only]
- `FH Diabetes` — family history of diabetes (0/1)
- `Other Endocrine problems` — binary (0/1)
- `Parity` — number of previous live births (numeric) [First-Trimester only]

**Multiparous & Past-Pregnancy cohorts:**
- `Hx_GDM` — history of GDM (0/1)
- `FH Diabetes` — family history of diabetes (0/1)
- `Inter-pregnancy weight change` — weight change between pregnancies (kg) [Multiparous only]
- `Inter-pregnancy interval` — time between pregnancies (years) [Multiparous only]
- `Previous birth weight percentile` — percentile of previous infant birth weight (0-100)

### Synthetic Data Generation

Run `scripts/generate_dummy_data.py` to create realistic test datasets with:
- Appropriate feature distributions (BMI, blood pressure, age, etc.)
- GDM prevalence ~12-16% (matches study population)
- Valid medical ranges for all variables
- 200 samples (small dataset) or 1500 samples (large dataset)



```
BMJDHAI_Paper/
├─ models/                       # Git-LFS tracked bundles
│  ├─ First_Trimester.joblib        # First-Trimester
│  ├─ Nulliparous.joblib   # Nulliparous
│  ├─ Past_Pregnancy.joblib           # Past-Pregnancy
│  └─ Multiparous_Models.joblib          # Multiparous
├─ preprocessors/
│  ├─ First-Trimester_preprocessor.pkl
│  ├─ Nulliparous_preprocessor.pkl
│  ├─ Multiparous_preprocessor.pkl
│  └─ PastPregnancy_preprocessor.pkl
├─ src/
│  └─ utils.py                  # simple loaders
├─ notebooks/
│  └─ 01_quick_demo.ipynb       # short load-and-predict demo
├─ requirements.txt
└─ .gitattributes               # tells Git to use LFS for *.joblib
```

Each bundle stores the four fitted estimators (RF, LR, XGB, EBM) plus ROC/PR arrays and calibration data.

## Model feature sets

| Cohort / bundle | # Predictors | Key variables |
|-----------------|-------------|---------------|
| **First-trimester** | 9 | Ethnicity, family-history-diabetes, history-of-GDM, endocrine problems, parity, age, BMI, systolic BP, diastolic BP |
| **Nulliparous** | 7 | Same as FTP-9 except parity & history-of-GDM |
| **Multiparous** | 8 | Ethnicity, age, inter-pregnancy weight-change, history-of-GDM, BMI, inter-pregnancy interval, previous birth-weight percentile, family-history-diabetes |
| **Past-pregnancy** | 6 | Ethnicity, age, history-of-GDM, BMI, previous birth-weight percentile, family-history-diabetes |

---

## ✅ Fact-Check Against Research Methods

| Claim | Source | Status |
|-------|--------|--------|
| Study population: 27,561 pregnancies | Abstract | ✅ Documented in README |
| GDM prevalence: 11.6% | Abstract | ✅ Synthetic data ~12-16% to match |
| First-Trimester model AUROC ~0.82 | Abstract & notebook cell 5 | ✅ Expected range: 0.80-0.85 |
| Multiparous model AUROC 0.88-0.89 | Abstract & notebook cell 15 | ✅ Expected range with previous pregnancy data |
| Past-Pregnancy model AUROC ~0.86 | Abstract | ✅ Expected range: 0.84-0.88 |
| 4 ML algorithms evaluated | Methods & notebook | ✅ RF, LR, XGB, EBM implemented |
| 5-fold StratifiedKFold CV | Methods & notebook cell 5 | ✅ Used in `train_model_with_hyperparameter_tuning()` |
| Bootstrap 95% CI on AUROC (1000 iterations) | Methods & notebook | ✅ Implemented in `bootstrap_ci_auroc()` |
| Sensitivity, Specificity, F1, Brier Score | Methods & notebook | ✅ All calculated in `evaluate_model()` |
| Calibration slope & intercept reported | Methods & notebook | ✅ Implemented in `calculate_calibration_metrics()` |
| O:E ratio (observed:expected) | Methods & notebook | ✅ Implemented in `calculate_o_e_ratio()` |
| First-Trimester: 9 predictors | Methods Table 1 | ✅ See `COHORT_FEATURES["first_trimester"]` |
| Nulliparous: 7 predictors (no parity, no Hx_GDM) | Methods Table 1 | ✅ See `COHORT_FEATURES["nulliparous"]` |
| Multiparous: 8 predictors (includes inter-pregnancy vars) | Methods Table 1  | ✅ See `COHORT_FEATURES["multiparous"]` |
| Past-Pregnancy: 6 predictors | Methods Table 1 | ✅ See `COHORT_FEATURES["past_pregnancy"]` |
| GroupShuffleSplit keeps patient IDs together | Methods (avoiding data leakage) | ✅ Used in `prepare_*_data()` |
| 80/10/10 split (First-Trimester cohorts) | Methods & notebook | ✅ `SPLIT_RATIOS["first_trimester"]` |
| 70/15/15 split (Multiparous cohorts) | Methods & notebook | ✅ `SPLIT_RATIOS["multiparous"]` |

**Notes:**
- Pre-trained models in `Models/` bundles are from the original research
- Synthetic data is for **testing/development only** — not clinical-grade
- External validation on real EHR data strongly recommended before deployment
- This refactored code maintains full methodological fidelity to the original research

---

##---

## Usage

These models and code are provided for **research purposes only**. External validation on real EHR data is required before any clinical deployment.

Researchers can:
- Use the synthetic data generator for testing and development
- Train models on your own datasets using the provided pipeline
- Evaluate model performance on new data
- Build upon this work for further research



## Citation

If you use the models or code from this repository in your research, please cite our paper:

Germaine, M., O'Higgins, A. C., Egan, B., & Healy, G. (2024). Evaluation of Machine Learning Models for Early Prediction of Gestational Diabetes Using Retrospective Electronic Health Records from Current and Previous Pregnancies. *BMJ Digital Health and AI*.


## Data and Code Availability

Due to patient confidentiality and data use agreements, the individual-level data used in this study cannot be shared publicly. The code and saved models are available in this repository for other researchers to use for research purposes. 

## Funding

This research was supported in part by a grant from Research Ireland Ireland under Grant Number 18/CRT/6183. 

## Contact

For any questions or collaborations, please contact Mark Germaine at mark.germaine2@mail.dcu.ie.

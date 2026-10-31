# AROHAN: Clinical Machine Learning & Multimodal Data Intelligence Report
**Generation Timestamp:** `2026-10-31 12:34:23 UTC`  
**Research Institution / Target Population:** Indian Hospital & Clinical Cohorts  
**Evaluation Environment:** Python 3.14.6 · scikit-learn 1.4 · Windows 11  
---
## 1. Title, Purpose and Supervisor's Brief
This project executes the supervisor's mandate to design and validate an artificial intelligence pipeline for the early detection and clinical phenotyping of cardiovascular disease (CVD) using **Indian-population clinical data** (Mendeley Hospital Cohort, IndiaAI / AIKosh, Eka Care). The brief requires: 
1. Classifying individuals into **Healthy** versus **Cardiovascular disease (type not specified in the source data)**, avoiding Western benchmark dataset substitutes.
2. Evaluating both baseline supervised learning and a **hybrid architecture** incorporating unsupervised cluster-distance features.
3. Modular integration of multiple clinical modalities: structured clinical measurements, laboratory investigation reports, clinical consultation notes, longitudinal patient histories, medical report images, and echocardiography reports.
4. Strict scientific honesty: zero premature late fusion without genuine patient-level linkages, and explicit rejection of circular diagnostic labeling from laboratory thresholds.

## 2. Datasets Used, Provenance & Checksums
All data provenance, acquisition timestamps, and SHA-256 cryptographic checksums are recorded dynamically:
| Dataset | Modality | Source / DOI | Licence | Records | SHA-256 Checksum | Limitations |
|---|---|---|---|---|---|---|
| **Cardiovascular Disease Dataset** | Structured measurements | [Mendeley Data](https://data.mendeley.com/datasets/dzz48mvjht/1) | CC BY 4.0 | 1,000 | `9376367dde41e79e…` | Single Indian center; binary unverified target; no cross-modal patient ID |
| **NidaanKosha-100k-V1.0** | Laboratory reports | [Hugging Face](https://huggingface.co/datasets/ekacare/NidaanKosha-100k-V1.0) | CC-BY-SA-4.0 | 6,844,304 readings (100,000 lab reports) | `a55637e03e…`, `178bccdcbb…` | Extracted via vision LLM; OCR noise; unsupervised only; no patient linkage |
| **Clinical Note Generation** | Clinical consultations | [Eka Care HF](https://huggingface.co/datasets/ekacare/clinical_note_generation_dataset) | MIT | 100 consultations | `b1f85a5ec3bed07c…` | Medical transcription; weak labels require leakage guards |
| **Medical Records Parsing Validation** | Document images | [Eka Care HF](https://huggingface.co/datasets/ekacare/medical_records_parsing_validation_set) | MIT | 20 images (9.79 MB) | Documented parquet/image store | Diverse Indian diagnostics; vision AI profiling |
| **Medical History Summarisation** | Longitudinal journeys | [Eka Care HF](https://huggingface.co/datasets/ekacare/ekacare_medical_history_summarisation) | MIT | 58 trajectories | `6aa4a51a5fea1926…` | Longitudinal vitals and treatment histories |
| **Echocardiography Reports** | Echo parameter reports | [InfoBayAI](https://huggingface.co/datasets/InfoBayAI/Echocardiography-Reports-Dataset) | Research Access | 10 reports | `5e247188d83ce859…` | Geographic origin unconfirmed as Indian |

*Attribution Notice:* NidaanKosha-100k-V1.0 is licensed under Creative Commons Attribution-ShareAlike 4.0 International (CC-BY-SA-4.0) by Eka Care and IndiaAI.

## 3. Data Coverage & Modality Architecture
Data coverage is evaluated strictly on genuine files resident in `data/raw/`:
| Clinical Modality | Real Files Present | Execution Status | Reason / Justification |
|---|---|---|---|
| **Structured clinical measurements** | Yes (`Cardiovascular_Disease_Dataset.csv`) | **Available / Active** | Fully loaded, preprocessed, and evaluated across 6 baseline and hybrid models |
| **Blood/Laboratory reports** | Yes (2 parquet files) | **Available / Active** | Extracted lab readings processed via LOINC discovery, unit normalization, and K-Means/PCA |
| **Clinical notes & scribing** | Yes | **Available / Active** | 100 consultation transcripts processed for cardiology mentions and SVD topics |
| **Imaging reports & documents** | Yes | **Available / Active** | Profiled 20 document images and regex-extracted echo parameters |
| **Discharge summaries** | No | **Unavailable** | Not available in genuine open Indian clinical datasets |
| **Multimodal integration** | No | **Unavailable / Withheld** | Cross-cohort late fusion withheld due to absence of genuine shared patient identifiers |

## 4. Data Quality Screening Findings
### 4.1 Structured Indian Hospital Cohort (1,000 rows)
- **Missing Values:** 0 missing cells across all columns.
- **Duplicates:** 0 duplicates ignoring patient ID; 0 exact full duplicates.
- **Zero Values:** Zero values in continuous columns (e.g. cholesterol: 53) were preserved as authentic physiological records rather than arbitrarily imputed.
- **Outliers:** 5 numerical features screened under the 1.5 × IQR rule.

### 4.2 NidaanKosha Laboratory Investigation Reports
- **Cardiometabolic Rows Profiled:** 2,847
- **Non-Numeric Extraction Noise Filtered:** 111 entries (e.g. 'BORDERLINE', 'TRACE').
- **Unit Conversions:** 2 serum creatinine values converted from umol/L to mg/dL (factor 0.011312).
- **Physiological Bounds Screening:** 9 extreme values outside plausible biological limits were flagged as OCR extraction noise.

## 5. Machine Learning Methods per Modality
1. **Structured Modeling:**
   - Split: Stratified 80/20 train/test split (seed 42), yielding 800 training rows and 200 test rows.
   - Preprocessing: Numerical features median-imputed and standard-scaled; categorical features mode-imputed and one-hot encoded inside a strict `sklearn.pipeline.Pipeline` fitted only on training rows.
   - Baseline Classifiers: Logistic Regression, Decision Tree (max depth 5), Random Forest (300 estimators).
   - Hybrid Classifiers: Euclidean distances to K-Means cluster centroids (fitted on training features only) concatenated with original inputs.
   - Validation: 5-fold stratified cross-validation on training rows reported separately from held-out test predictions.
2. **Laboratory Reports:**
   - Unsupervised phenotyping only. To maintain scientific honesty, no synthetic 'disease labels' were manufactured from lab reference thresholds.
   - Continuous features standard-scaled, followed by K-Means clustering (K=2) and 2-component PCA.
3. **Clinical Notes:**
   - Transparent rule-based extraction across 4 categories (symptoms, risk factors, conditions, medications).
   - Latent topic modeling via TF-IDF (1-2 ngrams) and Truncated SVD.
   - Leakage guard: Any derived weak labels exclude the raw source diagnosis field from input features.

## 6. Real Results and Evaluation Benchmarks
### 6.1 Held-Out Test Set Performance
| Model | Accuracy | Precision | Recall (Sens.) | Specificity | F1 Score | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|---|---|
| **Logistic Regression** | 0.975 | 0.966 | 0.991 | 0.952 | 0.979 | 0.997 | 0.998 |
| **Decision Tree** | 0.965 | 0.950 | 0.991 | 0.929 | 0.970 | 0.992 | 0.989 |
| **Random Forest** | 0.980 | 0.975 | 0.991 | 0.964 | 0.983 | 0.999 | 1.000 |
| **Hybrid Logistic Regression** | 0.980 | 0.975 | 0.991 | 0.964 | 0.983 | 0.998 | 0.998 |
| **Hybrid Decision Tree** | 0.965 | 0.950 | 0.991 | 0.929 | 0.970 | 0.998 | 0.997 |
| **Hybrid Random Forest** | 0.985 | 0.983 | 0.991 | 0.976 | 0.987 | 0.999 | 1.000 |

### 6.2 Training Cross-Validation (5-Fold Stratified, Train Split Only)
| Model | Accuracy (Mean ± SD) | Recall (Mean ± SD) | F1 Score (Mean ± SD) | ROC-AUC (Mean ± SD) |
|---|---|---|---|---|
| **Logistic Regression** | 0.965 ± 0.016 | 0.970 ± 0.017 | 0.970 ± 0.014 | 0.989 ± 0.011 |
| **Decision Tree** | 0.935 ± 0.005 | 0.965 ± 0.028 | 0.945 ± 0.004 | 0.977 ± 0.009 |
| **Random Forest** | 0.970 ± 0.016 | 0.978 ± 0.025 | 0.974 ± 0.014 | 0.995 ± 0.006 |
| **Hybrid Logistic Regression** | 0.964 ± 0.014 | 0.970 ± 0.017 | 0.969 ± 0.013 | 0.989 ± 0.010 |
| **Hybrid Decision Tree** | 0.935 ± 0.006 | 0.968 ± 0.025 | 0.945 ± 0.004 | 0.977 ± 0.008 |
| **Hybrid Random Forest** | 0.964 ± 0.020 | 0.981 ± 0.024 | 0.969 ± 0.017 | 0.994 ± 0.008 |

### 6.3 Confusion Matrices (Held-Out Test Set, n=200)
| Model | True Negatives (TN) | False Positives (FP) | False Negatives (FN) | True Positives (TP) |
|---|---|---|---|---|
| **Logistic Regression** | 80 | 4 | 1 | 115 |
| **Decision Tree** | 78 | 6 | 1 | 115 |
| **Random Forest** | 81 | 3 | 1 | 115 |
| **Hybrid Logistic Regression** | 81 | 3 | 1 | 115 |
| **Hybrid Decision Tree** | 78 | 6 | 1 | 115 |
| **Hybrid Random Forest** | 82 | 2 | 1 | 115 |

### 6.4 Data Leakage Screen & Univariate Association
| Feature | Type | Univariate In-Sample AUC | Flagged (AUC >= 0.90) |
|---|---|---|---|
| `slope` | categorical | 0.953 | **FLAGGED** |
| `chestpain` | categorical | 0.823 | No |
| `noofmajorvessels` | categorical | 0.784 | No |
| `restingBP` | numeric | 0.779 | No |
| `restingrelectro` | categorical | 0.723 | No |
| `fastingbloodsugar` | categorical | 0.636 | No |
| `serumcholestrol` | numeric | 0.629 | No |
| `maxheartrate` | numeric | 0.603 | No |
| `oldpeak` | numeric | 0.557 | No |
| `exerciseangia` | categorical | 0.513 | No |
| `age` | numeric | 0.511 | No |
| `gender` | categorical | 0.507 | No |

### 6.5 Feature Ablation Experiment (Impact of Removing Flagged Feature: `slope`)
| Model | ROC-AUC (All Features) | ROC-AUC (Without Slope) | ROC-AUC Change | Recall Change | Accuracy Change |
|---|---|---|---|---|---|
| **Logistic Regression** | 0.997 | 0.944 | -0.053 | -0.129 | -0.115 |
| **Decision Tree** | 0.992 | 0.942 | -0.050 | -0.052 | -0.060 |
| **Random Forest** | 0.999 | 0.989 | -0.011 | -0.026 | -0.045 |
| **Hybrid Logistic Regression** | 0.998 | 0.946 | -0.051 | -0.121 | -0.120 |
| **Hybrid Decision Tree** | 0.998 | 0.941 | -0.057 | -0.052 | -0.070 |
| **Hybrid Random Forest** | 0.999 | 0.989 | -0.010 | -0.026 | -0.055 |

### 6.6 Laboratory Unsupervised Phenotyping (NidaanKosha)
- **Optimal Clusters (k):** 2
- **Silhouette Scores:** {}
| cluster | Lab Reports | creatinine | glucose | hdl_cholesterol | ldl_cholesterol | total_cholesterol | triglycerides | urea | uric_acid | vldl_cholesterol |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Cluster 1 | 85 | 0.98 | 152.5 | 37.55 | 106.85 | 200.83 | 309.48 | 28.57 | 5.79 | 55.99 |
| Cluster 2 | 329 | 0.8 | 104.5 | 45.89 | 107.99 | 180.74 | 128.21 | 21.83 | 5.3 | 24.73 |

### 6.7 Embedded Visualizations
The following figures were generated directly from real model outputs and test evaluations:

#### Figure: Roc Curves
![roc_curves](C:/arohan/outputs/figures/report_roc_curves.png)

#### Figure: Pr Curves
![pr_curves](C:/arohan/outputs/figures/report_pr_curves.png)

#### Figure: Confusion Matrix
![confusion_matrix](C:/arohan/outputs/figures/report_confusion_matrix.png)

#### Figure: Leakage Screen
![leakage_screen](C:/arohan/outputs/figures/report_leakage_screen.png)

#### Figure: Feature Ablation
![feature_ablation](C:/arohan/outputs/figures/report_feature_ablation.png)

#### Figure: Lab Clusters
![lab_clusters](C:/arohan/outputs/figures/report_lab_clusters.png)

## 7. Critical Interpretation and Research Caution
1. **Very High Nominal Metrics:** Ensembles achieve ROC-AUC > 0.99 on the test split. However, the leakage screen reveals that `slope` alone achieves an in-sample AUC of 0.953. When `slope` is ablated, discrimination drops by 5-6 percentage points. High metrics reflect feature-label correlation in this specific hospital cohort rather than generalizable clinical predictive validity.
2. **Binary Disease Target:** The Mendeley dataset provides a single binary target column without etiology or disease sub-classification (e.g. CAD, STEMI, Heart Failure). The model outputs generic cardiovascular disease risk, not a specific diagnostic entity.
3. **Small NLP Cohort:** Clinical notes ($N=100$) provide proof-of-concept for cardiology keyword extraction, but are insufficient for production NLP fine-tuning.

## 8. Limitations & Unavailable Data Modalities
- **No Multimodal Patient Linkage:** No public Indian dataset currently connects structured measurements, lab reports, and clinical notes for the same individuals. Cross-cohort late fusion is strictly impossible without violating record-linkage ethics.
- **Discharge Summaries:** No open Indian hospital discharge summaries could be acquired; this modality is marked unavailable.
- **Vision LLM Local Constraints:** Parrotlet-V-lite-4B requires local GPU resources (>=12GB VRAM) not present on the execution host; external APIs were avoided to preserve data privacy.

## 9. Reproducibility Specifications
### System Environment
- Python: `3.14.6`
- Operating System: `Windows`
- scikit-learn: `1.4`
- Random State / Seed: `42`
### Execution Commands
```powershell
$env:PYTHONPATH="src"
.venv\Scripts\python.exe -m pytest -v
.venv\Scripts\python.exe -m arohan_clinical.report_builder
.venv\Scripts\python.exe -m streamlit run app.py
```

## 10. References and Licences
1. **Mendeley Dataset:** DOI 10.17632/dzz48mvjht.1 (CC BY 4.0).
2. **NidaanKosha-100k-V1.0:** Eka Care & IndiaAI (CC-BY-SA-4.0).
3. **Eka Clinical Notes & Medical History:** Eka Care (MIT Licence).
4. **InfoBayAI Echocardiography Reports:** Research Access.
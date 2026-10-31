# AROHAN — Clinical Data Intelligence

Research prototype for **clinical-data-based cardiovascular disease detection and analysis**, with an Indian-population focus.
It is a local and cloud-deployable Streamlit dashboard (dark theme) with five core pages:
1. **Overview** *(Key metric cards, target distribution, feature distributions, held-out model comparison)*
2. **Dataset and Data Quality** *(Dataset inspection, schema dictionary, missingness, anomaly screening, correlations, and preprocessing)*
3. **Supervised Learning** *(Train/test split, logistic regression, decision tree, random forest, hybrid models, PR/ROC curves, threshold analysis, feature importance, ablation, and single-patient experimental prediction)*
4. **Unsupervised Learning** *(Exploratory K-Means clustering and PCA on continuous measurements)*
5. **Labs, Notes and Reports** *(Laboratory data analysis (NidaanKosha demo/full), clinical notes NLP, patient history trajectories, report image profiling, and echo parameter extraction & fusion readiness)*

---

## Dataset Attribution & Licensing
- **Mendeley Cardiovascular Disease Dataset**: Acquired from an Indian multispecialty hospital. Citation: Mendeley Data, DOI: [10.17632/dzz48mvjht.1](https://data.mendeley.com/datasets/dzz48mvjht/1), Version 1, 16 April 2021. Licensed under Creative Commons Attribution 4.0 International ([CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)). Committed directly in `data/raw/Cardiovascular_Disease_Dataset.csv` for cloud demonstration.
- **NidaanKosha-100k-V1.0 Demo Sample**: Derived from Eka Care & IndiaAI. Licensed under Creative Commons Attribution-ShareAlike 4.0 International ([CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/)). Distributed as a 25,000-reading sample in `data/demo/labs_cardiometabolic_sample.parquet`.

---

## What Works Now vs. What Is Unavailable

| Clinical data type | Status | Details |
|---|---|---|
| **Structured clinical measurements** (Mendeley CSV) | **Works** | Loaded, profiled, baseline & hybrid supervised models (LR, DT, RF) + unsupervised K-Means/PCA |
| **Laboratory reports** (NidaanKosha 100k) | **Works** | Parquet data loaded (6.84M readings, 100,000 lab reports across India); biomarker extraction (lipids, glucose, creatinine), unit normalization (umol/L to mg/dL), anomaly filtering, and unsupervised K-Means/PCA clustering |
| **Clinical notes & consultations** (Eka Care) | **Pipeline Ready / Pending Auth** | Tokenization, TF-IDF, TruncatedSVD topics, cardiology keyword extractor, and leak-free weak-label classifier implemented. Awaiting user login/consent on gated repository. |
| **Patient history & vital trajectories** (Eka Care) | **Pending Auth** | 58 longitudinal patient journeys. Awaiting user login/consent on gated repository. |
| **Report images** (Eka Care) | **Profile Ready / Pending Auth** | 288 lab-report and prescription images. Format and size profiling ready. |
| **Echocardiography reports** (InfoBayAI) | **Extractor Ready / Pending Auth** | Validated regex parameter extractor (LVEF, LVEDD, LA dimension). Dataset country of origin not confirmed as Indian. |
| **Vision LLM** (Parrotlet-V-lite-4B) | **Model not run** | Requires local GPU (>=12GB VRAM). Zero external API calls made to protect research integrity. |
| **Multimodal integration** | **Late fusion not performed** | Unlinked cohorts: no shared patient identifier exists across public datasets. Premature late fusion is rejected. Standardized export schema defined for future fusion integration. |

---

## Dataset Acquisition and Directory Structure

### Directory Layout
```
arohan/
  app.py                         Streamlit navigation entry point (6 pages)
  assets/arohan_logo.svg         Sidebar branding logo
  data/
    raw/
      Cardiovascular_Disease_Dataset.csv   Mendeley structured CSV
      labs/data/                           NidaanKosha parquet files
      notes/                               Clinical notes parquet / CSV files
      history/                             Patient history summarisation files
      report_images/                       Lab report & prescription images
      echo_reports/                        Echocardiography text reports
    processed/                   Train/test split data (identifiers excluded)
  models/                        Trained pipeline artifacts (.joblib)
  outputs/
    figures/                     Generated plots
    tables/                      Ablation, cluster and evaluation CSVs
    reports/
      dataset_registry.json      Formal provenance, checksums, and record counts
      supervised_results.json    Trained model evaluation metrics
  src/arohan_clinical/           Core source code modules:
    config.py, data_loader.py, data_quality.py, preprocessing.py,
    supervised.py, unsupervised.py, evaluation.py, clinical_coverage.py,
    utils.py, ui.py, views.py, acquisition.py, labs.py, notes.py, reports.py
  tests/                         Pytest suite with synthetic tmp_path fixtures
  REPORT.md                      Comprehensive research and clinical engineering report
  CITATIONS.md                   Formal citations and licence terms
```

### Dataset Provenance & Download Guides

1. **Cardiovascular Disease Dataset (Mendeley Data)**
   - **URL:** [https://data.mendeley.com/datasets/dzz48mvjht/1](https://data.mendeley.com/datasets/dzz48mvjht/1)
   - **Licence:** CC BY 4.0 (DOI: 10.17632/dzz48mvjht.1)
   - **Destination:** `arohan/data/raw/Cardiovascular_Disease_Dataset.csv`

2. **NidaanKosha-100k-V1.0 (Eka Care / IndiaAI)**
   - **URL:** [https://huggingface.co/datasets/ekacare/NidaanKosha-100k-V1.0](https://huggingface.co/datasets/ekacare/NidaanKosha-100k-V1.0) or [AIKosh](https://aikosh.indiaai.gov.in/home/datasets/details/nidaankosha_100k_v1.html)
   - **Licence:** CC-BY-SA-4.0
   - **Status:** Downloaded in `arohan/data/raw/labs/data/` (6,844,304 readings across 100,000 lab reports).

3. **Clinical Note Generation Dataset (Eka Care — Gated)**
   - **URL:** [https://huggingface.co/datasets/ekacare/clinical_note_generation_dataset](https://huggingface.co/datasets/ekacare/clinical_note_generation_dataset)
   - **Licence:** MIT
   - **Action Required by User:**
     1. Open the Hugging Face link above in your web browser.
     2. Log in with your Hugging Face credentials and click **Agree and access repository**.
     3. In your terminal, run `hf auth login` using your personal access token.
     4. Download `test-00000.parquet` into `arohan/data/raw/notes/`.

4. **Medical History Summarisation Dataset (Eka Care — Gated)**
   - **URL:** [https://huggingface.co/datasets/ekacare/ekacare_medical_history_summarisation](https://huggingface.co/datasets/ekacare/ekacare_medical_history_summarisation)
   - **Licence:** MIT
   - **Destination:** `arohan/data/raw/history/`

5. **Medical Records Parsing Validation Set (Eka Care — Gated)**
   - **URL:** [https://huggingface.co/datasets/ekacare/medical_records_parsing_validation_set](https://huggingface.co/datasets/ekacare/medical_records_parsing_validation_set)
   - **Licence:** MIT
   - **Destination:** `arohan/data/raw/report_images/`

6. **Echocardiography Reports Dataset (InfoBayAI — Gated / Optional)**
   - **URL:** [https://huggingface.co/datasets/InfoBayAI/Echocardiography-Reports-Dataset](https://huggingface.co/datasets/InfoBayAI/Echocardiography-Reports-Dataset)
   - **Contact:** `datareq@infobay.ai`
   - **Limitation:** Country of origin not confirmed as Indian.
   - **Destination:** `arohan/data/raw/echo_reports/`

---

## Installation & Running (Windows)

### 1. Environment Setup
```bat
cd C:\arohan
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install -r requirements.txt
```

### 2. Run Dashboard
```bat
$env:PYTHONPATH="src"
.\.venv\Scripts\python.exe -m streamlit run app.py
```

### 3. Run Pytest Suite
```bat
$env:PYTHONPATH="src"
.\.venv\Scripts\python.exe -m pytest -q
```
All tests use synthetic, self-contained fixtures generated inside `tmp_path`. Raw dataset files are never modified.

---

## Clinical Machine Learning Methodology

### 1. Diagnostic Honesty & Output Wording
- Binary predictions are classified strictly as **"Healthy"** or **"Cardiovascular disease (type not specified in the source data)"** unless a specific disease diagnosis is explicitly verified in the ground truth labels.
- Model scores are displayed as uncalibrated model confidences, not diagnostic probabilities.

### 2. Rejection of Circular Lab Thresholds
- Creating artificial disease labels from lab reference thresholds (e.g. labeling cholesterol > 200 as "heart disease") and then training a classifier to predict them is circular. Reference ranges are used strictly for descriptive profiling and exploratory unsupervised clustering.

### 3. Weak Labels & Leakage Prevention
- When training NLP models on clinical notes using derived condition labels, the raw source diagnosis field is strictly excluded from feature inputs. Preprocessors (TF-IDF, TruncatedSVD) are fitted exclusively on training rows within cross-validation folds.

### 4. Supervised and Hybrid Modeling
- Evaluates Logistic Regression, Decision Tree (max depth 5), and Random Forest (300 trees).
- For each classifier, a **Hybrid** variant is evaluated, adding K-Means distance features fitted strictly on training rows.
- 80/20 stratified train/test split (seed 42), with 5-fold cross-validation on training rows reported alongside held-out test metrics.

### 5. Unsupervised Learning & Exploration
- Standard scaling, K-Means (k chosen by silhouette score over 2–8 unless overridden), and 2-component PCA on continuous clinical measurements and NidaanKosha laboratory reports.
- Discovered clusters represent exploratory geometric groups, not clinical diagnoses.

### 6. Multimodal Fusion Readiness
- Because structured data, lab reports, and clinical notes belong to separate cohorts with no shared patient identifier, late fusion is explicitly withheld to maintain scientific honesty.
- A standardized export schema (`record_id`, `modality`, `score_or_embedding`, `cluster_id`, `source_dataset`) is provided for future integration when linked multi-center patient registries become available.

---

## Key Research Limitations

1. **No Shared Patient Identifier Across Modalities:** Structured clinical data, lab reports, and clinical notes come from distinct cohorts; late multimodal fusion cannot be performed without true patient-level linkage.
2. **Unverified Target Definition in Structured Data:** The binary target in the Mendeley dataset is cross-sectional with unverified clinical criteria; models predict generic cardiovascular disease status without specific sub-types.
3. **Extraction Noise in Lab Readings:** NidaanKosha readings were parsed by vision LLMs from heterogeneous lab reports across India, introducing OCR and non-numeric artifacts requiring automated screening.
4. **Uncalibrated Model Scores:** Predicted risk scores reflect raw model confidences from a single train/test split without prospective clinical calibration.
5. **Small Sample Sizes for NLP:** Clinical consultation datasets are small ($N < 200$) with derived weak labels, carrying risk of variance and lack of generalization.
6. **Research Prototype Only:** Not a validated medical device; provides no diagnostic determinations or treatment recommendations.

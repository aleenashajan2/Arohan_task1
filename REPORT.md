# AROHAN: Clinical Data Intelligence for Cardiovascular Disease Detection

## Technical and Research Report

**Project Name:** AROHAN (Clinical AI Prototype)  
**Target Domain:** Cardiovascular Disease (CVD) Early Detection and Multimodal Clinical Intelligence  
**Geographic & Population Focus:** Indian Clinical Population (AIKosh, Mendeley Data, Eka Care)  
**Environment:** Python 3.14 · scikit-learn · Streamlit · PyArrow · Plotly  
**Licence & Provenance:** See [CITATIONS.md](file:///C:/arohan/CITATIONS.md)

---

## 1. Executive Summary & Supervisor Brief

Cardiovascular disease is the leading cause of mortality in India, characterized by premature onset and rapid progression compared to Western cohorts. This project addresses the supervisor's brief:
1. **Target Indian-Population Data:** Prioritize genuine Indian datasets (AIKosh, open Indian hospital repositories, Eka Care) over Western benchmark datasets.
2. **Modular Multimodal Architecture:** Support structured clinical measurements, laboratory investigation reports, clinical notes/consultation transcripts, patient longitudinal history, report images, and echocardiography reports.
3. **Rigorous Scientific Honesty:**
   - Clearly delineate working components from pending or gated components.
   - Refuse artificial late fusion when cohorts do not share a true patient identifier.
   - Prohibit circular target labeling based on laboratory reference thresholds.
   - Restrict diagnostic output terminology to *"Healthy"* or *"Cardiovascular disease (type not specified in the source data)"* when fine-grained disease subtypes are absent in the ground truth.
   - Restrict experimental notices strictly to prediction outputs.

---

## 2. Dataset Registrations & Provenance

The system maintains a centralized dataset registry (`outputs/reports/dataset_registry.json`) tracking SHA-256 checksums, record counts, and licencing:

| Modality / Dataset | Source Repository | Licence | Size & Scope | Status in AROHAN |
|---|---|---|---|---|
| **Structured Clinical Measurements** | Mendeley Data (DOI 10.17632/dzz48mvjht.1) | CC BY 4.0 | 1,000 subjects, 12 clinical features | **Fully Active:** Preprocessed, supervised baseline & hybrid models, unsupervised K-Means/PCA |
| **Laboratory Reports (NidaanKosha-100k-V1.0)** | Eka Care / IndiaAI / Hugging Face | CC-BY-SA-4.0 | 6,844,304 readings across 100,000 lab reports | **Fully Active:** Dynamic LOINC test discovery, unit conversion ($\mu\text{mol/L} \to \text{mg/dL}$), physiological anomaly filtering, unsupervised cardiometabolic clustering |
| **Clinical Notes & Consultations** | Eka Care (`clinical_note_generation_dataset`) | MIT | Multi-lingual clinical consultations (EN, HI, MR) | **Pipeline Ready / Pending Auth:** Transparent keyword/entity extractor, TF-IDF + TruncatedSVD topics, leak-free weak-label classifier |
| **Longitudinal Patient History** | Eka Care (`ekacare_medical_history_summarisation`) | MIT | 58 longitudinal patient timelines with vitals | **Pending Auth:** Trajectory parser and risk trend visualizer |
| **Lab Report & Prescription Images** | Eka Care (`medical_records_parsing_validation_set`) | MIT | 288 diverse Indian diagnostic images | **Profile Ready / Pending Auth:** Image dimension/format profiler |
| **Echocardiography Reports** | InfoBayAI | Research Access | Semi-structured echo reports | **Extractor Ready / Pending Auth:** Validated regex extractor for LVEF, LVEDD, and LA dimension |

---

## 3. Modality Separation & Multimodal Fusion Readiness

A core scientific finding of this project is that **no public Indian clinical dataset currently links laboratory reports, clinical notes, imaging, and structured measurements for the same individuals**.
- **Rejection of Pseudo-Fusion:** Simulating late fusion by concatenating unlinked records from different individuals produces spurious correlations and violates clinical research ethics.
- **Modular Pipeline:** Each clinical data type is treated as an independent diagnostic module with dedicated quality checks and exploratory phenotyping.
- **Standardized Export Schema for Future Linkage:**
  ```json
  {
    "record_id": "document_or_subject_identifier",
    "modality": "structured | labs | notes | reports | echo",
    "score_or_embedding": [0.12, 0.85, ...],
    "cluster_id": 1,
    "source_dataset": "NidaanKosha-100k-V1.0"
  }
  ```
  When a multi-center hospital cohort with genuine cross-modality patient identifiers becomes available, this schema allows immediate late fusion without restructuring model architectures.

---

## 4. Data Quality Screening, Unit Normalization, & Filtering

### 4.1 NidaanKosha Laboratory Investigation Reports
NidaanKosha readings were extracted using vision LLMs from diverse Indian diagnostic reports, introducing OCR artifacts and unit heterogeneity:
- **Unit Normalization:** Serum creatinine values recorded in $\mu\text{mol/L}$ are safely converted to $\text{mg/dL}$ ($\text{value} \times 0.011312$).
- **Non-Numeric Handling:** Textual entries (e.g., `"NEGATIVE"`, `"BORDERLINE"`, `"TRACE"`) are isolated and excluded from continuous biomarker matrices.
- **Physiological Bounds Screening:** Plausible cardiometabolic bounds are enforced:
  - Fasting Blood Sugar: $30 - 600\text{ mg/dL}$
  - Serum Creatinine: $0.2 - 25.0\text{ mg/dL}$
  - Total Cholesterol: $50 - 600\text{ mg/dL}$
  - Serum Triglycerides: $30 - 1500\text{ mg/dL}$
  Values outside these boundaries represent optical character recognition noise or instrument errors and are filtered out rather than silently passed to downstream models.

### 4.2 Echocardiography Parameter Extraction
Clinical parameters are parsed with regex boundary guards:
- **Left Ventricular Ejection Fraction (LVEF):** Plausible range $10\% - 85\%$.
- **Left Ventricular End-Diastolic Dimension (LVEDD):** Plausible range $20 - 80\text{ mm}$.
- **Left Atrial (LA) Dimension:** Plausible range $15 - 70\text{ mm}$.
Values exceeding these clinical boundaries (e.g., $120\text{ mm}$) are flagged as `plausible=False` to prevent model distortion.

---

## 5. Supervised & Hybrid Machine Learning Benchmarks

### 5.1 Modeling Protocol
- **Dataset:** Mendeley Indian Hospital CVD Cohort ($N = 1,000$).
- **Features:** 11 continuous and discrete clinical variables (age, blood pressure, cholesterol, resting ECG, blood sugar, etc.).
- **Split:** 80% train / 20% held-out test split, stratified by the binary target class (seed 42). All scalers, imputers, and encoders are fit strictly on training data.
- **Baseline Models:** Logistic Regression, Decision Tree (max depth 5), Random Forest (300 estimators, untuned).
- **Hybrid Models:** To test whether unsupervised geometric representations improve supervised prediction, each classifier is also evaluated as a **Hybrid**, where Euclidean distances to $K$-Means cluster centers (fit on training rows only) are concatenated with the original features.

### 5.2 Held-Out Performance Metrics

| Classifier | Model Type | Accuracy | Recall (Sensitivity) | Precision | F1 Score | ROC-AUC |
|---|---|---|---|---|---|---|
| **Logistic Regression** | Baseline | 0.865 | 0.852 | 0.880 | 0.866 | 0.938 |
| **Logistic Regression** | Hybrid (+ K-Means) | 0.865 | 0.861 | 0.872 | 0.867 | 0.939 |
| **Decision Tree** | Baseline | 0.965 | 0.954 | 0.981 | 0.967 | 0.979 |
| **Decision Tree** | Hybrid (+ K-Means) | 0.970 | 0.963 | 0.981 | 0.972 | 0.980 |
| **Random Forest** | Baseline | 0.985 | 0.981 | 0.991 | 0.986 | 0.999 |
| **Random Forest** | Hybrid (+ K-Means) | 0.985 | 0.981 | 0.991 | 0.986 | 0.999 |

*Note: While tree-based ensembles achieve high nominal discrimination on this split, cross-sectional evaluation does not establish prospective clinical validity.*

---

## 6. Unsupervised Phenotyping

Unsupervised clustering identifies latent cardiometabolic risk groups without relying on circular diagnostic thresholds:
- **Laboratory Cohort Clustering (NidaanKosha):**
  - Features: Jointly available glucose, creatinine, cholesterol, and triglyceride measurements per lab report.
  - Methodology: Standard scaling followed by K-Means clustering ($K=2$ to $K=6$) and 2-component Principal Component Analysis (PCA).
  - Silhouette Scores: Tracked across $K$ values to identify natural geometric clustering tendencies.
  - Biomarker Dissection: Cluster summaries show distinct clusters separating high-triglyceride/low-HDL phenotypes from normal metabolic profiles.

---

## 7. Diagnostic Honesty & Guardrails

1. **Strict Placement of Experimental Disclaimers:**
   The warning notice *"Experimental model output — not a clinical diagnosis."* is presented **strictly and exclusively** where a live model prediction or risk score is computed (the single-patient prediction form and its generated output). It is omitted from general exploratory tabs, documentation pages, and data preview tables to eliminate UI clutter while maintaining compliance.
2. **Rejection of Circular Reference Thresholds:**
   Models are never trained on synthetic labels derived from lab reference ranges (e.g., labeling high cholesterol as cardiovascular disease).
3. **Leakage-Free Clinical NLP:**
   Derived diagnosis keywords from clinical consultation texts are strictly excluded from the feature space during supervised NLP training.
4. **Terminology Policy:**
   NidaanKosha records are identified by `document_id` and are strictly designated as **"Lab reports"**, not patients or subjects.

---

## 8. Limitations & Future Clinical Work

1. **No Shared Patient Identifier Across Modalities:** Structured clinical data, lab reports, and clinical notes belong to separate cohorts; true multimodal late fusion cannot be performed.
2. **Unverified Target Definition in Structured Data:** The binary target in the Mendeley dataset is cross-sectional with unverified criteria; models output generic cardiovascular disease status without specific sub-types.
3. **Extraction Noise in Lab Readings:** NidaanKosha readings were parsed by vision LLMs from heterogeneous lab reports across India, introducing OCR and non-numeric artifacts requiring automated screening.
4. **Uncalibrated Model Scores:** Predicted risk scores reflect raw model confidences from a single train/test split without prospective clinical calibration.
5. **Small Sample Sizes for NLP:** Clinical consultation datasets are small ($N < 200$) with derived weak labels, carrying risk of variance and lack of generalization.
6. **Research Prototype Only:** Not a validated medical device; provides no diagnostic determinations or treatment recommendations.

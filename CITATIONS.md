# Citations, Attributions and Licences

This document records the formal citations, licenses, and terms of use for all datasets incorporated or supported in **AROHAN Clinical AI**.

---

## 1. Mendeley Cardiovascular Disease Dataset
- **Citation**:
  > Cardiovascular Disease Dataset, Mendeley Data, DOI: [10.17632/dzz48mvjht.1](https://data.mendeley.com/datasets/dzz48mvjht/1), Version 1, 16 April 2021.
- **License**: Creative Commons Attribution 4.0 International ([CC BY 4.0](https://creativecommons.org/licenses/by/4.0/))
- **Source**: Acquired from a multispecialty hospital in India.
- **Permitted Use**: Academic research, adaptation, sharing with attribution.
- **Attribution Note**: Retain DOI reference and CC BY 4.0 license notice.

---

## 2. NidaanKosha-100k-V1.0 (Eka Care)
- **Citation**:
  > Eka Care MedAI Team. *NidaanKosha: A Comprehensive Laboratory Investigation Dataset of 100,000 Indian Subjects with 6.8 Million+ Readings.* Hugging Face Datasets: `ekacare/NidaanKosha-100k-V1.0`, 2024. Available from [AIKosh](https://aikosh.indiaai.gov.in/home/datasets/details/nidaankosha_100k_v1.html) and [Hugging Face](https://huggingface.co/datasets/ekacare/NidaanKosha-100k-V1.0).
- **License**: Creative Commons Attribution-ShareAlike 4.0 International ([CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/))
- **Source**: De-identified laboratory investigation reports collected through Eka Care's Personal Health Record applications across diverse diagnostic facilities in India, parsed via Parrotlet vision LLM.
- **Permitted Use**: Research and development with attribution and share-alike terms.
- **Attribution Note**: Explicit attribution to Eka Care and IndiaAI.

### 2.1 Derived Demo Sample (`data/demo/labs_cardiometabolic_sample.parquet`)
- **Origin**: Derived directly from `ekacare/NidaanKosha-100k-V1.0` (sampling 25,000 cardiometabolic test readings with fixed random seed `42`).
- **License**: Creative Commons Attribution-ShareAlike 4.0 International ([CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/)).
- **ShareAlike Notice**: Under CC BY-SA 4.0 terms, any derivative works, adaptations, or downstream distributions must retain this attribution to Eka Care & IndiaAI and must be shared under identical CC BY-SA 4.0 licensing.

---

## 3. Eka Clinical Note Generation Dataset (Pending User Auth / Consent)
- **Citation**:
  > Eka Care MedAI Team. *Clinical Note Generation Dataset: 156 Transcribed Medical Conversations with Entity-Level Structuring.* Hugging Face Datasets: `ekacare/clinical_note_generation_dataset`, 2024. Available from [AIKosh](https://aikosh.indiaai.gov.in/home/datasets/details/eka_clinical_note_generation_dataset.html) and [Hugging Face](https://huggingface.co/datasets/ekacare/clinical_note_generation_dataset).
- **License**: MIT License (listed on Hugging Face / AIKosh page)
- **Source**: Doctor-patient consultation transcripts across English, Hindi, and Marathi with standardized clinical note annotations (symptoms, diagnoses, medications, labs).
- **Access Status**: Gated on Hugging Face / AIKosh (requires user account login and terms acceptance).

---

## 4. Eka Medical History Summarisation Dataset (Pending User Auth / Consent)
- **Citation**:
  > Eka Care MedAI Team. *Medical History Summarisation Dataset: 58 Longitudinal Patient Journeys.* Hugging Face Datasets: `ekacare/ekacare_medical_history_summarisation`, 2024. Available from [AIKosh](https://aikosh.indiaai.gov.in/home/datasets/details/ekacare_medical_history_summarisation.html) and [Hugging Face](https://huggingface.co/datasets/ekacare/ekacare_medical_history_summarisation).
- **License**: MIT License (listed on Hugging Face / AIKosh page)
- **Access Status**: Gated on Hugging Face / AIKosh.

---

## 5. Eka Medical Records Parsing Validation Set (Pending User Auth / Consent)
- **Citation**:
  > Eka Care MedAI Team. *Medical Records Parsing Validation Set: 288 Images of Lab Reports and Prescriptions.* Hugging Face Datasets: `ekacare/medical_records_parsing_validation_set`, 2024. Available from [AIKosh](https://aikosh.indiaai.gov.in/home/datasets/details/eka_medical_records_parsing_validation_set.html) and [Hugging Face](https://huggingface.co/datasets/ekacare/medical_records_parsing_validation_set).
- **License**: MIT License (listed on Hugging Face / AIKosh page)
- **Access Status**: Gated on Hugging Face / AIKosh.

---

## 6. InfoBayAI Echocardiography Reports (Optional / Gated)
- **Citation**:
  > InfoBay AI. *Echocardiography Reports Dataset.* Hugging Face Datasets: `InfoBayAI/Echocardiography-Reports-Dataset`, 2024. Available from [Hugging Face](https://huggingface.co/datasets/InfoBayAI/Echocardiography-Reports-Dataset). Contact: `datareq@infobay.ai`.
- **License**: Custom Research Terms (User agreement required).
- **Limitation**: Country of origin not confirmed as Indian; must be flagged as non-Indian / unverified origin.

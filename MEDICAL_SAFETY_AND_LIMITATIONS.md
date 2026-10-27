# MEDICAL_SAFETY_AND_LIMITATIONS.md
**Date:** 2026-10-02 12:56:57

## Medical Safety

SWASTHYAGUIDE 2.0 is a **research prototype** for health **risk assessment**. It is **NOT** a medical device, diagnostic system, or substitute for clinical judgment.

**Prohibited Claims:** No diagnosis, guaranteed accuracy, clinical validation claim for general use, doctor replacement, treatment/medication recommendations, or emergency diagnosis.

**Appropriate Language:** "Model-estimated risk" (not "you have heart disease"). Distinguish MODEL OUTPUT from CLINICAL DIAGNOSIS.

**Emergency Guidance:** For severe chest pain, severe shortness of breath, fainting, or other medical emergencies, call emergency services immediately. The model must not attempt to assess medical emergencies.

## Model Limitations

- Based on established UCI dataset (cohort-dependent)
- Holdout size finite; no established external clinical validation across diverse populations
- Potential cohort shift; known missingness considerations
- Three models near-interchangeable on discrimination; calibration varies
- Not a clinical diagnostic device; should not be represented as clinically validated


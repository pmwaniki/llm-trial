# Treatment failure analyses (main_analysis.R)

Summary
-------
This script performs the primary analyses for the trial, focusing on the primary outcome "treatment failure" by study arm. It includes descriptive tables, multilevel frequentist models, and Bayesian multilevel models (brms), plus forest/summary plots and saved HTML/PNG outputs.

Files & key symbols
-------------------
- Script: [main_analysis.R](main_analysis.R) — primary analysis script.
- Environment: [.env](.env) — contains OUTPUT_FOLDER, DATA_FOLDER, RESULTS_FOLDER used by the script.
- Main data object loaded in the script: [`main_analysis.outcome_data`](main_analysis.R).


Prerequisites
-------------
- R >= 4.5 and the packages used at the top of the script: arrow, dplyr, gtsummary, gt, lme4, stringr, janitor, performance, brms, tidybayes, ggplot2, ggthemes, ggridges, glue, readxl, tidyr.
- Working .env with correct paths (see [.env](.env)).
- Input files (examples referenced in the script):
  - outcome data: file.path(results_folder, 'outcome_data_cleaned.parquet')
  - clinician and operational files loaded from DATA_FOLDER / RESULTS_FOLDER as in [main_analysis.R](main_analysis.R).

Quick start
-----------
1. Ensure environment variables are set in [.env](.env):
   - OUTPUT_FOLDER, DATA_FOLDER, RESULTS_FOLDER
2. From project root run:
   ```sh
   Rscript main_analysis.R
   ```
   or open and run interactively within R / RStudio.

What the script does (high level)
--------------------------------
- Loads and merges:
  - cleaned outcome data into [`main_analysis.outcome_data`](main_analysis.R),
  - clinician metadata and operational timestamps,
  - diagnosis recodes from results files.
- Re-codes covariates (age groups, day/night, etc.) and builds descriptive tables and plots:
  - Baseline tables and prevalence maps saved to OUTPUT_FOLDER (HTML/PNG).
- Fits frequentist multilevel logistic regression models (glmer) for the primary outcome (treatment failure) and co-primary outcomes; saves summary tables:
  - primary multilevel model outputs and per-protocol variant.
- Fits Bayesian multilevel model with brms (`[`main_analysis.model_brms`](main_analysis.R)`) and computes:
  - posterior predicted risks per arm,
  - posterior risk difference (median and 95% CrI),
  - saves `predicted_risk_brms.html` and density plot of posterior risk difference.
- Fits a brms random-slope model (`[`main_analysis.mod_random_slope`](main_analysis.R)`) and produces a forest-style plot `meta_analysis_forest_plot.png`.

Key outputs (saved to OUTPUT_FOLDER)
------------------------------------
- baseline_characteristics.html
- clinician_characteristics.html
- Prevalence of diagnoses by hospital.png
- diagnoses_by_study_arm.html
- model_primary_multilevel.html
- model_primary_multilevel_per_protocol.html
- model_adjusted_multilevel.html
- model_coprimary_multilevel.html
- predicted_risk_brms.html
- meta_analysis_forest_plot.png

Notes & pointers
----------------
- Posterior summaries and transformation to ORs are computed from [`main_analysis.model_brms`](main_analysis.R).
- Random-slope hospital effects and forest data come from [`main_analysis.mod_random_slope`](main_analysis.R).
- If running interactively, set options for brms (cores, chains) appropriately for available CPU / RAM.
- For reproducibility, the script sets seeds in brms model calls.

# Clinical documentation analyses (clinical_documentation.R)
---------------------------------------------------------
Purpose
- Summarises and models clinician documentation quality measured via item-level ratings (min_score) across three domains: appropriateness of diagnosis, comprehensiveness of documentation, and appropriateness of proposed treatment plan.

Inputs
- results_folder/outcome_data_cleaned.parquet (merged to get arm, clinician and hospital ids)
- data_folder/CLINICAL_DOCUMENTATION_COLLATED.parquet (item ratings)

What it does
- Merges clinical documentation ratings with trial metadata (arm, clinician_num, hospital_num).
- Recodes numeric min_score to an ordered factor y_factor (1 = Strongly Disagree ... 5 = Strongly Agree).
- Produces stacked bar charts of rating distributions by study arm and domain.
- Exports wide tables of percentage ratings:
  - clinical_documentation_ratings.docx
  - clinical_documentation_ratings_by_arm.docx
- Fits an ordinal mixed-effects model (clmm from the ordinal package):
  - model: y_factor ~ arm * variable + (1 | clinician_num) + (1 | hospital_num)
  - Tests the intervention effect overall and within domains (interaction).
- Computes odds ratios and 95% CI for arm (and arm-by-domain interactions) and saves:
  - clinical_documentation_ordinal_logistic.html
- Saves a stacked-plot PNG:
  - clinical_documentation_ratings.png
- Also saves a combined summary HTML of ratings by arm:
  - clinical_documentation_ratings_summary by arm.html

Outputs (saved to OUTPUT_FOLDER)
- clinical_documentation_ratings.png
- clinical_documentation_ratings.docx
- clinical_documentation_ratings_by_arm.docx
- clinical_documentation_ordinal_logistic.html
- clinical_documentation_ratings_summary by arm.html

Notes
- Requires the ordinal package (clmm) in addition to packages used elsewhere.
- Reported OR > 1 indicates higher odds of a greater (more favorable) rating in the Intervention arm.


# Clinical safety (clinical_safety.R)
------------------------------------
Purpose
- Summarises clinician safety assessments and adherence for sampled cases and visualises relationships between assessment, adherence and justification.

Inputs
- data_folder/CLINICAL_SAFETY_COLLATED.parquet (safety ratings)
- .env paths to locate OUTPUT_FOLDER, DATA_FOLDER, RESULTS_FOLDER

What it does
- Loads safety data and recodes factor levels for:
  - assess_advice (ordered safety assessment)
  - adherence (Yes / Partially / No)
  - justification (Yes / No)
- Builds node and link tables and renders a Sankey diagram showing flows across:
  assess_advice → adherence → justification
- Saves Sankey outputs:
  - Sankey diagram of clinical safety.html
  - Sankey diagram of clinical safety.pdf (via webshot2)
- Produces a summary table (tbl_summary) of the three variables and saves:
  - table_clinical_safety.html

Outputs (saved to OUTPUT_FOLDER)
- Sankey diagram of clinical safety.html
- Sankey diagram of clinical safety.pdf
- table_clinical_safety.html

Notes
- Requires networkD3 and webshot2 (and a headless webshot prerequisite) to produce the PDF.
- Sankey nodes are created from observed factor levels; ensure levels are correct before plotting.
- Table lists counts/percentages by category and is saved as an HTML gt table.

# Patient satisticaction (patient_satisfaction.R)
------------------------------------------------
Purpose
- Summarises consultation duration and patient-reported satisfaction items, compares by study arm, and fits mixed / ordinal mixed-effects models to estimate intervention effects.

Inputs
- results_folder/outcome_data_cleaned.parquet (trial metadata: arm, clinician/hospital ids)
- data_folder/user_satisfaction.pq (patient satisfaction items)
- data_folder/Operational_data.parquet (consultation duration)

What it does
- Calculates consultation length summary for single-clinician encounters and compares by arm:
  - Median/IQR table and Wilcoxon test saved as consultation_length_summary.html
  - Linear mixed-effects model (lmer) with random intercepts for clinician and hospital; results saved as consultation_length_mixed_effects.html
- Processes patient satisfaction items (steps_explained, consult_comprehensive, concerns_addressed, recommend_others):
  - Joins to trial metadata, filters to single-clinician encounters, recodes responses to ordered factors 1–5.
  - Produces descriptive summary table saved as patient_satisfaction_summary.html.
  - Creates stacked bar plot of rating distributions by arm and domain, saved as patient_satisfaction_ratings.png.
- Fits an ordinal mixed-effects model (clmm) with interaction:
  - model: y_factor ~ arm * variable + (1 | clinician_num) + (1 | hospital_num)
  - Computes odds ratios and 95% CIs for intervention overall and within items (via combined coef/covariance), saved as patient_satisfaction_ordinal_logistic.html
- Computes an overall satisfaction score per record (mean across items), rounds to nearest integer and fits clmm:
  - model: score_factor ~ arm + (1 | clinician_num) + (1 | hosp_id)
  - Results saved as patient_satisfaction_overall_ordinal_logistic.html

Outputs (saved to OUTPUT_FOLDER)
- consultation_length_summary.html
- consultation_length_mixed_effects.html
- patient_satisfaction_summary.html
- patient_satisfaction_ratings.png
- patient_satisfaction_ordinal_logistic.html
- patient_satisfaction_overall_ordinal_logistic.html

Notes
- Requires packages: ordinal (clmm), lme4, gtsummary, gt and tidyverse family.
- Analyses restrict to single-clinician consultations where specified.
- Ordinal model OR > 1 indicates higher odds of a more favorable rating in the Intervention arm.

# Sentinel conditions (malnutrition.R, hypertension.R, antimalarials.R, diabetes.R, antibiotics.R)

## Sentinel conditions analyses

Purpose
- Evaluate diagnosis, testing, referral and treatment for key sentinel conditions: malnutrition, hypertension, malaria (antimalarial prescribing), diabetes, and antibiotic prescribing.

Inputs
- outcome_data_cleaned.parquet (trial metadata: arm, hospital/clinician ids)
- Per-condition recode files (saved in RESULTS_FOLDER), e.g.:
  - "Malnutrition recode - alibayram__medgemma:27b.xlsx"
  - "Hypertension recode - alibayram__medgemma:27b.xlsx"
  - "Anti-malarial recode - alibayram__medgemma:27b.xlsx"
  - "Diabetes recode - alibayram__medgemma:27b.xlsx"
  - (antibiotics recode file name as in RESULTS_FOLDER)

What each script does (high level)
- malnutrition.R
  - Defines malnutrition using documented MUAC/colour flags or WHZ.
  - Computes identification and referral rates and fits multilevel logistic regression models (hospital and clinician random effects) for referral/identification.
  - Saves summary tables and model HTML outputs.
- hypertension.R
  - Identifies new and chronic hypertension, tabulates treatment for new cases, fits multilevel logistic regression models for incidence and treatment, and saves summaries and model outputs.
- antimalarials.R
  - Joins antimalarial prescription recode to trial metadata, computes testing and correctness (e.g., antimalarial prescribed without positive test), and fits multilevel logistic regression for incorrect prescribing.
  - Saves summary tables and model HTML outputs.
- diabetes.R
  - Flags patients at risk, tested, diagnosed, and treated; fits multilevel and logistic models for each stage and saves result pages.
- antibiotics.R
  - (Same pattern) Summarises antibiotic prescribing vs diagnosis/testing and fits regression models to assess inappropriate prescribing; saves summaries and model outputs.

Outputs
- Per-script summary HTMLs and model result HTMLs saved to OUTPUT_FOLDER (see filenames in each script).
- Typical file examples: "malnutrition_referred_model_results.html", "antimalarial_summary.html", "diabetes_summary.html", etc.

Notes & usage
- Run scripts individually, e.g.:
  Rscript malnutrition.R
- Scripts assume .env is present and exports OUTPUT_FOLDER, DATA_FOLDER, RESULTS_FOLDER.
- Analyses generally restrict to single‑clinician encounters where appropriate and use glmer for multilevel
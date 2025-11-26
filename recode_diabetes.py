import json
import os
from pathlib import Path

import pandas as pd
import numpy as np
from dotenv import load_dotenv
from pydantic import BaseModel, Field
import re
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
import pygrowup
from tqdm import tqdm

load_dotenv(".env")

data_folder = Path(os.getenv("DATA_FOLDER"))
results_folder = Path(os.getenv("RESULTS_FOLDER"))

outcome_data = pd.read_parquet(data_folder / "outcome_data.parquet")
outcome_data = outcome_data.loc[outcome_data['age'] >=18 * 12].copy()
clinical_data = pd.read_excel(data_folder / "clinical_data-PATH.xlsx")
clinical_data['visit_number'] = clinical_data['visit_number'].map(lambda x: re.sub(r'[^0-9]', '', str(x)))
outcome_data['visit_number'] = outcome_data['visit_number'].map(lambda x: re.sub(r'[^0-9]', '', str(x)))

outcome_data = pd.merge(outcome_data, clinical_data, how='left', on='visit_number')
outcome_data['diabetic'] = outcome_data['ChronicIllness'].str.replace("Chronic Illness:\n", "").str.lower().str.contains("diabet")
outcome_data=outcome_data.loc[outcome_data['diabetic'] == False].copy()
# weight for height z-score


model_name = "alibayram/medgemma:27b"


class Schema(BaseModel):
    """Schema for diagnoses and treatment of type 2 diabetes mellitus"""
    diabetes_at_risk: bool = Field(False, description="Patient is at risk of type 2 diabetes mellitus")
    risk_reason:str = Field("", description="Reason for indicating the patient is at risk of type 2 diabetes mellitus")
    diabetes_test:bool=Field(False, description="HbA1c or random/fasting blood sugar testing")
    diabetes_positive:bool=Field(False, description="HbA1c or random/fasting blood sugar positive")
    diabetes_treated:bool=Field(False, description="Patient treated for diabetes")
    diabetes_chronic:bool=Field(False, description="Patient is a known type 2 diabetic")


model = ChatOllama(model=model_name, temperature=0.)
# model with structured output
structured_model = model.with_structured_output(Schema)

sys_prompt = """
You are a professor of medicine working in a primary care setting in Kenya.
Your task is to review clinical notes of adults in an outpatient department for proper diagnosis and treatment of type 2 diabetes mellitus.

From the clinical notes:
- Identify if a patient is at risk of diabetes mellitus: diabetes_at_risk
- State the reason for indicating the patient is at risk: risk_reason
- Identify whether diabetes test was ordered or done (HbA1c or random/fasting blood sugar testing): diabetes_test
- Patient has type 2 diabetes according to HbA1c or random/fasting blood sugar testing results: diabetes_positive
- Diabetes treatment was initiated: diabetes_treated
- Whether a patient is a known type 2 diabetic (eg known diabetes mellitus): diabetes_chronic

Diabetes risk (diabetes_at_risk) is defined as:
- Adults who are overweight/obese (BMI ≥ 25kg/m2 or ≥ 23kg/m2 in Asians) who have ≥1 of the following:
    - First degree relative with diabetes
    - History of cardiovascular disease (CVD)
    - Hypertension
    - HDL cholesterol < 35mg/dl and/or triglyceride level > 250mg/dl, 
    - Women with polycystic ovary disease
    - <150 hours physical activity each week 
- OR patient with prediabetes (HbA1c  ≥ 5.7%), impaired glucose tolerance, or impaired fasting glucose
- OR women with a previous diagnosis of gestational diabetes (GDM)
- OR patient with a diagnosis of HIV or TB
- ALL patients 35 or older should be classified as at risk
Do not include patients where diabetes is indicated as a known chronic condition


Diabetes diagnosis (diabetes_positive) is defined as:
- HbA1c of  ≥ 6.5% OR
- Fasting blood sugar of ≥ 7.0 mmol/L on >1 occasion OR
- Random blood sugar of  ≥ 11.1 mmol/L in symptomatic patients OR
- 2hr OGTT plasma glucose ≥ 11.1 mmol/L

Diabetes treatment (diabetes_treated) if patient is placed on appropriate treatment according to the Kenyan clinical guidelines (eg metformin)



- Output MUST be a single JSON object with ONLY these keys as shown:
{{
  "diabetes_at_risk": true/false,
  "risk_reason: "brief reason for indicating the patient is at risk of diabetes mellitus (from the criteria provided)"
  "diabetes_test": true/false,
  "diabetes_positive": true/false,
  "diabetes_treated": true/false
  "diabetes_chronic": true/false,
}}
"""

# use prompt template as before
prompt_template = ChatPromptTemplate.from_messages(
    [("system", sys_prompt), ("user", "{text}")]
)

extracted_diagnoses = {}
error_visits = []
for _, row in tqdm(outcome_data.iterrows(), total=outcome_data.shape[0]):
    text = row.get('clinical_documentation')
    # ensure consistent key for missing text
    key = row.get('record_id')
    if not text or pd.isna(text):
        extracted_diagnoses[key] = None
        continue

    prompt_text = text
    prompt = prompt_template.invoke({'text': prompt_text})
    result = structured_model.invoke(prompt)

    # robust parsing + logging
    try:
        # structured_model may return a pydantic model-like object
        if hasattr(result, "dict"):
            result_dict = result.model_dump()
        elif isinstance(result, dict):
            result_dict = result
        else:
            # fallback: try extracting raw content and parse JSON
            raw = getattr(result, "content", None) or str(result)
            # strip triple backticks if present
            raw = re.sub(r'```(?:json)?', '', raw).strip('` \n')
            result_dict = json.loads(raw)
        # attach metadata

        result_dict['clinical_documentation'] = row.get('clinical_documentation')
        result_dict['record_id'] = row.get('record_id', key)
        extracted_diagnoses[result_dict['record_id']] = result_dict
    except Exception as e:
        # save raw output for debugging
        raw = getattr(result, "content", None) or str(result)
        print(f"PARSE ERROR record={key} error={e}")
        print("RAW OUTPUT:", raw[:1000])
        error_visits.append(key)
        extracted_diagnoses[key] = {"parse_error": True, "raw": raw, "record_id": key}

system = pd.DataFrame(extracted_diagnoses.values())
system2 = pd.merge(system, outcome_data[['record_id', 'LabTest', 'Rx']], how="left", on='record_id')
system2.to_excel(results_folder / f'Diabetes recode - {model_name.replace("/", "__")}.xlsx', index=False)



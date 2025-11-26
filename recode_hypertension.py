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
outcome_data['hypertension'] = outcome_data['ChronicIllness'].str.replace("Chronic Illness:\n", "").str.lower().str.contains("hyperten")
outcome_data=outcome_data.loc[outcome_data['hypertension'] == False].copy()
# weight for height z-score


model_name = "alibayram/medgemma:27b"


class Schema(BaseModel):
    """Schema for diagnoses and treatment of hypertension"""
    hypertension_new: bool = Field(False, description="Has new hypertension")
    hypertension_reason: str = Field("None", description="Reason for indicating the patient has hypertension")
    hypertension_chronic:bool=Field(False, description="Identified has having chronic hypertension")
    hypertension_treated:bool=Field(False, description="Patient treated for hypertension")
    bp_second:str=Field("", description="Secondary blood pressure if measured")


model = ChatOllama(model=model_name, temperature=0.)
# model with structured output
structured_model = model.with_structured_output(Schema)

sys_prompt = """
You are a professor of medicine working in a primary care setting in Kenya.
Your task is to review clinical notes of adults in an outpatient department for proper diagnosis and treatment of hypertension.

From the clinical notes:
- Identify if a patient has a new diagnoses of hypertension: hypertension_new
- State the reason for indicating the patient has a new diagnoses of hypertension: hypertension_reason
- State if the patient has prior diagnoses of hypertension: hypertension_chronic

- Hypertension treatment was initiated: hypertension_treated
- Value of repeat BP if present

New diagnosis is defined as:
 - Two systolic blood pressure (BP) readings >140mmHg, or
 - Diastolic blood pressure readings >90mmHg on two separate encounters OR,
 - A patient who presents with hypertensive urgency or emergency at first contact without further confirmatory readings 


Prior diagnoses is defined as:
 - Hypertension indicated as a chronic condition
 - Patient currently taking hypertension medication

Initiation of hypertension medication is defined as:
 - initiated on appropriate antihypertensive treatment, adapted in line with suggestions represented by WHO guidelines. 
 - Appropriate treatment for all patients with a diagnosis of hypertension includes guidance on lifestyle modification and hypertension education. Evidence of lifestyle modification advice given or the provision of multimedia content focused on lifestyle modification, by the clinician.
 - If patients have 2 or more BPs >140/90mmHg also initiate antihypertensive medications, either calcium channel blockers (e.g. Amlodipine) or angiotensin receptor blocker (Losartan) 

- Output MUST be a single JSON object with ONLY these keys as shown:
{{
  "hypertension_new": true/false,
  "hypertension_reason: "brief reason for indicating why the patient has new diagnoses of hypertension (eg elevated BP or hypertensive urgency)"
  "hypertension_chronic": true/false,
  "hypertension_treated": true/false
  "bp_second": "second bp measurement (eg 146/73)
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
system2 = pd.merge(system, outcome_data[['record_id', 'doc_bpr', 'Rx']], how="left", on='record_id')
system2.to_excel(results_folder / f'Hypertension recode - {model_name.replace("/", "__")}.xlsx', index=False)


